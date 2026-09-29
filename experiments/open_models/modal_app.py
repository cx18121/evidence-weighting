"""Run open-model experiments on Modal using a pinned Kev checkout."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
COMMIT = 'd32a973cc2375f1c76e898cbf6e759c4499042de'
RATE = 1.95 / 3600
MAX_SECONDS = 12 * 3600
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('git')
         .pip_install('uv==0.9.18')
         .run_commands('git clone https://github.com/jaredpalmer/kev.git /root/kev',
                       f'cd /root/kev && git checkout --detach {COMMIT}',
                       'cd /root/kev && uv sync --frozen --no-dev --python 3.12',
                       'uv pip install --python /root/kev/.venv/bin/python matplotlib==3.10.8')
         .add_local_dir(str(HERE), remote_path='/root/experiments/open_models',
                        ignore=['results/**', 'kev_repo/**', '__pycache__/**', '*.pyc'])
         .add_local_file(str(HERE.parent / 'evidence_count' / 'generate.py'),
                         remote_path='/root/experiments/evidence_count/generate.py'))
app = modal.App('evidence-weighting-reproduction', image=image)
hf_cache = modal.Volume.from_name('evidence-weighting-hf', create_if_missing=True)
output = modal.Volume.from_name('evidence-weighting-output', create_if_missing=True)
hf_secret = [modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']})] if os.environ.get('HF_TOKEN') else []
ROOT = '/root/experiments/open_models'
RESULTS = '/vol/out/results'
PY = '/root/kev/.venv/bin/python'
QUOTAS = {'smoke': 240, 'behavior08': 1950, 'behavior4b': 2400,
          'behaviorq4': 2400, 'behavior7b': 2800, 'probe': 2400,
          'padded': 900, 'patch': 1200, 'generate': 120, 'baseline': 780,
          'train08': 1900, 'train4b': 1900, 'evaluate': 900}


def command(step, model, seed_count, pairs, steps, curriculum=''):
    if step == 'smoke': return [PY, 'behavior.py', '--model', 'smoke', '--out', RESULTS, '--limit', '1']
    if step == 'behavior': return [PY, 'behavior.py', '--model', model, '--out', RESULTS,
                                    '--seed-count', str(seed_count)]
    if step == 'probe': return [PY, 'probe.py', '--model', model, '--out', RESULTS, '--ensure-tokens']
    # patch v2: soft deadline inside the quota; exit 75 = partial (resumable, no done marker)
    if step == 'padded': return [PY, 'patch.py', '--model', model, '--out', RESULTS, '--mode', 'padded',
                                 '--seed-count', str(seed_count), '--max-seconds', str(QUOTAS['padded'] - 120)]
    if step == 'patch': return [PY, 'patch.py', '--model', model, '--out', RESULTS, '--mode', 'patch',
                                '--pairs', str(pairs), '--seed-count', str(seed_count),
                                '--max-seconds', str(QUOTAS['patch'] - 150)]
    if step == 'generate': return [PY, 'kev_train/generate.py', '--out', RESULTS+'/training', '--total', '4000']
    if step == 'baseline': return [PY, 'kev_train/evaluate.py', '--model',
                                    'jaredpalmer/kev-4b' if model == '4b' else 'jaredpalmer/kev-0.8b',
                                    '--out', RESULTS+'/trained_evaluation', '--seed-count', str(seed_count)]
    if step == 'train': return [PY, 'kev_train/train.py', '--size', model,
                                  '--curriculum', curriculum, '--steps', str(steps),
                                  '--data', RESULTS+'/training', '--out', RESULTS+'/trained']
    if step == 'evaluate': return [PY, 'kev_train/evaluate.py', '--model',
                                     RESULTS+'/trained/'+model+'-'+curriculum,
                                     '--out', RESULTS+'/trained_evaluation', '--seed-count', str(seed_count)]
    raise ValueError(step)


def costs(root):
    p = root / 'costs.jsonl'
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []


def markdown(rows):
    lines = ['# E5 Modal GPU cost (spec: v3.4)', '',
             f'L40S estimate $1.95/hour, limit {MAX_SECONDS:,} GPU seconds (~${MAX_SECONDS*RATE:.2f}); '
             'measured inside function; cold-start and image build not included. Check actual Modal billing.', '',
             '| Step | GPU seconds (approx.) | Estimated USD | Status |', '|---|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['step']} | {r['seconds']:.1f} | ${r['usd']:.3f} | {r['status']} |")
    used = sum(x['seconds'] for x in rows)
    lines.append(f'\nTotal: {used:.1f} seconds, ${used*RATE:.3f}; remaining ceiling {max(0,MAX_SECONDS-used):.1f} seconds.')
    return '\n'.join(lines)+'\n'


def run_job(step: str, model: str = '', seed_count: int = 3, pairs: int = 6,
            steps: int = 300, curriculum: str = '', gpu: bool = True):
    from pathlib import Path
    import os, json, time
    output.reload(); hf_cache.reload()
    root = Path(RESULTS); root.mkdir(parents=True, exist_ok=True)
    allowed = {'smoke': ('smoke',), 'behavior': ('kev08','kev4','qwen4','qwen','llama'),
               'probe': ('kev08','kev4','qwen4','qwen','llama'), 'patch': ('kev08','kev4','qwen4','qwen','llama'),
               'padded': ('kev08','kev4','qwen4','qwen','llama'),
               'generate': ('',), 'baseline': ('','4b'), 'train': ('08b','4b'), 'evaluate': ('08b','4b')}
    if model not in allowed[step] or curriculum not in ('', 'varied', 'fixed8'):
        raise ValueError('unrecognized step/model/curriculum')
    key = f'{step}-{model}-{curriculum}-s{seed_count}-p{pairs}-t{steps}'
    marker = root/'steps'/f'{key}.done.json'; marker.parent.mkdir(exist_ok=True)
    if marker.exists(): return {'step': key, 'status': 'skipped_finished'}
    quota_key = ('behavior08' if model=='kev08' else 'behavior4b' if model=='kev4' else
                 'behaviorq4' if model=='qwen4' else 'behavior7b') if step=='behavior' else (
                 'train08' if model=='08b' else 'train4b') if step=='train' else step
    quota = QUOTAS[quota_key]
    used = sum(x['seconds'] for x in costs(root)); remaining = MAX_SECONDS - used
    if gpu and remaining < quota:
        return {'step': key, 'status': 'budget_blocked', 'remaining_seconds': remaining, 'required': quota}
    env = os.environ.copy(); env.update({'HF_HOME': '/vol/hf', 'HF_HUB_CACHE': '/vol/hf/hub',
                                          'E5_CURRICULUM': curriculum, 'PYTHONUNBUFFERED':'1'})
    argv = command(step, model, seed_count, pairs, steps, curriculum)
    log = root/'logs'/f'{key}.log'; log.parent.mkdir(exist_ok=True)
    start = time.monotonic(); status = 'error'; error = ''
    try:
        with log.open('a') as f:
            f.write(f'\nspec: v3.4 argv={argv!r} limit_seconds={quota}\n'); f.flush()
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            try: proc.wait(timeout=quota)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(); raise TimeoutError(f'{key} exceeded {quota}s')
            if proc.returncode == 75 and step in ('padded', 'patch'):
                status = 'partial'   # soft deadline hit; outputs are resumable; no marker so a rerun continues
                raise StopIteration
            if proc.returncode: raise RuntimeError(f'{key} exit={proc.returncode}; inspect {log}')
        status = 'ok'
        marker.write_text(json.dumps({'spec':'v3.4','step':key,'argv':argv,'seconds':time.monotonic()-start}))
    except StopIteration:
        error = 'soft deadline reached; rerun the same command to resume'
    except Exception as ex:
        error = str(ex)
    finally:
        elapsed = time.monotonic()-start
        row = {'spec':'v3.4','step':key,'seconds':elapsed if gpu else 0.,
               'wall_seconds':elapsed, 'usd':elapsed*RATE if gpu else 0., 'status':status,
               'error':error,'wall_time_note':'excludes image build / container startup; CPU jobs GPU seconds=0'}
        with (root/'costs.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        (root/'costs.md').write_text(markdown(costs(root)))
        output.commit(); hf_cache.commit()
    return row


@app.function(gpu='L40S', timeout=3750, volumes={'/vol/hf': hf_cache, '/vol/out': output},
              secrets=hf_secret, max_containers=1)
def execute(step: str, model: str = '', seed_count: int = 3, pairs: int = 6,
            steps: int = 300, curriculum: str = ''):
    return run_job(step, model, seed_count, pairs, steps, curriculum, gpu=True)


@app.function(cpu=4, memory=8192, timeout=2700, volumes={'/vol/hf': hf_cache, '/vol/out': output},
              max_containers=1)
def execute_cpu(step: str, model: str = '', seed_count: int = 3, pairs: int = 6,
                steps: int = 300, curriculum: str = ''):
    if step not in ('probe','generate'): raise ValueError('CPU step must be probe or generate')
    return run_job(step, model, seed_count, pairs, steps, curriculum, gpu=False)


def invoke(step, model='', **kw):
    result = (execute_cpu if step in ('probe','generate') else execute).remote(step, model, **kw)
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] == 'partial':
        print(f'WARNING {step} {model}: partial (soft deadline); rerun intervene to resume before interpreting', flush=True)
        return result
    if result['status'] not in ('ok','skipped_finished'):
        raise RuntimeError(f'E5 halted: {result}; sync and inspect the logs before resuming')
    return result


@app.local_entrypoint()
def main(smoke_only: bool = False, seed_count: int = 3, pairs: int = 6,
         train_steps: int = 300):
    # Ephemeral Modal run, not a deployed service. Check pricing before invoking.
    invoke('smoke','smoke')
    print('Modal smoke passed (unless already marked finished).', flush=True)
    if smoke_only: return
    model8 = 'qwen'
    if os.environ.get('HF_TOKEN'):
        import urllib.request
        req = urllib.request.Request('https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json',
                                     headers={'Authorization': 'Bearer '+os.environ['HF_TOKEN']}, method='HEAD')
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status == 200: model8 = 'llama'
        except Exception: print('Llama gated/token inaccessible; using Qwen2.5-7B.', flush=True)
    for model in ('kev08','kev4','qwen4',model8):
        invoke('behavior', model, seed_count=seed_count)
        invoke('probe', model)
    # Probe.json is on the output volume. Inspect the actual behavior and choose two manually
    # before interventions: there is no automatic scientific choice based on unreviewed metrics.
    print('E5a behavior/probes complete. Sync and review slopes before patching; '
          'then run --patch-models with the two behavior-confirmed model keys.', flush=True)


@app.function(cpu=4, memory=8192, timeout=3600, volumes={'/vol/hf': hf_cache, '/vol/out': output}, max_containers=2)
def reprobe_job(model: str):
    """CPU-only probe v2 (tokens.json if missing, then probe.py). Writes only new per-model files and its own log;
    never touches costs.jsonl/costs.md or step markers, so it is safe to run while a GPU app writes the same Volume."""
    output.reload(); hf_cache.reload()
    if model not in ('kev08','kev4','qwen4','qwen','llama'): raise ValueError(model)
    root = Path(RESULTS); log = root/'logs'/f'probe_v2-{model}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update({'HF_HOME': '/vol/hf', 'HF_HUB_CACHE': '/vol/hf/hub', 'PYTHONUNBUFFERED': '1',
                                          'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4'})
    argv = [PY, 'probe.py', '--model', model, '--out', RESULTS, '--ensure-tokens']
    start = time.monotonic()
    with log.open('a') as f:
        f.write(f'\nspec: v3.4 probe v2 argv={argv!r}\n'); f.flush()
        rc = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=3300).returncode
    output.commit()
    return {'step': f'probe_v2-{model}', 'status': 'ok' if rc == 0 else 'error', 'returncode': rc,
            'wall_seconds': time.monotonic() - start, 'log': str(log)}


@app.local_entrypoint(name='reprobe')
def reprobe(models: str = 'kev08'):
    """CPU-only: modal run modal_app.py::reprobe --models kev08,kev4"""
    for result in reprobe_job.map(models.split(',')):
        print(json.dumps(result, indent=2), flush=True)


@app.local_entrypoint(name='intervene')
def intervene(models: str, pairs: int = 18, seed_count: int = 3, train_steps: int = 300,
              train_4b: bool = False, resume_training: bool = False):
    names = models.split(',')
    if len(names) != 2 or len(set(names)) != 2 or not set(names) <= {'kev08','kev4','qwen4','qwen','llama'}:
        raise ValueError('Provide exactly two comma-separated model keys after behavior review')
    # On a training retry, require completed patch markers rather than repeating the CPU probes.
    for name in names:
        if not resume_training:
            invoke('padded', name, seed_count=seed_count)
            result = reprobe_job.remote(name); print(json.dumps(result, indent=2), flush=True)
            if result['status'] != 'ok': raise RuntimeError(f'probe v2 failed for {name}; see {result["log"]}')
        patch = invoke('patch', name, pairs=pairs, seed_count=seed_count)
        if resume_training and patch['status'] != 'skipped_finished':
            raise RuntimeError(f'Expected an existing patch marker for {name} before training resume')
    invoke('generate')
    invoke('baseline',seed_count=seed_count)
    if train_4b: invoke('baseline','4b',seed_count=seed_count)
    for size in ('08b','4b') if train_4b else ('08b',):
        for curriculum in ('varied','fixed8'):
            invoke('train',size,steps=train_steps,curriculum=curriculum)
            invoke('evaluate',size,seed_count=seed_count,curriculum=curriculum)
