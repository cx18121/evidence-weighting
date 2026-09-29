"""E5a: local model behavior and per-case, per-layer final-token residual cache."""
import argparse
import gc
import json
import math
from pathlib import Path
import numpy as np
import torch
import prompts
from runtime import load_model, layers_of, layer_hook, score

MODELS = {'smoke': 'Qwen/Qwen2.5-0.5B-Instruct', 'llama': 'meta-llama/Llama-3.1-8B-Instruct', 'mistral7': 'mistralai/Mistral-7B-Instruct-v0.3',
          'qwen': 'Qwen/Qwen2.5-7B-Instruct',
          'qwen4': 'Qwen/Qwen3-4B-Instruct-2507',
          'kev4': 'jaredpalmer/kev-4b', 'kev08': 'jaredpalmer/kev-0.8b',
          'mistral': 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'}


def capture(tok, model, kev, case):
    """Collect embeddings and all layer outputs at final prompt/decide token, in forward order."""
    L = len(layers_of(model, kev)); vectors = [None] * (L + 1)
    with __import__('contextlib').ExitStack() as stack:
        for i in range(L + 1):
            def collect(v, j=i):
                vectors[j] = v.cpu().to(torch.float16).numpy().copy()
            stack.enter_context(layer_hook(model, kev, i, 'residual', collect))
        p = score(tok, model, kev, case)
    if any(x is None for x in vectors):
        raise RuntimeError('a layer was not executed')
    return p, np.stack(vectors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', choices=('cuda', 'mps', 'cpu'), default='cuda')
    ap.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed-count', type=int, default=6, help='budgeted pilot: first N of 6 E1 seeds; mark incomplete')
    ap.add_argument('--numeric', action='store_true', help='optional single-token 0..100 readout, separate file')
    args = ap.parse_args()
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    tok, model, kev = load_model(MODELS[args.model], args.device, args.dtype)
    meta = {'spec': 'v3.4', 'prompt_spec': 'v3.1', 'generator_sha256': prompts.GENERATOR_SHA256,
            'model': MODELS[args.model], 'device': args.device, 'dtype': args.dtype,
            'model_revision': getattr(getattr(model, 'config', None), '_commit_hash', None),
            'tokenizer_revision': getattr(tok, 'init_kwargs', {}).get('_commit_hash'),
            'layers': len(layers_of(model, kev)), 'kind': 'unnormalized layer-output residual; 0=layer0 input',
            'question': prompts.QUESTION}
    meta_path = out / 'meta.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise RuntimeError(f'cache metadata conflict: {meta_path}')
    meta_path.write_text(json.dumps(meta, indent=2))
    filename = 'numeric.jsonl' if args.numeric else 'behavior.jsonl'
    prev = {}
    if (out / filename).exists():
        for line in (out / filename).read_text().splitlines():
            row = json.loads(line); prev[row['id']] = row
    cs = [c for c in prompts.cases() if c['seed_idx'] < args.seed_count]
    if args.limit: cs = cs[:args.limit]
    with (out / filename).open('a') as f:
        for c in cs:
            vec_path = out / 'residual' / (c['id'] + '.npy')
            if c['id'] in prev and (args.numeric or vec_path.exists()):
                continue
            if args.numeric:
                p = score(tok, model, kev, c, numeric=True)
            else:
                p, vectors = capture(tok, model, kev, c)
                vec_path.parent.mkdir(exist_ok=True)
                tmp = vec_path.with_suffix('.tmp')
                with tmp.open('wb') as stream: np.save(stream, vectors, allow_pickle=False)
                tmp.replace(vec_path)
            row = {'spec': 'v3.4', 'id': c['id'], 'seed': c['seed'], 'template': c['template'],
                   'domain': c['domain'], 'r': c['r'], 'n': c['n'], 'k': c['k'], 'share': c['share'],
                   'format': c['format'], 'truth': c['truth'], 'p_yes': p, 'readout': 'kev_noul' if kev else
                   ('restricted_numeric_first_token' if args.numeric else 'Yes_vs_No_single_next_token')}
            f.write(json.dumps(row) + '\n'); f.flush()
            print(c['id'], p, flush=True)
    gc.collect()


if __name__ == '__main__': main()
