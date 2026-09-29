"""Interchange interventions on saved open-model states.

Two GPU modes and one CPU mode:
  --mode padded     forward passes on the length-matched padded listed prompts (state_padded from the E1 generator,
                    n < 64, seeds < --seed-count): residual_padded/<id>.npy and behavior_padded.jsonl, for probe.py.
  --mode patch      interchange interventions (needs probes_v2.npz from probe.py v2). Writes patch_v2.jsonl, then
                    patch_v2_summary.json.
  --mode summarize  recompute patch_v2_summary.json from patch_v2.jsonl (CPU, no model).

Design (Geiger et al. 2021 interchange interventions; Zhang & Nanda 2023 and Heimersheim & Nanda 2024 patching
practice; Makelov et al. 2023 subspace caveat):
  * pairs: receiver and donor differ only in n. Same share, r, format, domain, seed and therefore the same story
    template, report-ID stream and case wording (prompts.matches()). Donor n = 64, receiver n in {8, 16, 32}, share
    != 0.5 (so the normative answers differ), r in {0.52, 0.55} (at r = 0.70 the n = 64 normative log-odds reach 27
    logits). Pairs are stratified over (receiver n, format, r) and drawn from seeds
    not used to fit the probes when available. The reverse direction (receiver n = 64) is run for the two main
    interventions. Positive-control pairs differ only in share (same n; 0.25 vs 0.75), where the output should move.
  * positions: only token positions whose ids are identical in donor and receiver, aligned from the end (the question
    and chat-template suffix; for Kev the question branch). "readout" = the positions the output is read from: the last
    token for LLMs; <decide> plus both </opt> tokens for Kev (its pointer head reads all three).
  * interventions per layer: full residual at readout positions; full residual at the whole common suffix; attention
    output at readout positions (layers with self-attention); count-subspace interchange at the final position
    (h + U U^T (h_donor - h), U = orthonormal basis of the v2 probe directions for log2 n, k and margin at that layer);
    the same with a random 3-D subspace (control); share-subspace on positive-control pairs; steering along the
    normative-log-odds probe direction by 1x and 4x the donor-receiver normative difference in probe units, and a random
    direction of the same norm.
  * metrics: output logit (logit P(YES), unclipped up to 1e-7). shift = patched - receiver. Fraction of the behavioral
    gap closed toward the donor's own output, (patched - receiver) / (donor - receiver), only where |donor - receiver|
    >= GAP_MIN; fraction of the normative gap, shift / (normative_donor - normative_receiver); final-layer probe readout
    of n after the patch (does the patched information propagate?). Pair-bootstrap 95% CIs.
  * sanity: full-readout patch at the last layer must reproduce the donor output (hook check); full-suffix patch at layer
    0 must change nothing (identical tokens).
"""
import argparse
import json
import math
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path
import numpy as np
import prompts

GAP_MIN = 0.25          # logit; below this the donor-receiver behavioral gap is treated as absent
SANITY_TOL = 0.05       # logit; last-layer full-readout patch must reproduce the donor
USED_FRAC = 0.10        # Exploratory threshold as a fraction of the normative gap.
POSCTRL_FRAC = 0.25     # positive control must close at least this fraction of the share gap at some layer
MAX_SUFFIX = 64
SUBSPACE_TARGETS = ('log2_n', 'k', 'margin')


# ----------------------------------------------------------------------------- pairs

def pair_sets(seed_count, n_same=18, n_pos=6, rng_seed=5):
    rng = np.random.default_rng(rng_seed)
    cs = prompts.cases()
    held = [s for s in range(6) if s >= seed_count] or list(range(6))
    same = [(s, l) for s, l in prompts.matches() if s['n'] in (8, 16, 32) and s['seed_idx'] in held]
    # r = 0.70 excluded: the n = 64 normative log-odds reach 27 logits, so fractions of the normative gap are meaningless
    strata = [(n, f, r) for n in (8, 16, 32) for f in ('listed', 'count') for r in (0.52, 0.55)]
    rng.shuffle(strata)
    chosen = []
    while len(chosen) < n_same:
        progressed = False
        for key in strata:
            pool = [p for p in same if (p[0]['n'], p[0]['format'], p[0]['r']) == key and p not in chosen]
            if pool and len(chosen) < n_same:
                chosen.append(pool[rng.integers(len(pool))]); progressed = True
        if not progressed: break
    by = {(c['r'], c['seed_idx'], c['domain'], c['format'], c['n'], c['k']): c for c in cs}
    pos = []
    for c in cs:
        if c['n'] in (16, 32) and math.isclose(c['share'], .25) and c['seed_idx'] in held and c['r'] < .7:
            d = by.get((c['r'], c['seed_idx'], c['domain'], c['format'], c['n'], 3 * c['n'] // 4))
            if d: pos.append((c, d))
    pos = [pos[i] for i in rng.permutation(len(pos))[:n_pos]]
    return chosen, pos


# ----------------------------------------------------------------------------- model-side helpers (GPU modes)

def token_ids(tok, model, kev, case):
    if kev:
        from runtime import kev_enc
        enc = kev_enc(tok, model, case)
        L = len(enc['ids'])
        return enc['ids'], [enc['decide_idx'][0] - L] + [o - L for o in enc['opt_idx'][0]]
    from runtime import lm_inputs
    return lm_inputs(tok, case)[0].tolist(), [-1]


def common_suffix(a, b, cap=MAX_SUFFIX):
    s = 0
    while s < min(len(a), len(b), cap) and a[-1 - s] == b[-1 - s]:
        s += 1
    return s


def _first(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, t):
    return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t


def _module(layers, layer, kind):
    if kind == 'residual':
        return layers[layer - 1]
    mod = getattr(layers[layer - 1], 'self_attn', None)
    if mod is None: raise LookupError('no self_attn')
    return mod


@contextmanager
def edit_hook(model, kev, layer, kind, positions, fn):
    """fn(v [P, d] float32 at the given negative positions) -> replacement [P, d] or None (capture only).
    layer 0 residual = input to the first block; layer i residual = output of block i-1; attention = block i-1 attention."""
    import torch
    from runtime import layers_of
    layers = layers_of(model, kev)
    pos = list(positions)

    def apply(t):
        idx = torch.tensor([t.shape[1] + p for p in pos], device=t.device)
        y = fn(t[0, idx].detach().float())
        if y is None: return None
        t = t.clone(); t[0, idx] = y.to(t.dtype); return t

    if kind == 'residual' and layer == 0:
        def pre(_m, args):
            t = apply(args[0])
            return None if t is None else (t,) + tuple(args[1:])
        h = layers[0].register_forward_pre_hook(pre)
    else:
        def post(_m, _a, out):
            t = apply(_first(out))
            return None if t is None else _replace(out, t)
        h = _module(layers, layer, kind).register_forward_hook(post)
    try: yield
    finally: h.remove()


def logit_of(p):
    p = min(max(float(p), 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


# ----------------------------------------------------------------------------- padded mode

def run_padded(args, tok, model, kev, root, deadline):
    from behavior import capture
    out_dir = root / 'residual_padded'; out_dir.mkdir(exist_ok=True)
    log = root / 'behavior_padded.jsonl'
    done = {json.loads(x)['id'] for x in log.read_text().splitlines()} if log.exists() else set()
    todo = [c for c in prompts.cases() if c['format'] == 'listed' and c['n'] < 64 and c['seed_idx'] < args.seed_count]
    with log.open('a') as f:
        for c in todo:
            if time.monotonic() > deadline:
                print('padded: soft deadline reached; rerun to resume', flush=True); return False
            path = out_dir / (c['id'] + '.npy')
            if c['id'] in done and path.exists(): continue
            p, vectors = capture(tok, model, kev, dict(c, state=c['state_padded']))
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as s: np.save(s, vectors, allow_pickle=False)
            tmp.replace(path)
            f.write(json.dumps({'spec': 'v3.4', 'patch_version': 'v2', 'id': c['id'], 'n': c['n'], 'k': c['k'],
                                'r': c['r'], 'pad_lines': c['pad_lines'], 'p_yes': p}) + '\n'); f.flush()
    return True


# ----------------------------------------------------------------------------- patch mode

def run_patch(args, tok, model, kev, root, deadline):
    import torch
    from runtime import layers_of, score
    probes = np.load(root / 'probes_v2.npz')
    n_layers = len(layers_of(model, kev))
    if list(probes['layers']) != list(range(n_layers + 1)):
        raise RuntimeError('probes_v2.npz must cover every residual layer; rerun probe.py without --layers')
    dev = next(model.parameters()).device
    W = {t: torch.as_tensor(probes[f'w_{t}'], device=dev, dtype=torch.float32) for t in
         SUBSPACE_TARGETS + ('share', 'normative_log_odds')}
    B = {t: torch.as_tensor(probes[f'b_{t}'], device=dev, dtype=torch.float32) for t in W}
    gen = torch.Generator(device='cpu').manual_seed(1234)
    d = W['k'].shape[1]

    def basis(vs):
        q, _ = torch.linalg.qr(torch.stack(vs, 1))
        return q

    U_count = [basis([W[t][l] for t in SUBSPACE_TARGETS]) for l in range(n_layers + 1)]
    U_share = [basis([W['share'][l]]) for l in range(n_layers + 1)]
    U_rand = [basis([torch.randn(d, generator=gen).to(dev) for _ in SUBSPACE_TARGETS]) for l in range(n_layers + 1)]
    R_dir = [torch.randn(d, generator=gen).to(dev) for _ in range(n_layers + 1)]

    def readout(vec, t, layer=n_layers):
        return float(vec @ W[t][layer] + B[t][layer])

    out = root / 'patch_v2.jsonl'
    done = set()
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            done.add((r['receiver'], r['donor'], r['layer'], r['intervention']))

    def run(case, hooks):
        """Forward with a list of (layer, kind, positions, fn); also capture final-layer final-token residual."""
        final = {}

        def keep(v):
            final['v'] = v[0].clone()
            return None
        with ExitStack() as st:
            for layer, kind, pos, fn in hooks:
                st.enter_context(edit_hook(model, kev, layer, kind, pos, fn))
            # registered last, so it sees the edited output when an edit targets the last layer
            st.enter_context(edit_hook(model, kev, n_layers, 'residual', [-1], keep))
            p = score(tok, model, kev, case)
        return logit_of(p), final['v']

    def capture_all(case, positions):
        acts, attn = {}, {}
        with ExitStack() as st:
            for l in range(n_layers + 1):
                st.enter_context(edit_hook(model, kev, l, 'residual', positions,
                                           lambda v, l=l: acts.__setitem__(l, v.clone()) or None))
                if l > 0:
                    try:
                        st.enter_context(edit_hook(model, kev, l, 'attention', positions,
                                                   lambda v, l=l: attn.__setitem__(l, v.clone()) or None))
                    except LookupError:
                        pass
            p = score(tok, model, kev, case)
        return logit_of(p), acts, attn

    same, pos_pairs = pair_sets(args.seed_count, args.pairs, args.positive_pairs)
    jobs = [('positive', r, dn) for r, dn in pos_pairs] + [('same_share', r, dn) for r, dn in same] + \
           [('same_share_reverse', dn, r) for r, dn in same]
    with out.open('a') as f:
        for kind_pair, rec, don in jobs:
            if time.monotonic() > deadline:
                print('patch: soft deadline reached; rerun the same command to resume', flush=True)
                summarize(root); return False
            ids_r, ro_r = token_ids(tok, model, kev, rec); ids_d, ro_d = token_ids(tok, model, kev, don)
            s = common_suffix(ids_r, ids_d)
            if ro_r != ro_d or min(ro_r) < -s:
                raise RuntimeError(f'readout positions not in the common suffix: {rec["id"]} {don["id"]}')
            suffix = list(range(-s, 0)); readout_pos = ro_r
            base, acts_r, _ = capture_all(rec, suffix)
            donor, acts_d, attn_d = capture_all(don, suffix)
            _, fin_r = run(rec, []); _, fin_d = run(don, [])
            ix = {p: i for i, p in enumerate(suffix)}
            ro_i = [ix[p] for p in readout_pos]; last_i = ix[-1]
            meta = dict(spec='v3.4', patch_version='v2', pair=kind_pair, receiver=rec['id'], donor=don['id'],
                        n_receiver=rec['n'], n_donor=don['n'], k_receiver=rec['k'], k_donor=don['k'], r=rec['r'],
                        format=rec['format'], share_receiver=rec['share'], share_donor=don['share'],
                        normative_receiver=prompts.log_odds(rec), normative_donor=prompts.log_odds(don),
                        base=base, donor_logit=donor, common_suffix=s, readout_positions=readout_pos,
                        n_readout_base=2 ** readout(fin_r, 'log2_n'), n_readout_donor=2 ** readout(fin_d, 'log2_n'))
            main_only = kind_pair == 'same_share_reverse'
            t_pair = time.monotonic()
            for layer in (args.layers or range(n_layers + 1)):
                Ad = acts_d[layer]; Ar = acts_r[layer]
                todo = {'full_readout': [(layer, 'residual', readout_pos, lambda v, Ad=Ad: Ad[ro_i])]}
                Uc = U_count[layer]; Ur = U_rand[layer]; Us = U_share[layer]
                sub = lambda U, Ad=Ad: (lambda v: v + ((Ad[[last_i]] - v) @ U) @ U.T)
                todo['count_subspace'] = [(layer, 'residual', [-1], sub(Uc))]
                if not main_only:
                    todo['random_subspace'] = [(layer, 'residual', [-1], sub(Ur))]
                if kind_pair == 'positive':
                    todo['share_subspace'] = [(layer, 'residual', [-1], sub(Us))]
                if kind_pair == 'same_share':
                    todo['full_suffix'] = [(layer, 'residual', suffix, lambda v, Ad=Ad: Ad)]
                    if layer in attn_d:
                        todo['attention_readout'] = [(layer, 'attention', readout_pos,
                                                      lambda v, a=attn_d[layer]: a[ro_i])]
                    w = W['normative_log_odds'][layer]
                    delta = meta['normative_donor'] - meta['normative_receiver']
                    unit = w / (w @ w).clamp_min(1e-12)       # shifts the probe readout by 1 per unit
                    rnd = R_dir[layer] / R_dir[layer].norm() * unit.norm()
                    for a in (1., 4.):
                        todo[f'steer_normative_x{a:g}'] = [(layer, 'residual', [-1], lambda v, s_=a * delta * unit: v + s_)]
                    todo['steer_random_x4'] = [(layer, 'residual', [-1], lambda v, s_=4. * delta * rnd: v + s_)]
                for name, hooks in todo.items():
                    if (rec['id'], don['id'], layer, name) in done: continue
                    patched, fin = run(rec, hooks)
                    row = dict(meta, layer=layer, intervention=name, patched=patched, shift=patched - base,
                               n_readout_patched=2 ** readout(fin, 'log2_n'),
                               residual_norm=float(Ar[last_i].norm()))
                    if name.startswith('steer'):
                        row['steer_norm'] = float((4. if 'x4' in name else 1.) * abs(meta['normative_donor'] -
                                                  meta['normative_receiver']) * unit.norm())
                    f.write(json.dumps(row) + '\n'); f.flush()
            print(f"{kind_pair} {rec['id']} <- {don['id']}: {time.monotonic() - t_pair:.0f}s", flush=True)
    summarize(root)
    return True


# ----------------------------------------------------------------------------- summary (CPU)

def _boot(x, rng, B=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return [float('nan'), float('nan')]
    m = [np.mean(x[rng.integers(len(x), size=len(x))]) for _ in range(B)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(root):
    rows = [json.loads(x) for x in (root / 'patch_v2.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(0)
    table = {}
    for r in rows:
        table.setdefault((r['pair'], r['intervention'], r['layer']), []).append(r)
    cells = []
    for (pair, inter, layer), rs in sorted(table.items()):
        shift = np.array([r['shift'] for r in rs])
        gap = np.array([r['donor_logit'] - r['base'] for r in rs])
        ngap = np.array([r['normative_donor'] - r['normative_receiver'] for r in rs])
        ok = np.abs(gap) >= GAP_MIN
        fg = np.where(ok, shift / np.where(ok, gap, 1), np.nan)
        fn = shift / ngap
        nr = np.array([(r['n_readout_patched'] - r['n_readout_base']) /
                       (r['n_readout_donor'] - r['n_readout_base']) if abs(r['n_readout_donor'] - r['n_readout_base']) > 1 else np.nan
                       for r in rs])
        cells.append(dict(pair=pair, intervention=inter, layer=layer, n_pairs=len(rs),
                          shift_mean=float(shift.mean()), shift_ci=_boot(shift, rng),
                          gap_mean=float(gap.mean()), n_gap_eligible=int(ok.sum()),
                          frac_gap_mean=float(np.nanmean(fg)) if ok.any() else None,
                          frac_gap_ci=_boot(fg, rng) if ok.sum() > 1 else None,
                          frac_normative_mean=float(fn.mean()), frac_normative_ci=_boot(fn, rng),
                          n_readout_frac_mean=float(np.nanmean(nr)) if (~np.isnan(nr)).any() else None))
    by = {(c['pair'], c['intervention'], c['layer']): c for c in cells}
    layers = sorted({c['layer'] for c in cells})
    last = max(layers) if layers else None
    # sanity checks
    sanity = [abs(r['patched'] - r['donor_logit']) for r in rows if r['intervention'] == 'full_readout' and r['layer'] == last]
    null0 = [abs(r['shift']) for r in rows if r['intervention'] == 'full_suffix' and r['layer'] == 0]
    # subspace minus random subspace, in fractions of the normative gap (paired by pair and layer)
    diff = {}
    fnorm = lambda r: r['shift'] / (r['normative_donor'] - r['normative_receiver'])
    for pair, inter in (('same_share', 'count_subspace'), ('positive', 'count_subspace'), ('positive', 'share_subspace')):
        for layer in layers:
            a = {(r['receiver'], r['donor']): fnorm(r) for r in table.get((pair, inter, layer), [])}
            b = {(r['receiver'], r['donor']): fnorm(r) for r in table.get((pair, 'random_subspace', layer), [])}
            keys = sorted(set(a) & set(b))
            if keys:
                dd = np.array([a[k] - b[k] for k in keys])
                diff[f'{pair}_{inter}_L{layer}'] = {'mean': float(dd.mean()), 'ci': _boot(dd, rng), 'n': len(keys)}
    # positive control: share or count subspace closes >= POSCTRL_FRAC of the share gap at some layer, beyond random
    pc = []
    for layer in layers:
        for inter in ('share_subspace', 'count_subspace', 'full_readout'):
            c = by.get(('positive', inter, layer))
            if c and c['frac_gap_mean'] is not None:
                pc.append((c['frac_gap_mean'], layer, inter))
    pc_sub = [x for x in pc if x[2] != 'full_readout']
    pos_ok = any(frac >= POSCTRL_FRAC and diff.get(f'positive_{inter}_L{layer}', {}).get('ci', [0.])[0] > 0
                 for frac, layer, inter in pc_sub)
    cs = [by[('same_share', 'count_subspace', l)] for l in layers if ('same_share', 'count_subspace', l) in by]
    upper = max((c['frac_normative_ci'][1] for c in cs if not math.isnan(c['frac_normative_ci'][1])), default=float('nan'))
    toward = [c for c in cs if c['frac_normative_mean'] >= USED_FRAC and c['frac_normative_ci'][0] > 0
              and diff.get(f"same_share_count_subspace_L{c['layer']}", {}).get('ci', [0.])[0] > 0]
    sane = bool(sanity) and max(sanity) < SANITY_TOL and (not null0 or max(null0) < SANITY_TOL)
    if not sane:
        verdict = 'invalid: hook sanity failed'
    elif not pos_ok:
        verdict = 'inconclusive: positive control failed (patching cannot move the output even along share)'
    elif toward:
        verdict = f'count used: count-subspace interchange moves the output toward the donor normative answer by >= {USED_FRAC:.0%} at layers {[c["layer"] for c in toward]}'
    elif not math.isnan(upper) and upper < USED_FRAC:
        verdict = (f'unused (causal part): count-subspace interchange moves the output by < {USED_FRAC:.0%} of the normative '
                   f'gap at every layer (max upper 95% CI {upper:.3f}) while the positive control works. '
                   '"Present" must come separately from the length-controlled probe results (probe_v2.json).')
    else:
        verdict = 'inconclusive: count-subspace effect not distinguishable from the threshold'
    summary = {'spec': 'v3.4', 'patch_version': 'v2', 'n_rows': len(rows),
               'n_pairs': {p: len({(r['receiver'], r['donor']) for r in rows if r['pair'] == p}) for p in
                           ('same_share', 'same_share_reverse', 'positive')},
               'thresholds': {'GAP_MIN': GAP_MIN, 'SANITY_TOL': SANITY_TOL, 'USED_FRAC': USED_FRAC,
                              'POSCTRL_FRAC': POSCTRL_FRAC, 'note': 'exploratory thresholds; the historical positive control failed'},
               'sanity': {'last_layer_full_readout_max_abs_err': max(sanity) if sanity else None,
                          'layer0_full_suffix_max_abs_shift': max(null0) if null0 else None},
               'positive_control_ok': pos_ok, 'positive_control_best': max(pc_sub) if pc_sub else None,
               'count_minus_random': diff, 'verdict': verdict,
               'ci_method': 'bootstrap over pairs (2000 draws)', 'cells': cells}
    (root / 'patch_v2_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cells', 'count_minus_random')}, indent=1))
    return summary


def main():
    from behavior import MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--mode', choices=('padded', 'patch', 'summarize'), default='patch')
    ap.add_argument('--out', default='results')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--seed-count', type=int, default=3, help='seeds used for behavior/probes (pairs use the others)')
    ap.add_argument('--pairs', type=int, default=18, help='same-share pairs (stratified over receiver n x format x r)')
    ap.add_argument('--positive-pairs', type=int, default=6)
    ap.add_argument('--max-seconds', type=float, default=1050., help='soft deadline; resumable (exit code 75 if cut)')
    ap.add_argument('--layers', type=int, nargs='*', default=None, help='smoke tests only: subset of layers')
    args = ap.parse_args()
    root = Path(args.out) / args.model
    if args.mode == 'summarize':
        summarize(root); return
    deadline = time.monotonic() + args.max_seconds
    meta = json.loads((root / 'meta.json').read_text())
    if meta['generator_sha256'] != prompts.GENERATOR_SHA256: raise RuntimeError('generator differs from cached run')
    from runtime import load_model, layers_of
    tok, model, kev = load_model(MODELS[args.model], args.device, args.dtype)
    if len(layers_of(model, kev)) != meta['layers']: raise RuntimeError('layer count mismatch')
    complete = run_padded(args, tok, model, kev, root, deadline) if args.mode == 'padded' else \
        run_patch(args, tok, model, kev, root, deadline)
    if not complete:
        raise SystemExit(75)   # partial but resumable: modal_app records 'partial' and does not write the done marker


if __name__ == '__main__': main()
