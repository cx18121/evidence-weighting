"""E5a probes (spec v3.4; corrected held-out evaluation and nuisance controls).

The earlier, excluded analysis selected a ridge penalty on test folds and did not control for prompt length.
In the listed format the number of
reports n is almost a linear function of the token length (= final-token position), and at fixed n the token length
also moves with k because the positive and negative labels have different token lengths. v2 therefore reports, per
layer and target:

  * r2            linear (ridge) probe, penalty chosen by *nested* group CV inside the training folds;
  * nuisance_r2   the same CV with only nuisance features Z (token length, log length, each interacted with
                  format x domain, plus format x domain and template indicators): what length/wording alone predicts;
  * partial mode  Frisch-Waugh-Lovell residualization: target and activations residualized on Z (fit on the training
                  fold only); r2 is the share of the *non-nuisance* target variance that the probe recovers;
  * control_r2 / selectivity   Hewitt & Liang (2019) control task adapted to regression: each evidence cell
                  (n, k, r) gets a random value drawn by permuting the target's cell values across cells; selectivity
                  = r2(target) - mean r2(control);
  * nl_*          a nonlinear probe: RBF-kernel ridge on the same standardized activations, bandwidth and penalty
                  chosen by the same nested CV, same folds and control task. (A 1x64 MLP on the top 50 principal
                  components underfit badly with ~800 training prompts; Kev-0.8B layer 12, log2 n: 0.71 vs linear 0.93.)
  * analyses      all prompts (seed- and template-held-out), count format only (length nearly constant), listed only,
                  padded listed prompts when residual_padded/ exists (length matched to n=64), k at fixed n and fixed
                  format (partial on Z), log2 n at fixed share (count format and padded listed), and n <= 32 -> n = 64
                  extrapolation (MAE for n, R^2 within n = 64 for the others; R^2 is undefined for a constant target).

`--tokens-only` (needs the model tokenizer; run in the Modal image) writes tokens.json with exact token lengths and the
readout positions. Without tokens.json the analysis refuses to run unless --allow-char-proxy is given (tests only).
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
import prompts

PROBE_VERSION = 'v2'
TARGETS = ('n', 'log2_n', 'k', 'margin', 'share', 'normative_log_odds')
NL_TARGETS = ('log2_n', 'k', 'margin', 'share', 'normative_log_odds')
DIRECTION_TARGETS = ('log2_n', 'k', 'margin', 'share', 'normative_log_odds')
ALPHAS = 10.0 ** np.arange(0, 7)
N_CONTROL = 3


# ----------------------------------------------------------------------------- targets and nuisance features

def target_values(cases):
    n = np.array([c['n'] for c in cases], float)
    k = np.array([c['k'] for c in cases], float)
    # margin = 2k - n (positive minus negative reports): the r-free evidence count; normative log-odds = margin * logit(r)
    return {'n': n, 'log2_n': np.log2(n), 'k': k, 'margin': 2 * k - n, 'share': k / n,
            'normative_log_odds': np.array([prompts.log_odds(c) for c in cases], float)}


def control_values(cases, y, seed):
    """Hewitt-Liang style control task: the evidence cell (n, k, r) is the 'type'; each cell receives the target value
    of a randomly permuted cell, so the control is a deterministic but arbitrary function of the cell."""
    cells = [(c['n'], c['k'], c['r']) for c in cases]
    uniq = sorted(set(cells))
    value = {cell: float(np.mean(y[[i for i, x in enumerate(cells) if x == cell]])) for cell in uniq}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    remap = {uniq[i]: value[uniq[j]] for i, j in enumerate(perm)}
    return np.array([remap[x] for x in cells])


def nuisance(cases, L, with_template=True):
    """Z: format x domain indicators and *relative* token length interacted with format x domain. Relative length =
    token length minus the mean token length of the same (format, domain, template) wording over the full design grid
    (identical grid for every wording, so this removes each wording's fixed offset without using any target). Length =
    final-token position + 1 (decide token for Kev, last chat-template token for LLMs). Why relative: with 3 seeds each
    held-out seed brings (domain, template) pairs unseen in training (template = (seed + domain) mod 3), and an OLS on
    absolute length then fitted wording offsets as if they were length effects (count-format nuisance R^2 of -5).
    with_template is kept for API compatibility; wording offsets are always removed by the centering."""
    L = np.asarray(L, float)
    key = [(c['format'], c['domain'], c['template']) for c in cases]
    mean = {k: L[[i for i, x in enumerate(key) if x == k]].mean() for k in set(key)}
    dL = np.array([l - mean[k] for l, k in zip(L, key)])
    fd = [(c['format'], c['domain']) for c in cases]
    cols = []
    for cell in sorted(set(fd)):
        m = np.array([x == cell for x in fd], float)
        cols += [m, m * dL / 10.]
    return np.stack(cols, 1)


# ----------------------------------------------------------------------------- kernel-form ridge (fast for d >> N)

def _center(Ktr, Kte):
    rm, cm, am = Ktr.mean(1, keepdims=True), Ktr.mean(0, keepdims=True), Ktr.mean()
    return Ktr - rm - cm + am, Kte - Kte.mean(1, keepdims=True) - cm + am


def _krr(Ktr, Kte, Ytr, alphas):
    """Ridge with unpenalized intercept in kernel form on centered Grams -> [A, q, t]."""
    ym = Ytr.mean(0)
    lam, Q = np.linalg.eigh(Ktr)
    lam = np.clip(lam, 0, None)
    QtY = Q.T @ (Ytr - ym)
    KQ = Kte @ Q
    return np.stack([KQ @ (QtY / (lam + a)[:, None]) + ym for a in alphas]), (lam, Q)


def _lstsq_resid(Ztr, Zte, Mtr, Mte):
    B = np.linalg.lstsq(Ztr, Mtr, rcond=None)[0]
    return Mtr - Ztr @ B, Mte - Zte @ B, Zte @ B


def _pool_r2(T, P):
    ok = ~np.isnan(P).any(1) & ~np.isnan(T).any(1)
    if ok.sum() < 3: return np.full(T.shape[1], np.nan)
    T, P = T[ok], P[ok]
    ss = ((T - T.mean(0)) ** 2).sum(0)
    return np.where(ss > 1e-12, 1 - ((T - P) ** 2).sum(0) / np.maximum(ss, 1e-12), np.nan)


RBF_SCALES = (0.25, 1., 4.)   # gamma = scale / median squared distance in the training fold


def _select_predict(Gs, m, gtr, Ytr, alphas):
    """Gs: list of (train+test) x (train+test) kernels, training rows first (m of them). Inner leave-one-group-out CV
    over (kernel, alpha) per target inside the training fold; refit on the whole training fold."""
    t = Ytr.shape[1]
    sse = np.zeros((len(Gs), len(alphas), t))
    for gi, G in enumerate(Gs):
        Gtr = G[:m, :m]
        for h in np.unique(gtr):
            iv = gtr == h; it = ~iv
            Kt, Kv = _center(Gtr[np.ix_(it, it)], Gtr[np.ix_(iv, it)])
            sse[gi] += ((_krr(Kt, Kv, Ytr[it], alphas)[0] - Ytr[iv][None]) ** 2).sum(1)
    flat = sse.reshape(-1, t).argmin(0)
    out = np.zeros((Gs[0].shape[0] - m, t)); choice = []; preds = {}
    for j in range(t):
        gi, ai = divmod(int(flat[j]), len(alphas))
        if gi not in preds:
            Kt, Kv = _center(Gs[gi][:m, :m], Gs[gi][m:, :m])
            preds[gi] = _krr(Kt, Kv, Ytr, alphas)[0]
        out[:, j] = preds[gi][ai, :, j]
        choice.append((gi, float(alphas[ai])))
    return out, choice


def cv_krr(X, Y, groups, Z=None, nonlinear=False, seed=0, alphas=ALPHAS):
    """Group-held-out probes. X [N,d]; Y [N,t]; groups [N]; Z [N,z] nuisance (partial mode) or None (raw mode).
    Linear: ridge (kernel form, unpenalized intercept). Nonlinear: RBF kernel ridge, bandwidth and penalty chosen
    jointly. Penalty (and bandwidth) per target by inner leave-one-group-out CV inside each outer training fold, never
    on the test fold. Returns targets T (residualized in partial mode), linear predictions P, nonlinear predictions M
    (or None) and the nuisance-only predictions NU of the original Y (partial mode)."""
    X = np.asarray(X, np.float32); Y = np.asarray(Y, float); groups = np.asarray(groups)
    N, t = Y.shape
    T = Y.copy(); P = np.full((N, t), np.nan); NU = np.full((N, t), np.nan)
    M = np.full((N, t), np.nan) if nonlinear else None
    chosen, chosen_nl, folds = [], [], []
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
        if Z is not None:
            Xtr, Xte, _ = _lstsq_resid(Z[tr], Z[te], Xtr.astype(np.float64), Xte.astype(np.float64))
            Ytr, Yte, NU[te] = _lstsq_resid(Z[tr], Z[te], Ytr, Yte)
            T[te] = Yte
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-6] = 1.
        A = ((np.vstack([Xtr, Xte]) - mu) / sd).astype(np.float32)
        G = (A @ A.T).astype(np.float64)
        m = int(tr.sum()); gtr = groups[tr]
        P[te], ch = _select_predict([G], m, gtr, Ytr, alphas)
        chosen.append([c[1] for c in ch])
        if nonlinear:
            sq = np.diag(G)
            D2 = np.clip(sq[:, None] + sq[None, :] - 2 * G, 0, None)
            med = max(float(np.median(D2[:m, :m][np.triu_indices(m, 1)])), 1e-9)   # layer 0: identical final tokens
            M[te], ch = _select_predict([np.exp(-s_ * D2 / med) for s_ in RBF_SCALES], m, gtr, Ytr, alphas)
            chosen_nl.append([(RBF_SCALES[c[0]], c[1]) for c in ch])
        folds.append(te)
    fold_r2 = [_pool_r2(T[f], P[f]).tolist() for f in folds]
    return dict(T=T, P=P, NU=NU, M=M, alphas=chosen, alphas_nl=chosen_nl, fold_r2=fold_r2)


def cv_ols(Z, Y, groups):
    P = np.full(Y.shape, np.nan)
    for g in np.unique(groups):
        te = groups == g
        P[te] = Z[te] @ np.linalg.lstsq(Z[~te], Y[~te], rcond=None)[0]
    return P


def fit_direction(X, y, groups, alphas=ALPHAS):
    """Probe on all given rows (penalty by group CV). Returns raw-space weight w and intercept b: y_hat = X @ w + b."""
    X = np.asarray(X, np.float64); y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-6] = 1.
    A = (X - mu) / sd; G = A @ A.T
    sse = np.zeros(len(alphas))
    for g in np.unique(groups):
        iv = groups == g; it = ~iv
        Kt, Kv = _center(G[np.ix_(it, it)], G[np.ix_(iv, it)])
        sse += ((_krr(Kt, Kv, y[it, None], alphas)[0][:, :, 0] - y[iv]) ** 2).sum(1)
    a = alphas[int(sse.argmin())]
    Ac = A - A.mean(0)
    dual = np.linalg.solve(Ac @ Ac.T + a * np.eye(len(y)), y - y.mean())
    w_std = Ac.T @ dual
    w = w_std / sd
    # y_hat = (x - mu)/sd . w_std - mean(A) . w_std + ybar  ==  x . w + [ybar - mean(A).w_std - mu . w]
    b = float(y.mean() - A.mean(0) @ w_std - mu @ w)
    return w.astype(np.float32), b, float(a)


# ----------------------------------------------------------------------------- legacy API (kept for tests)

def cv_probe(X, y, groups, seed=41):
    """Legacy single-target interface; now nested penalty selection. Returns (metrics, (reg-like, mu, std))."""
    groups = np.asarray(groups); y = np.asarray(y, float)
    out = cv_krr(X, np.column_stack([y, np.random.default_rng(seed).permutation(y)]), groups)
    r2 = _pool_r2(out['T'], out['P'])
    w, b, a = fit_direction(X, y, groups)

    class _Lin:
        def __init__(self, w, b): self.coef_, self.intercept_ = w, b
        def predict(self, Xs): return Xs @ self.coef_ + self.intercept_
    # (X - mu)/std with mu=0, std=1 reproduces the raw-space linear map
    d = np.asarray(X).shape[1]
    return {'r2': float(r2[0]), 'alpha': a, 'shuffled_r2': float(r2[1])}, (_Lin(w, b), np.zeros(d), np.ones(d))


def predict(model, X):
    reg, mu, std = model
    return reg.predict((X - mu) / std)


def layer_analysis(X, cases, probs=None):
    """Legacy summary (raw probes, seed CV on n<=32, n=64 OOD) + unit normative directions; used by the unit test."""
    tr = np.asarray([c['n'] <= 32 for c in cases]); te = ~tr
    if not tr.any() or not te.any(): raise ValueError('need n<=32 AND n=64')
    legacy = ('k', 'n', 'n_minus_k', 'share', 'normative_log_odds')
    results, directions = [], {}
    tv = target_values(cases); tv['n_minus_k'] = tv['n'] - tv['k']
    for layer in range(X.shape[1]):
        A = X[tr, layer].astype(np.float32); B = X[te, layer].astype(np.float32)
        for target in legacy:
            y = tv[target]
            cv, fitted = cv_probe(A, y[tr], np.array([c['seed'] for c in cases])[tr], seed=layer + 120)
            tc, _ = cv_probe(A, y[tr], np.array([c['template'] for c in cases])[tr], seed=layer + 290)
            results.append(dict(layer=layer, target=target, seed_cv=cv, template_cv=tc,
                                n64_r2=float(r2_score(y[te], predict(fitted, B)))))
            if target == 'normative_log_odds':
                w = fitted[0].coef_
                directions[f'layer_{layer}'] = (w / max(np.linalg.norm(w), 1e-9)).astype(np.float32)
    return results, directions


# ----------------------------------------------------------------------------- token features

def compute_tokens(model_key, cases, path):
    """Exact token ids of every (unpadded and padded) prompt with the model's own input path."""
    from behavior import MODELS
    name = MODELS[model_key]
    if name.startswith('jaredpalmer/kev-'):
        from kev.checkpoint import Checkpoint
        from kev.model import load_tokenizer, encode
        from kev.api import SystemOneRequest, to_record
        ck = Checkpoint(name)
        tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)

        def ids_of(state):
            req = SystemOneRequest.model_validate({'state': state, 'questions': {
                'answer': {'type': 'noul', 'instructions': prompts.QUESTION}}})
            rec, _ = to_record(req)
            enc = encode(tok, rec, strict=True, max_state=65536, max_branch=73728,
                         option_isolation=ck.meta.option_isolation)
            L = len(enc['ids'])
            return enc['ids'], [enc['decide_idx'][0] - L] + [o - L for o in enc['opt_idx'][0]]
    else:
        from transformers import AutoTokenizer
        from runtime import lm_inputs
        tok = AutoTokenizer.from_pretrained(name)

        def ids_of(state):
            return lm_inputs(tok, {'state': state})[0].tolist(), [-1]
    feats = {}
    for c in cases:
        for key, state in ((c['id'], c['state']), (c['id'] + '|padded', c.get('state_padded'))):
            if state is None: continue
            ids, readout = ids_of(state)
            feats[key] = {'len': len(ids), 'readout': readout, 'suffix_ids': ids[-160:]}
    Path(path).write_text(json.dumps({'model': name, 'features': feats}))
    return feats


def token_lengths(root, cases, padded=False, allow_proxy=False):
    p = root / 'tokens.json'
    suffix = '|padded' if padded else ''
    if p.exists():
        f = json.loads(p.read_text())['features']
        return np.array([f[c['id'] + suffix]['len'] for c in cases], float), 'exact_tokens'
    if not allow_proxy:
        raise RuntimeError(f'{p} missing: run probe.py --tokens-only in the model image first')
    key = 'state_padded' if padded else 'state'
    return np.array([len(c[key]) / 4. for c in cases], float), 'char_length_proxy'


# ----------------------------------------------------------------------------- analyses

def _selectivity(r2, ctrl):
    """Selectivity with the control R^2 floored at 0 (a control worse than the mean is 'not learned', like chance)."""
    return float(r2 - max(ctrl, 0.))


def _record(res, names, ctrl_index, nuis=None, Y=None):
    r2 = _pool_r2(res['T'], res['P'])
    nl = None if res['M'] is None else _pool_r2(res['T'], res['M'])
    out = {}
    for j, name in enumerate(names):
        row = {'r2': float(r2[j]), 'r2_folds': [float(f[j]) for f in res['fold_r2']],
               'alpha': [a[j] for a in res['alphas']]}
        if name in ctrl_index:
            c = float(np.mean([r2[i] for i in ctrl_index[name]]))
            row['control_r2'] = c; row['selectivity'] = _selectivity(r2[j], c)
        if nuis is not None:
            row['nuisance_r2'] = float(nuis[j])
        if not np.isnan(res['NU']).all():   # partial mode: joint R^2 of nuisance + probe on the original target
            row['r2_joint'] = float(_pool_r2(Y[:, [j]], (res['NU'] + res['P'])[:, [j]])[0])
        if nl is not None and name in NL_TARGETS:
            row['nl_r2'] = float(nl[j])
            if name in ctrl_index:
                c = float(np.mean([nl[i] for i in ctrl_index[name]]))
                row['nl_control_r2'] = c; row['nl_selectivity'] = _selectivity(nl[j], c)
            row['nl_choice'] = [a[j] for a in res['alphas_nl']]
        out[name] = row
    return out


def run_block(X, cases, L, groups, targets, partial, controls=True, nonlinear=False, with_template=True, seed=0):
    tv = target_values(cases)
    names = list(targets); cols = [tv[t] for t in targets]; ctrl_index = {}
    if controls:
        for t in targets:
            ctrl_index[t] = []
            for s in range(N_CONTROL):
                ctrl_index[t].append(len(cols)); cols.append(control_values(cases, tv[t], seed=1000 * s + 17))
    Y = np.column_stack(cols)
    Z = nuisance(cases, L, with_template)
    res = cv_krr(X, Y, groups, Z=Z if partial else None, nonlinear=nonlinear, seed=seed)
    nuis = _pool_r2(Y, cv_ols(Z, Y, groups))
    return _record(res, names, ctrl_index, nuis, Y)


def extrapolate(X, cases, groups, targets=('log2_n', 'k', 'margin', 'share', 'normative_log_odds')):
    tv = target_values(cases)
    tr = np.array([c['n'] <= 32 for c in cases]); te = ~tr
    out = {}
    for t in targets:
        w, b, a = fit_direction(X[tr], tv[t][tr], groups[tr])
        pred = X[te].astype(np.float64) @ w + b; y = tv[t][te]
        row = {'bias': float(np.mean(pred - y)), 'mae': float(np.mean(np.abs(pred - y))), 'alpha': a}
        if np.std(y) > 1e-9: row['r2_within_n64'] = float(r2_score(y, pred))
        if t == 'log2_n': row['mean_pred_n'] = float(np.mean(2 ** pred))
        out[t] = row
    return out


def layer_block(Xl, cases, L, seeds, templates, padded=None, nonlinear=True, seed=0):
    """All analyses for one layer. Xl [N,d] (float32). padded = (Xp, cases_p, Lp) or None."""
    fmt = np.array([c['format'] for c in cases]); nn = np.array([c['n'] for c in cases])
    shares = np.array([c['share'] for c in cases])
    out = {}
    for mode in ('raw', 'partial'):
        p = mode == 'partial'
        out[f'all_seed_{mode}'] = run_block(Xl, cases, L, seeds, TARGETS, p, nonlinear=nonlinear, seed=seed)
        out[f'all_template_{mode}'] = run_block(Xl, cases, L, templates, TARGETS, p, controls=False,
                                                with_template=False, seed=seed)
        for f in ('count', 'listed'):
            s = fmt == f
            sub = [c for c, x in zip(cases, s) if x]
            out[f'{f}_seed_{mode}'] = run_block(Xl[s], sub, L[s], seeds[s], TARGETS, p,
                                                nonlinear=nonlinear and f == 'count', seed=seed)
        # k at fixed n (and fixed format); length residualized in partial mode
        for f in ('count', 'listed'):
            for n in (8, 16, 32, 64):
                s = (fmt == f) & (nn == n)
                sub = [c for c, x in zip(cases, s) if x]
                out[f'fixed_n{n}_{f}_{mode}'] = run_block(Xl[s], sub, L[s], seeds[s], ('k',), p, seed=seed)
        # log2 n at fixed share (count format: length nearly constant)
        for sh in (.25, .375, .5, .625, .75):
            s = (fmt == 'count') & np.isclose(shares, sh)
            sub = [c for c, x in zip(cases, s) if x]
            out[f'fixed_share{sh}_count_{mode}'] = run_block(Xl[s], sub, L[s], seeds[s], ('log2_n',), p, seed=seed)
        if padded is not None:
            Xp, cp, Lp = padded
            sp = np.array([c['seed'] for c in cp])
            out[f'padded_seed_{mode}'] = run_block(Xp, cp, Lp, sp, TARGETS, p, nonlinear=nonlinear, seed=seed)
            shp = np.array([c['share'] for c in cp]); nnp = np.array([c['n'] for c in cp])
            for sh in (.25, .375, .5, .625, .75):
                s = np.isclose(shp, sh)
                out[f'fixed_share{sh}_padded_{mode}'] = run_block(Xp[s], [c for c, x in zip(cp, s) if x], Lp[s],
                                                                  sp[s], ('log2_n',), p, seed=seed)
            for n in (8, 16, 32, 64):
                s = nnp == n
                out[f'fixed_n{n}_padded_{mode}'] = run_block(Xp[s], [c for c, x in zip(cp, s) if x], Lp[s],
                                                             sp[s], ('k',), p, seed=seed)
    out['extrapolate_all'] = extrapolate(Xl, cases, seeds)
    s = fmt == 'count'
    out['extrapolate_count'] = extrapolate(Xl[s], [c for c, x in zip(cases, s) if x], seeds[s])
    return out


# ----------------------------------------------------------------------------- behavior summary

def _logit(p):
    p = np.clip(np.asarray(p, float), .005, .995)
    return np.log(p / (1 - p))


def behavior_summary(cases, p, padded_p=None, boot=1000, seed=7):
    y = _logit(p); norm = np.array([prompts.log_odds(c) for c in cases])
    share = np.array([c['share'] for c in cases]); n = np.array([c['n'] for c in cases], float)
    clusters = np.array([f"{c['seed']}-{c['domain']}" for c in cases])

    def stats(idx):
        X = np.column_stack([norm[idx], np.log(share[idx] / (1 - share[idx])), np.ones(idx.sum())])
        c, d, _ = np.linalg.lstsq(X, y[idx], rcond=None)[0]
        # within-share n sensitivity: logit on sign(share - .5) * log2 n with one intercept per share (share != .5);
        # the same regression on the normative log-odds gives the normative reference
        off = idx & ~np.isclose(share, .5)
        sgn = np.sign(share[off] - .5)
        D = np.column_stack([sgn * np.log2(n[off])] + [np.isclose(share[off], s).astype(float) for s in np.unique(share[off])])
        wn = np.linalg.lstsq(D, y[off], rcond=None)[0][0]
        nn_ = np.linalg.lstsq(D, norm[off], rcond=None)[0][0]
        return {'slope_normative': float(np.polyfit(norm[idx], y[idx], 1)[0]), 'c_normative': float(c),
                'd_logit_share': float(d), 'n_sensitivity_fixed_share': float(wn),
                'normative_n_sensitivity_fixed_share': float(nn_),
                'corr_share': float(np.corrcoef(share[idx], y[idx])[0, 1]),
                'corr_normative': float(np.corrcoef(norm[idx], y[idx])[0, 1]),
                'output_logit_sd': float(np.std(y[idx])), 'output_p_range': [float(np.min(p[idx])), float(np.max(p[idx]))]}

    out = {}
    fmt = np.array([c['format'] for c in cases])
    rng = np.random.default_rng(seed); uc = np.unique(clusters)
    for label, idx in (('all', np.ones(len(cases), bool)), ('listed', fmt == 'listed'), ('count', fmt == 'count')):
        s = stats(idx)
        draws = []
        for _ in range(boot):
            pick = rng.choice(uc, len(uc))
            sel = np.concatenate([np.flatnonzero(idx & (clusters == u)) for u in pick])
            draws.append(np.polyfit(norm[sel], y[sel], 1)[0])
        s['slope_normative_ci95'] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        s['logit_error_by_n'] = {str(v): float(np.mean(y[idx & (n == v)] - norm[idx & (n == v)]))
                                 for v in sorted(set(n[idx].astype(int)))}
        out[label] = s
    out['ci_method'] = f'cluster bootstrap over seed x domain ({len(uc)} clusters, {boot} draws)'
    if padded_p is not None:
        ids = [i for i, c in enumerate(cases) if c['id'] in padded_p]
        if ids:
            d = np.array([_logit(padded_p[cases[i]['id']]) - y[i] for i in ids])
            nn_ = n[ids]
            out['padding_effect_logit_by_n'] = {str(int(v)): float(d[nn_ == v].mean()) for v in sorted(set(nn_))}
    return out


# ----------------------------------------------------------------------------- summary and plot

def headline(layers):
    """Max over layers (optimistic: layer chosen on the same CV scores) and last-layer value."""
    keys = {}
    for li, blk in enumerate(layers):
        for a, tgt in blk.items():
            if a.startswith('extrapolate') or not isinstance(tgt, dict): continue
            for t, row in tgt.items():
                for metric in ('r2', 'nuisance_r2', 'selectivity', 'nl_r2', 'nl_selectivity', 'control_r2', 'r2_joint'):
                    if metric in row:
                        keys.setdefault((a, t, metric), []).append((li, row[metric]))
    out = {}
    for (a, t, metric), vals in keys.items():
        li, v = max(vals, key=lambda x: -np.inf if np.isnan(x[1]) else x[1])
        out.setdefault(a, {}).setdefault(t, {})[metric] = {'max': float(v), 'layer': li, 'last': float(vals[-1][1])}
    # mean over n of k-at-fixed-n, and over shares of n-at-fixed-share, per layer; then max/last
    for fam, pat, tgt in (('fixed_n_mean', 'fixed_n{}_{}_{}', 'k'), ('fixed_share_mean', 'fixed_share{}_{}_{}', 'log2_n')):
        vals = (8, 16, 32, 64) if fam == 'fixed_n_mean' else (.25, .375, .5, .625, .75)
        for f in ('count', 'listed', 'padded'):
            for mode in ('raw', 'partial'):
                per = []
                for blk in layers:
                    rs = [blk[pat.format(v, f, mode)][tgt]['r2'] for v in vals if pat.format(v, f, mode) in blk]
                    if rs: per.append(float(np.nanmean(rs)) if not np.isnan(rs).all() else np.nan)
                if per and not np.isnan(per).all():
                    li = int(np.nanargmax(per))
                    out[f'{fam}_{f}_{mode}'] = {tgt: {'r2': {'max': per[li], 'layer': li, 'last': per[-1]}}}
    return out


def plot(layers, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    L = np.arange(len(layers))
    g = lambda a, t, m='r2': [blk.get(a, {}).get(t, {}).get(m, np.nan) for blk in layers]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, lab, st in (('all_seed_raw', 'raw', '-'), ('all_seed_partial', 'length-residualized', '--'),
                       ('count_seed_partial', 'count fmt, residualized', ':'), ('padded_seed_partial', 'padded, residualized', '-.')):
        if a in layers[0]:
            ax[0].plot(L, g(a, 'log2_n'), st, label=f'log2 n {lab}')
    ax[0].plot(L, g('all_seed_raw', 'log2_n', 'nuisance_r2'), c='gray', label='length/template only')
    ax[0].set_title('n')
    for f, st in (('listed', '-'), ('count', '--'), ('padded', ':')):
        per = [np.nanmean([blk[f'fixed_n{v}_{f}_partial']['k']['r2'] for v in (8, 16, 32, 64)])
               if f'fixed_n8_{f}_partial' in blk else np.nan for blk in layers]
        ax[1].plot(L, per, st, label=f'k at fixed n, {f} (residualized)')
    for f, st in (('count', '--'), ('padded', ':')):
        per = [np.nanmean([blk[f'fixed_share{v}_{f}_partial']['log2_n']['r2'] for v in (.25, .375, .5, .625, .75)])
               if f'fixed_share0.25_{f}_partial' in blk else np.nan for blk in layers]
        ax[1].plot(L, per, st, label=f'log2 n at fixed share, {f} (residualized)')
    ax[1].set_title('fixed-n / fixed-share')
    for t in ('log2_n', 'k', 'normative_log_odds', 'share'):
        ax[2].plot(L, g('all_seed_partial', t, 'selectivity'), label=f'{t} linear')
        ax[2].plot(L, g('all_seed_partial', t, 'nl_selectivity'), ':', label=f'{t} RBF')
    ax[2].set_title('selectivity vs control task (residualized)')
    for a in ax:
        a.axhline(0, c='k', lw=.5); a.set_xlabel('residual layer (0 = input)'); a.set_ylabel('held-out R²')
        a.set_ylim(-0.3, 1.02); a.legend(fontsize=6)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------------- main

def load_cases(root, allow_proxy=False):
    meta = json.loads((root / 'meta.json').read_text())
    if meta['generator_sha256'] != prompts.GENERATOR_SHA256:
        raise RuntimeError('E1 generator changed; cache invalid')
    log = {r['id']: r for r in map(json.loads, (root / 'behavior.jsonl').read_text().splitlines())}
    cases = [c for c in prompts.cases() if c['id'] in log and (root / 'residual' / (c['id'] + '.npy')).exists()]
    return meta, log, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--tokens-only', action='store_true', help='write tokens.json (needs tokenizer) and exit')
    ap.add_argument('--ensure-tokens', action='store_true', help='compute tokens.json first if missing (model image)')
    ap.add_argument('--allow-char-proxy', action='store_true', help='tests only: char length / 4 instead of tokens')
    ap.add_argument('--linear-only', action='store_true')
    ap.add_argument('--layers', default='', help='comma-separated subset (default all)')
    args = ap.parse_args()
    root = Path(args.out) / args.model
    meta, log, cases = load_cases(root)
    tok_path = root / 'tokens.json'
    if args.tokens_only or (args.ensure_tokens and (not tok_path.exists() or not all(
            c['id'] in json.loads(tok_path.read_text())['features'] for c in cases[:1] + cases[-1:]))):
        compute_tokens(args.model, cases, tok_path); print(tok_path, flush=True)
        if args.tokens_only: return
    if len(cases) < 50: raise ValueError(f'Only {len(cases)} cases; >=50 needed')
    t0 = time.time()
    L, length_source = token_lengths(root, cases, allow_proxy=args.allow_char_proxy)
    X = np.stack([np.load(root / 'residual' / (c['id'] + '.npy')) for c in cases])   # float16 [N, layers+1, d]
    seeds = np.array([c['seed'] for c in cases]); templates = np.array([c['template'] for c in cases])
    # padded listed prompts (n < 64) captured by patch.py --mode padded; n = 64 listed prompts are already the target length
    pdir = root / 'residual_padded'
    padded_cases = [c for c in cases if c['format'] == 'listed' and c['n'] < 64 and (pdir / (c['id'] + '.npy')).exists()]
    have_padded = len(padded_cases) >= 50
    if have_padded:
        n64 = [c for c in cases if c['format'] == 'listed' and c['n'] == 64]
        cp = padded_cases + n64
        Xp = np.stack([np.load(pdir / (c['id'] + '.npy')) for c in padded_cases] +
                      [X[cases.index(c)] for c in n64])
        Lp = np.concatenate([token_lengths(root, padded_cases, padded=True, allow_proxy=args.allow_char_proxy)[0],
                             token_lengths(root, n64, allow_proxy=args.allow_char_proxy)[0]])
    layer_ids = [int(x) for x in args.layers.split(',')] if args.layers else list(range(X.shape[1]))
    layers, W = [], {t: [] for t in DIRECTION_TARGETS}
    Bias = {t: [] for t in DIRECTION_TARGETS}
    tv = target_values(cases)
    for li in layer_ids:
        Xl = X[:, li].astype(np.float32)
        pad = (Xp[:, li].astype(np.float32), cp, Lp) if have_padded else None
        blk = layer_block(Xl, cases, L, seeds, templates, pad, nonlinear=not args.linear_only, seed=li)
        blk['layer'] = li
        layers.append(blk)
        for t in DIRECTION_TARGETS:   # raw-space probes on all prompts (for patch.py subspaces and readouts)
            w, b, _ = fit_direction(Xl, tv[t], seeds)
            W[t].append(w); Bias[t].append(b)
        print(f'layer {li} done {time.time() - t0:.0f}s', flush=True)
    np.savez(root / 'probes_v2.npz', layers=np.array(layer_ids),
             **{f'w_{t}': np.stack(W[t]) for t in DIRECTION_TARGETS}, **{f'b_{t}': np.array(Bias[t]) for t in DIRECTION_TARGETS})
    (root / 'probe_v2_layers.json').write_text(json.dumps(layers))
    p = np.array([log[c['id']]['p_yes'] for c in cases])
    padded_p = None
    if (root / 'behavior_padded.jsonl').exists():
        padded_p = {r['id']: r['p_yes'] for r in map(json.loads, (root / 'behavior_padded.jsonl').read_text().splitlines())}
    summary = {'spec': 'v3.4', 'probe_version': PROBE_VERSION, 'model': meta['model'], 'n_cases': len(cases),
               'n_seeds': int(len(set(seeds))), 'length_source': length_source, 'padded_available': have_padded,
               'n_padded': len(padded_cases), 'clip': [.005, .995],
               'cv': 'outer: leave-one-seed-out (or leave-one-template-out); ridge penalty by inner leave-one-group-out '
                     'within the outer training fold; R^2 pooled over held-out folds; per-fold R^2 in r2_folds',
               'nuisance_features': 'format x domain indicators; relative token length (minus the mean of the same '
                                    'format x domain x template wording) x format x domain',
               'control_task': f'{N_CONTROL} permutations of cell-level target values across (n,k,r) cells',
               'nonlinear': f'RBF kernel ridge, gamma in {RBF_SCALES} / median squared distance, penalty grid as linear; '
                            'chosen by the same nested CV',
               'selectivity': 'r2 - max(control_r2, 0)',
               'headline_note': 'max over layers is optimistic (layer chosen on the reported CV scores); last-layer value also given',
               'behavior': behavior_summary(cases, p, padded_p),
               'headline': headline(layers), 'layers': layers,
               'seconds': time.time() - t0}
    (root / 'probe_v2.json').write_text(json.dumps(summary, indent=1))
    title = f"{meta['model']} probe v2 ({len(cases)} prompts, {len(set(seeds))} seeds, {length_source})"
    plot(layers, root / 'probe_v2.png', title); plot(layers, root / 'probe_v2.pdf', title)
    print(json.dumps({'behavior': summary['behavior'], 'seconds': summary['seconds']}, indent=1))


if __name__ == '__main__': main()
