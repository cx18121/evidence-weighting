"""Evaluate untouched E1 v3.1 (both formats, including n=64) plus decisive E3."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prompts
from runtime import load_model, score


def summarize(rows):
    p = np.clip([x['p_yes'] for x in rows], .005, .995)
    logits = np.log(p / (1-p)); truth = np.array([x['normative_log_odds'] for x in rows])
    summary = {'spec': 'v3.4', 'n': len(rows), 'clip': [.005, .995],
               'logit_error_by_n': {}, 'slope': float(np.polyfit(truth, logits, 1)[0]), 'gate': {}}
    for n in sorted({x['n'] for x in rows}):
        sel = np.array([x['n'] == n for x in rows]); summary['logit_error_by_n'][str(n)] = float(np.mean(logits[sel]-truth[sel]))
    for threshold in (.9, .95):
        gt = np.array([x['truth'] >= threshold for x in rows]); pred = np.asarray(p) >= threshold
        summary['gate'][str(threshold)] = {'misses': int((gt & ~pred).sum()), 'false_alarms': int((~gt & pred).sum()),
                                         'positive_n': int(gt.sum()), 'negative_n': int((~gt).sum())}
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='results/trained_evaluation'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bfloat16'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed-count', type=int, default=6)
    args = ap.parse_args()
    if args.model.startswith('jaredpalmer/kev-') is False and not (Path(args.model)/'head.pt').exists():
        raise ValueError('evaluation requires a Kev checkpoint')
    tok, model, kev = load_model(args.model, args.device, args.dtype)
    out = Path(args.out) / args.model.replace('/', '_'); out.mkdir(parents=True, exist_ok=True)
    for exp in ('e1','e3'):
        path = out / (exp+'.jsonl'); prior = {x['id'] for x in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
        cs = [c for c in prompts.cases(exp) if c['seed_idx'] < args.seed_count]
        if args.limit: cs = cs[:args.limit]
        with path.open('a') as f:
            for c in cs:
                if c['id'] in prior: continue
                prob = score(tok, model, kev, c)
                row = {'spec': 'v3.4', 'id': c['id'], 'generator_sha256': prompts.GENERATOR_SHA256,
                       'exp': exp, 'n': c['n'], 'seed': c['seed'], 'template': c['template'],
                       'truth': c['truth'], 'normative_log_odds': prompts.log_odds(c), 'p_yes': prob,
                       'format': c['format'], 'r': c['r']}
                f.write(json.dumps(row)+'\n'); f.flush()
        rows = list(map(json.loads, path.read_text().splitlines()))
        (out / (exp + '_summary.json')).write_text(json.dumps(summarize(rows), indent=2))
        print(exp, summarize(rows))

if __name__ == '__main__': main()
