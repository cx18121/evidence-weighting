"""Check paired, held-out Kev evaluations against the original synthetic cases."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prompts

RESULTS = Path(__file__).resolve().parents[1] / 'results'
EVAL = RESULTS / 'trained_evaluation'


def load(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {row['id']: row for row in rows}
    assert len(by_id) == len(rows) == len(expected), (path, len(rows), len(expected))
    assert set(by_id) == set(expected), path
    for id_, row in by_id.items():
        case = expected[id_]
        assert row['n'] == case['n'] and row['format'] == case['format']
        assert math.isclose(row['truth'], case['truth'], abs_tol=1e-12)
        assert math.isfinite(row['p_yes']) and 0 <= row['p_yes'] <= 1
        if 'generator_sha256' in row:
            assert row['generator_sha256'] == prompts.GENERATOR_SHA256
    return by_id


def path_for(size, arm, experiment):
    if arm == 'base':
        if size == '08b':
            return EVAL / 'jaredpalmer_kev-0.8b' / f'{experiment}.jsonl'
        if experiment == 'e1':
            return RESULTS / 'kev4' / 'behavior.jsonl'
        return EVAL / 'jaredpalmer_kev-4b' / 'e3.jsonl'
    return EVAL / f'_vol_out_results_trained_{size}-{arm}' / f'{experiment}.jsonl'


def mse(rows, ids):
    return sum((rows[i]['p_yes'] - rows[i]['truth']) ** 2 for i in ids) / len(ids)


def interval(a, b, cases, ids):
    """Paired bootstrap, grouping repeat orderings of the same evidence counts."""
    groups = defaultdict(list)
    for i in ids:
        c = cases[i]
        key = (c['domain'], c['r'], c['n'], c['k'], c['format']) if c['exp'] == 'e1' else (
            c['domain'], c['n'], c['truth'], c['format'], c.get('variant', i))
        groups[key].append((b[i]['p_yes'] - b[i]['truth']) ** 2 - (a[i]['p_yes'] - a[i]['truth']) ** 2)
    means = [sum(values) / len(values) for values in groups.values()]
    rng = random.Random(47)
    trials = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(2000))
    return trials[49], trials[1949], len(means)


def main():
    expected = {exp: {c['id']: c for c in prompts.cases(exp)} for exp in ('e1', 'e3')}
    assert len(expected['e1']) == 2484 and len(expected['e3']) == 828
    selected = {exp: {i: c for i, c in cases.items() if c['seed_idx'] < 3}
                for exp, cases in expected.items()}
    assert len(selected['e1']) == 1242 and len(selected['e3']) == 414
    lines = ['# Independent E5 training audit', '',
             f'E1 generator SHA-256: `{prompts.GENERATOR_SHA256}`. All IDs, truths, predictions, and sample counts checked against the generated held-out cases.',
             'Training used 4,000 synthetic examples per arm and only n=4, 8, 16, 32. Evaluation used three separate seeds and three held-out story templates, including n=64. Training and evaluation share domains and report reliabilities.',
             'Deviation is mean squared distance from the specified posterior, not observed-outcome Brier. For Bernoulli outcomes generated from that posterior, paired expected-Brier differences equal the differences shown.',
             'Intervals are 95% paired bootstrap intervals grouped by domain, reliability, count, list size and format. A negative change favors training.', '']
    for size in ('08b', '4b'):
        b = load(path_for(size, 'base', 'e1'), selected['e1'])
        be3 = load(path_for(size, 'base', 'e3'), selected['e3'])
        lines += [f'## Kev-{size}', '',
                  '| Arm | E1 all deviation | E1 n=64 deviation | Change from base, all [95% CI] | Change from base, n=64 [95% CI] | E3 deviation |',
                  '|---|---:|---:|---:|---:|---:|']
        arms = {'base': b}
        ids = list(selected['e1']); ids64 = [i for i in ids if selected['e1'][i]['n'] == 64]
        for arm in ('base', 'fixed8', 'varied'):
            a = b if arm == 'base' else load(path_for(size, arm, 'e1'), selected['e1'])
            arms[arm] = a
            e3 = be3 if arm == 'base' else load(path_for(size, arm, 'e3'), selected['e3'])
            alle = mse(a, ids); large = mse(a, ids64)
            if arm == 'base':
                delta, delta64 = 'reference', 'reference'
            else:
                lo, hi, groups = interval(b, a, selected['e1'], ids)
                lo64, hi64, groups64 = interval(b, a, selected['e1'], ids64)
                delta = f'{alle - mse(b, ids):+.4f} [{lo:+.4f}, {hi:+.4f}] ({groups} groups)'
                delta64 = f'{large - mse(b, ids64):+.4f} [{lo64:+.4f}, {hi64:+.4f}] ({groups64} groups)'
            lines.append(f'| {arm} | {alle:.4f} | {large:.4f} | {delta} | {delta64} | {mse(e3, list(selected["e3"])):.4f} |')
        lines.append('')
        fixed, varied = arms['fixed8'], arms['varied']
        lo, hi, _ = interval(fixed, varied, selected['e1'], ids)
        lo64, hi64, _ = interval(fixed, varied, selected['e1'], ids64)
        lines.append(f'Varied minus fixed8 E1 deviation: {mse(varied, ids)-mse(fixed, ids):+.4f} [{lo:+.4f}, {hi:+.4f}] overall; {mse(varied, ids64)-mse(fixed, ids64):+.4f} [{lo64:+.4f}, {hi64:+.4f}] at unseen n=64.')
        positives = [i for i in ids if selected['e1'][i]['truth'] >= .95]
        gate = {arm: sum(arms[arm][i]['p_yes'] >= .95 for i in positives) for arm in arms}
        lines.append(f'At the 0.95 E1 probability gate, detected true positives out of {len(positives)}: base {gate["base"]}, fixed8 {gate["fixed8"]}, varied {gate["varied"]}.')
        lines.append('')
    lines += ['A lower error does not establish a reasoning mechanism. E3 was not part of the training task. The training arms use one seed and 300 optimization steps each; these are exploratory runs, not a multi-seed method comparison. The curricula differ in n distribution, so this is not a controlled test of all possible training-data explanations.', '']
    path = RESULTS / 'training-audit.md'
    path.write_text('\n'.join(lines))
    print(path)


if __name__ == '__main__':
    main()
