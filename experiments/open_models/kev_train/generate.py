"""Synthetic, normative-only curricula. Training wording is disjoint from all three E1 eval templates."""
import argparse
import json
import math
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import e1
from prompts import QUESTION

R_VALUES = (.52, .55, .6, .7, .8)


def example(i, curriculum, seed):
    rng = random.Random(f'kev-e5b:{seed}:{i}')
    n = rng.choice((4, 8, 16, 32)) if curriculum == 'varied' else 8
    f = rng.choice([x for x in e1.SHARES if x * n == round(x * n)])
    k = round(n * f)
    r = rng.choice(R_VALUES)
    domain = rng.choice(e1.DOMAIN_ORDER); D = e1.DOMAINS[domain]
    yes = [D['pos']] * k + [D['neg']] * (n-k); rng.shuffle(yes)
    intro = (f'Exercise: classify a {D["item"]}. A YES world is {D["pos_state"]}; '
             f'a NO world is {D["neg_state"]}. Initially the worlds are equally probable. '
             f'The synthetic {D["reporters"]} make conditionally independent predictions with no shared errors. '
             f'For either world, an individual prediction matches that world with chance {r:.2f} '
             f'({int(r*100)}%). There are {n} reports. The two permitted report words are {D["pos"]} and {D["neg"]}.')
    if rng.random() < .5:
        evidence = '\n'.join(f'Observation {j+1}: {word}' for j, word in enumerate(yes))
    else:
        evidence = f'Tally: {k} observations are {D["pos"]}, and {n-k} are {D["neg"]}.'
    p = e1.posterior(k, n, r)
    return {'state': intro + '\n' + evidence,
            'questions': {'answer': {'type': 'noul', 'instructions': QUESTION, 'label': rng.random() < p}}}


def generate(total=4000, seed=155):
    return {curriculum: [example(i, curriculum, seed) for i in range(total)]
            for curriculum in ('varied', 'fixed8')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/training')
    ap.add_argument('--total', type=int, default=4000); ap.add_argument('--seed', type=int, default=155)
    args = ap.parse_args(); folder = Path(args.out); folder.mkdir(parents=True, exist_ok=True)
    for curriculum, rows in generate(args.total, args.seed).items():
        with (folder / (curriculum + '.jsonl')).open('w') as f:
            for row in rows: f.write(json.dumps(row) + '\n')
    (folder / 'metadata.json').write_text(json.dumps({'spec': 'v3.4', 'seed': args.seed,
         'samples_per_curriculum': args.total, 'labels': 'Bernoulli(normative posterior), never Jev',
         'evaluation_templates_used_for_training': False}, indent=2))

if __name__ == '__main__': main()
