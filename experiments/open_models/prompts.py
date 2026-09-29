"""Exact E1 v3.1 prompts, imported from the E1 builder (no independent reimplementation)."""
import importlib.util
import hashlib
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'evidence_count' / 'generate.py'
spec = importlib.util.spec_from_file_location('e1_v31_generate', SOURCE)
e1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e1)
GENERATOR_SHA256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
QUESTION = 'Given these reports, is the hidden state YES? Answer Yes or No.'
NUMERIC_QUESTION = 'Give the probability as a number from 0 to 100, where 73 means 73%. Reply with the number only.'


def cases(exp='e1'):
    return [c for c in e1.generate() if c['exp'] == exp]


def log_odds(c):
    if c['exp'] == 'e1':
        return (2 * c['k'] - c['n']) * math.log(c['r'] / (1 - c['r']))
    return math.log(c['truth'] / (1 - c['truth']))


def targets(c):
    return {'k': c['k'], 'n': c['n'], 'n_minus_k': c['n'] - c['k'],
            'share': c['share'], 'normative_log_odds': log_odds(c)}


def matches():
    """Same domain/reliability/share/format/seed; small vs n=64, same E1 story template."""
    cs = cases()
    by = {(c['r'], c['seed_idx'], c['domain'], c['format'], c['share'], c['n']): c for c in cs}
    for c in cs:
        if c['n'] == 64 or c['share'] == .5:
            continue
        other = by.get((c['r'], c['seed_idx'], c['domain'], c['format'], c['share'], 64))
        if other:
            yield c, other
