"""Unit tests for generate.py (spec v3.1).  Run: python3 -m unittest -v test_generate"""
import itertools
import math
import re
import unittest
from collections import defaultdict

from generate import (DOMAINS, E3_N, N_GRID, PAD_TOL, R_GRID, SEEDS, SHARES, generate, posterior,
                      posterior_brute, posterior_reports)

CASES = generate()
E1 = [c for c in CASES if c["exp"] == "e1"]
E3 = [c for c in CASES if c["exp"] == "e3"]


class Normative(unittest.TestCase):
    def test_anchor(self):
        self.assertAlmostEqual(posterior(48, 64, 0.55), 0.99838, places=5)
        self.assertAlmostEqual(posterior(32, 64, 0.55), 0.5)

    def test_matches_brute_force_and_symmetry(self):
        for r in R_GRID:
            for n in N_GRID:
                for k in range(n + 1):
                    self.assertAlmostEqual(posterior(k, n, r), posterior_brute(k, n, r), places=12)
                    self.assertAlmostEqual(posterior(k, n, r) + posterior(n - k, n, r), 1, places=12)

    def test_case_truth_recomputed_from_prompt(self):
        for c in E1:
            D = DOMAINS[c["domain"]]
            if c["format"] == "listed":
                k = len(re.findall(rf': {D["pos"]}$', c["state"], re.M))
                n = k + len(re.findall(rf': {D["neg"]}$', c["state"], re.M))
            else:
                m = re.search(rf'(\d+) of the (\d+) reports say "{D["pos"]}"', c["state"])
                if m:
                    k, n = int(m.group(1)), int(m.group(2))
                else:
                    m = re.search(rf'(\d+) of the (\d+) reports say "{D["neg"]}"', c["state"])
                    n = int(m.group(2)); k = n - int(m.group(1))
            self.assertEqual((k, n), (c["k"], c["n"]))
            self.assertIn(f'{c["r"]:.2f} ({round(c["r"] * 100)}%)', c["state"])
            self.assertAlmostEqual(c["truth"], posterior_brute(k, n, c["r"]), places=12)

    def test_e3_truth(self):
        for c in E3:
            rel = [int(x) / 100 for x in re.findall(r"\(reliability (\d+)%\)", c["state"])]
            D = DOMAINS[c["domain"]]
            signs = [1 if s == D["pos"] else 0 for s in re.findall(r": (\w+) \(reliability", c["state"])]
            self.assertEqual(len(rel), c["n_reports"])
            self.assertAlmostEqual(c["truth"], posterior_reports(list(zip(signs, rel))), places=12)
            if c["control"] == "weak_only":
                self.assertAlmostEqual(c["truth"], 0.5)
                self.assertEqual(rel.count(0.99), 0)
            else:
                self.assertAlmostEqual(c["truth"], 0.99 if c["decisive"] else 0.01)
                self.assertEqual(rel.count(0.99), 1)
                self.assertEqual(rel.index(0.99), 0 if c["location"] == "early" else len(rel) - 1)
                self.assertEqual(c["n_reports"] % 2, 1)


class Design(unittest.TestCase):
    def test_grid_size(self):
        cells = [(n, int(n * f)) for n in N_GRID for f in SHARES if n * f == int(n * f)]
        self.assertEqual(len(cells), 23)
        self.assertEqual(len(E1), 23 * 3 * 2 * len(SEEDS) * len(R_GRID))
        self.assertEqual(len(SEEDS), 6)
        self.assertEqual(len({c["id"] for c in CASES}), len(CASES))

    def test_disjoint_labels(self):
        for d, D in DOMAINS.items():
            self.assertNotIn(D["pos"], D["neg"]); self.assertNotIn(D["neg"], D["pos"])
            self.assertNotIn(D["pos"].lower(), D["neg"].lower())
        self.assertEqual({(D["pos"], D["neg"]) for D in DOMAINS.values()},
                         {("contaminated", "clean"), ("recommend", "avoid"), ("malicious", "benign")})

    def test_distinct_prompts_across_seeds(self):
        by_cell = defaultdict(set)
        for c in E1:
            by_cell[(c["domain"], c["n"], c["k"], c["r"], c["format"])].add(c["state"])
        for cell, states in by_cell.items():
            self.assertEqual(len(states), len(SEEDS), cell)
        for c in E3:
            pass
        by_cell = defaultdict(set)
        for c in E3:
            by_cell[(c["domain"], c["n"], c["decisive"], c["location"], c["control"])].add(c["state"])
        for cell, states in by_cell.items():
            self.assertEqual(len(states), len(SEEDS), cell)

    def test_three_templates_balanced(self):
        seen = defaultdict(set)
        for c in E1:
            seen[c["domain"]].add(c["template"])
        for d in DOMAINS:
            self.assertEqual(seen[d], {0, 1, 2})
        per = defaultdict(int)
        for c in E1:
            if c["n"] == 4 and c["k"] == 1 and c["format"] == "listed" and c["r"] == 0.55:
                per[(c["domain"], c["template"])] += 1
        self.assertEqual(set(per.values()), {2})

    def test_listed_signs_ids(self):
        for c in E1:
            if c["format"] == "listed":
                self.assertEqual(sum(c["signs"]), c["k"]); self.assertEqual(len(set(c["ids"])), c["n"])
                for i in c["ids"]:
                    self.assertEqual(c["state"].count(f" {i}: "), 1)

    def test_story_states_model(self):
        for c in E1 + E3:
            s = c["state"].lower()
            self.assertIn("1/2", s); self.assertIn("independen", s); self.assertIn("fictional" if "fictional" in s else "simulated", s)
            self.assertTrue("no shared errors" in s or "no common or shared" in s or "no errors are shared" in s)

    def test_padding_exact_length(self):
        n_pad = 0
        for c in E1:
            if c["format"] != "listed":
                continue
            L, T = len(c["state_padded"]), c["pad_target_len"]
            self.assertLessEqual(abs(L - T), PAD_TOL * T, c["id"])
            self.assertTrue(c["state_padded"].startswith(c["state"]))
            if c["n"] < 64:
                self.assertGreater(c["pad_lines"], 0); n_pad += 1
                self.assertNotIn(DOMAINS[c["domain"]]["pos"], c["state_padded"][len(c["state"]):])
                self.assertNotIn(DOMAINS[c["domain"]]["neg"], c["state_padded"][len(c["state"]):])
        self.assertGreater(n_pad, 0)
        for c in E3:
            if c["control"] == "padding":
                self.assertLessEqual(abs(len(c["state"]) - c["pad_target_len"]), PAD_TOL * c["pad_target_len"])
                self.assertIn("Log 1:", c["state"])

    def test_e3_matched_weak_reports(self):
        by = {c["id"]: c for c in E3}
        for c in E3:
            if c["control"] != "decisive":
                continue
            wk = by[c["id"].replace(f'-{c["location"]}-decisive', "-none-weak_only")]
            weak = c["signs"][1:] if c["location"] == "early" else c["signs"][:-1]
            self.assertEqual(weak, wk["signs"]); self.assertEqual(sum(weak) * 2, len(weak))
        self.assertEqual(len([c for c in E3 if c["control"] == "decisive"]), 5 * 2 * 2 * 3 * 6)

    def test_deterministic(self):
        self.assertEqual(generate(), CASES)


if __name__ == "__main__":
    unittest.main()
