"""Pure CPU tests; never load/download a pretrained model."""
import math
import unittest
import numpy as np
import prompts
from probe import cv_probe, layer_analysis
from kev_train.generate import generate
from kev_train.evaluate import summarize
from runtime import layer_hook
import torch

class GeneratorTests(unittest.TestCase):
    def test_identity_and_grid(self):
        cs = prompts.cases()
        self.assertEqual(len(cs), 23*3*2*6*3)
        self.assertEqual(len({c['state'] for c in cs}), len(cs))
        self.assertEqual(len(list(prompts.matches())), len(list(prompts.matches())))
        for c in cs:
            self.assertAlmostEqual(1/(1+math.exp(-prompts.log_odds(c))), c['truth'], places=12)
            self.assertEqual(sum(c['signs']), c['k']) if c['format']=='listed' else None
        for s,l in prompts.matches():
            self.assertEqual((s['share'],s['r'],s['domain'],s['seed'],s['format']),
                             (l['share'],l['r'],l['domain'],l['seed'],l['format']))
            self.assertLess(s['n'], l['n'])

    def test_curricula(self):
        a = generate(200, 33); b = generate(200, 33)
        self.assertEqual(a, b)
        for row in a['fixed8']:
            self.assertIn('Exercise:', row['state'])
            self.assertNotIn('Case ', row['state'])
            self.assertIs(type(row['questions']['answer']['label']), bool)
            self.assertIn('There are 8 reports.', row['state'])

class ProbeTests(unittest.TestCase):
    def test_group_cv_and_shuffle(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=120); X = y[:,None] + rng.normal(size=(120,1))*.1
        groups = np.repeat(np.arange(6), 20)
        result, fitted = cv_probe(X, y, groups)
        self.assertGreater(result['r2'], .95)
        self.assertLess(result['shuffled_r2'], .2)
        self.assertAlmostEqual(fitted[0].predict(((X-fitted[1])/fitted[2]))[0], y[0], delta=.4)

    def test_n64_extrapolation_and_directions(self):
        rng = np.random.default_rng(8)
        cs = prompts.cases()[::2]
        n = np.array([c['n'] for c in cs]); k = np.array([c['k'] for c in cs]); r = np.array([prompts.log_odds(c) for c in cs])
        X = np.stack([k, n, n-k, k/n, r], 1).astype(np.float32)
        X = np.stack([X, X + rng.normal(0, .0001, X.shape)], 1).astype(np.float16)
        rows, dirs = layer_analysis(X, cs)
        self.assertEqual(len(rows), 2*5)
        self.assertGreater([x for x in rows if x['target']=='normative_log_odds' and x['layer']==0][0]['n64_r2'], .99)
        self.assertAlmostEqual(np.linalg.norm(dirs['layer_0']),1,places=4)
        self.assertEqual(summarize([{'n':8,'p_yes':.96,'truth':.96,'normative_log_odds':2.}])['gate']['0.95']['misses'],0)

class HookTests(unittest.TestCase):
    def test_final_token_only_and_hook_removed(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.layers = torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in range(3)])
                for a in self.layers: a.weight.data.copy_(torch.eye(2))
            def forward(self,x):
                for layer in self.layers: x=layer(x)
                return x
        model=Tiny(); x=torch.ones((1,2,2)); original=model(x).clone()
        with layer_hook(model,False,2,'residual',lambda _:torch.tensor([3.,4.])):
            y=model(x)
            self.assertTrue(torch.equal(y[0,0],original[0,0]))
            self.assertTrue(torch.equal(y[0,-1],torch.tensor([3.,4.])))
        self.assertTrue(torch.equal(model(x),original))

if __name__=='__main__': unittest.main()
