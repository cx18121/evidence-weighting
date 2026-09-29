# Independent E5 training audit

Historical E1 generator SHA-256 at audit time: `b3118de7cc22011633aeb133e18e97b5c63e34a617dd936ff37b353289e95014`. The released generator differs in its documentation header. The audit checked the historical generated held-out case IDs, truths, predictions, and sample counts. Those row-level records are not distributed here.
Training used 4,000 synthetic examples per arm and only n=4, 8, 16, 32. Evaluation used three separate seeds and three held-out story templates, including n=64. Training and evaluation share domains and report reliabilities.
Deviation is mean squared distance from the specified posterior, not observed-outcome Brier. For Bernoulli outcomes generated from that posterior, paired expected-Brier differences equal the differences shown.
Intervals are 95% paired bootstrap intervals grouped by domain, reliability, count, list size and format. A negative change favors training.

## Kev-08b

| Arm | E1 all deviation | E1 n=64 deviation | Change from base, all [95% CI] | Change from base, n=64 [95% CI] | E3 deviation |
|---|---:|---:|---:|---:|---:|
| base | 0.0869 | 0.1468 | reference | reference | 0.1807 |
| fixed8 | 0.0613 | 0.1107 | -0.0255 [-0.0319, -0.0194] (414 groups) | -0.0361 [-0.0534, -0.0191] (90 groups) | 0.1693 |
| varied | 0.0463 | 0.0769 | -0.0406 [-0.0470, -0.0344] (414 groups) | -0.0698 [-0.0880, -0.0523] (90 groups) | 0.1881 |

Varied minus fixed8 E1 deviation: -0.0150 [-0.0178, -0.0124] overall; -0.0338 [-0.0416, -0.0258] at unseen n=64.
At the 0.95 E1 probability gate, detected true positives out of 180: base 0, fixed8 0, varied 0.

## Kev-4b

| Arm | E1 all deviation | E1 n=64 deviation | Change from base, all [95% CI] | Change from base, n=64 [95% CI] | E3 deviation |
|---|---:|---:|---:|---:|---:|
| base | 0.0493 | 0.0770 | reference | reference | 0.1337 |
| fixed8 | 0.0256 | 0.0503 | -0.0237 [-0.0279, -0.0198] (414 groups) | -0.0267 [-0.0387, -0.0156] (90 groups) | 0.1214 |
| varied | 0.0175 | 0.0278 | -0.0318 [-0.0364, -0.0272] (414 groups) | -0.0492 [-0.0630, -0.0361] (90 groups) | 0.1254 |

Varied minus fixed8 E1 deviation: -0.0081 [-0.0100, -0.0062] overall; -0.0226 [-0.0271, -0.0180] at unseen n=64.
At the 0.95 E1 probability gate, detected true positives out of 180: base 0, fixed8 0, varied 1.

A lower error does not establish a reasoning mechanism. E3 was not part of the training task. The training arms use one seed and 300 optimization steps each; these are exploratory runs, not a multi-seed method comparison. The curricula differ in n distribution, so this is not a controlled test of all possible training-data explanations.
