# E4 Amazon real-data check — spec v3.2 (clean v3.1 calls)

**Secondary natural-set diagnostic:** Jev AUROC 0.718 [0.654, 0.779] at k=2 versus 0.930 [0.898, 0.958] at k=32; mean confidence for held-out positives 0.612→0.703. Discrimination improves, but no synthetic exponent or normative failure is inferred.
Amazon Reviews 2023, Health_and_Personal_Care (McAuley Lab). Seed 42709; 240 products sampled uniformly within held-out truth classes, not by extreme mean. Reviews ≥20 characters; product ≥60 eligible reviews. A deterministic 50/50 hash split precedes label calculation and selection; 32 nested, randomized shown reviews come exclusively from the shown pool. Truth is the mean of *all* held-out ratings (positive ≥4.0). Review text clipped at 1000 characters in prompts; ratings, title, metadata omitted.

Products: 240; held-out label 1/0: 120/120; held-out reviews per product median 53 (min 25, max 1388).
Held-out mean stars distribution (minimum, 10th, 25th, median, 75th, 90th, maximum): 1.527, 3.077, 3.581, 3.993, 4.436, 4.691, 4.963.
Distribution by class (10th, median, 90th): 0: 2.827, 3.575, 3.929; 1: 4.098, 4.436, 4.778.

All brackets are 95% bootstrap percentile intervals resampling products (400 replicates); AUROC replicates without both classes excluded. Numeric probabilities are *stated* for Llama and Haiku, TypeSafe Noul yes-probabilities for Jev. Invalid/refused calls excluded and counted.

## jev

| k | n (valid/240) | AUROC | Brier | mean confidence label 0 | mean confidence label 1 | invalid/missing |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 240 | 0.718 [0.654, 0.779] | 0.216 [0.198, 0.235] | 0.484 [0.450, 0.513] | 0.612 [0.591, 0.630] | 0 |
| 4 | 240 | 0.798 [0.736, 0.855] | 0.187 [0.168, 0.206] | 0.425 [0.390, 0.458] | 0.633 [0.611, 0.658] | 0 |
| 8 | 240 | 0.872 [0.824, 0.919] | 0.162 [0.144, 0.179] | 0.408 [0.373, 0.441] | 0.675 [0.650, 0.699] | 0 |
| 16 | 240 | 0.910 [0.871, 0.941] | 0.153 [0.139, 0.167] | 0.410 [0.382, 0.440] | 0.690 [0.668, 0.712] | 0 |
| 32 | 240 | 0.930 [0.898, 0.958] | 0.144 [0.128, 0.160] | 0.400 [0.371, 0.429] | 0.703 [0.683, 0.722] | 0 |

## llama

| k | n (valid/240) | AUROC | Brier | mean confidence label 0 | mean confidence label 1 | invalid/missing |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 240 | 0.684 [0.620, 0.743] | 0.286 [0.237, 0.335] | 0.442 [0.380, 0.517] | 0.706 [0.647, 0.772] | 0 |
| 4 | 240 | 0.794 [0.742, 0.843] | 0.198 [0.157, 0.232] | 0.373 [0.306, 0.434] | 0.752 [0.705, 0.803] | 0 |
| 8 | 240 | 0.874 [0.825, 0.917] | 0.156 [0.130, 0.184] | 0.381 [0.334, 0.433] | 0.796 [0.759, 0.828] | 0 |
| 16 | 240 | 0.878 [0.835, 0.915] | 0.157 [0.132, 0.186] | 0.421 [0.371, 0.469] | 0.826 [0.800, 0.850] | 0 |
| 32 | 240 | 0.910 [0.872, 0.940] | 0.140 [0.115, 0.168] | 0.397 [0.345, 0.448] | 0.832 [0.808, 0.854] | 0 |

## haiku

| k | n (valid/240) | AUROC | Brier | mean confidence label 0 | mean confidence label 1 | invalid/missing |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 240 | 0.698 [0.628, 0.761] | 0.238 [0.205, 0.269] | 0.450 [0.399, 0.511] | 0.641 [0.599, 0.684] | 0 |
| 4 | 240 | 0.781 [0.719, 0.838] | 0.193 [0.158, 0.224] | 0.398 [0.355, 0.449] | 0.676 [0.633, 0.718] | 0 |
| 8 | 240 | 0.849 [0.802, 0.890] | 0.167 [0.146, 0.193] | 0.386 [0.347, 0.431] | 0.705 [0.670, 0.739] | 0 |
| 16 | 240 | 0.889 [0.854, 0.922] | 0.152 [0.128, 0.177] | 0.406 [0.362, 0.451] | 0.755 [0.725, 0.782] | 0 |
| 32 | 240 | 0.906 [0.868, 0.936] | 0.150 [0.128, 0.171] | 0.426 [0.383, 0.467] | 0.782 [0.758, 0.800] | 0 |


## Per-review extraction and human-label-only count rule

Jev per-review accuracy at k=32 (threshold Noul ≥0.5 against human stars 4–5): 0.941 [0.935, 0.946], 7680 reviews from 240 products. Repeated prefixes not double counted.
k=2: count-based human-stars fraction logistic rule, training 120 products, test 120 disjoint products: AUROC 0.690 [0.610, 0.769], Brier 0.218 [0.195, 0.242]. This is an oracle-input comparison, since stars are concealed from models; no Jev values enter fitting.
k=4: count-based human-stars fraction logistic rule, training 120 products, test 120 disjoint products: AUROC 0.796 [0.717, 0.871], Brier 0.185 [0.161, 0.211]. This is an oracle-input comparison, since stars are concealed from models; no Jev values enter fitting.
k=8: count-based human-stars fraction logistic rule, training 120 products, test 120 disjoint products: AUROC 0.895 [0.841, 0.945], Brier 0.157 [0.137, 0.178]. This is an oracle-input comparison, since stars are concealed from models; no Jev values enter fitting.
k=16: count-based human-stars fraction logistic rule, training 120 products, test 120 disjoint products: AUROC 0.951 [0.917, 0.979], Brier 0.147 [0.131, 0.162]. This is an oracle-input comparison, since stars are concealed from models; no Jev values enter fitting.
k=32: count-based human-stars fraction logistic rule, training 120 products, test 120 disjoint products: AUROC 0.950 [0.910, 0.982], Brier 0.148 [0.134, 0.165]. This is an oracle-input comparison, since stars are concealed from models; no Jev values enter fitting.

## Spend / interpretation

Spec v3.1 natural-call cost including retries: jev $0.0746, llama $0.9101, haiku $0.9581; total $1.9428. 3600/3600 valid distinct calls; 3600 logged natural attempts. Costs estimated by provider helper. All v3.1 conditions share the exact mean-held-out-stars ≥4.0 holistic question; OpenRouter is Together-only.
jev readout categories (n=1200): valid=1200.
llama readout categories (n=1200): valid=1200; reported reasoning tokens: {0: 1200}.
haiku readout categories (n=1200): valid=1200.
Superseded, excluded from v3.1 comparisons: `calls.jsonl` (3601 attempts, $1.2599; unpinned Llama/mismatched Jev wording); `controlled_calls.jsonl` (4503 attempts, $1.5757; same issue). Earlier logs remain auditable and are never pooled with clean results.
Caveats: selection balances on held-out label (not on extreme means), changing population prevalence; shown reviews and held-out ratings are from different reviews but potentially same reviewers/variants; within-product reviews are correlated and may be manipulated; no exact normative posterior exists for real text. Label threshold 4.0 and finite held-out sample introduce noise. Stars used in matched-fraction analysis and the count rule are an offline audit, not shown to models. Matched fraction bins may have sparse or differently composed products; no causal or synthetic-exponent inference.

## Extension: independent empirical target and fixed shares

**Primary v3.2, exact 75% positive shown stars:** target 0.604 (k=4) → 0.640 (k=32); Jev 0.547 → 0.576; Together-pinned Llama 0.626 → 0.734; Haiku 0.575 → 0.661. Feasible product n changes with k; common-product sensitivity appears below.**

Empirical target (spec v3.1): **741 eligible calibration products, disjoint from all 240 evaluation products**, including every other product meeting ≥60 text reviews; k-specific calculations require ≥k shown-pool reviews. Their held-out label prevalence is 428/741=0.578; evaluation was deliberately balanced 120/120. For each calibration product and k, 16 independent uniform shown-pool draws without replacement *within each set*; held-out reviews never enter shown sets. Target bins pool the resulting draws; intervals resample products (180 draws), not reviews. Smooth reference for exactly 25%/75% uses human-label-only weighted logistic regression of held-out label on fraction, fraction², log₂(k) and their interactions; each product has total fitting weight one (over its feasible k). Bootstrap the entire fit over products (180 draws). It is observational, regularized and potentially extrapolates at rare k/share cells—not an exact normative posterior.

| k | shown positive fraction bin | target P(good) [95% CI] | sampled sets | contributing products |
|---:|---|---|---:|---:|
| 2 | [0.00,0.25) | 0.191 [0.156, 0.228] | 1195 | 408 |
| 2 | [0.50,0.75) | 0.455 [0.419, 0.490] | 4065 | 718 |
| 2 | [0.75,1.00] | 0.723 [0.693, 0.756] | 6596 | 721 |
| 4 | [0.00,0.25) | 0.056 [0.028, 0.100] | 321 | 114 |
| 4 | [0.25,0.50) | 0.154 [0.123, 0.185] | 1043 | 358 |
| 4 | [0.50,0.75) | 0.365 [0.326, 0.400] | 2344 | 620 |
| 4 | [0.75,1.00] | 0.714 [0.681, 0.744] | 8148 | 722 |
| 8 | [0.00,0.25) | 0.011 [0.000, 0.027] | 284 | 89 |
| 8 | [0.25,0.50) | 0.090 [0.064, 0.119] | 1100 | 281 |
| 8 | [0.50,0.75) | 0.350 [0.313, 0.386] | 3070 | 574 |
| 8 | [0.75,1.00] | 0.766 [0.735, 0.796] | 7402 | 686 |
| 16 | [0.00,0.25) | 0.000 [0.000, 0.000] | 224 | 41 |
| 16 | [0.25,0.50) | 0.034 [0.018, 0.054] | 1099 | 202 |
| 16 | [0.50,0.75) | 0.312 [0.273, 0.353] | 3592 | 499 |
| 16 | [0.75,1.00] | 0.820 [0.791, 0.845] | 6941 | 632 |
| 32 | [0.00,0.25) | 0.000 [0.000, 0.000] | 206 | 20 |
| 32 | [0.25,0.50) | 0.010 [0.001, 0.024] | 837 | 96 |
| 32 | [0.50,0.75) | 0.292 [0.236, 0.337] | 3565 | 366 |
| 32 | [0.75,1.00] | 0.871 [0.841, 0.896] | 5760 | 472 |

### Controlled shares, texts without stars

One independently randomized review set per feasible product and (k, share), sampled from its original shown pool. Feasibility requires exactly j positive (human rating 4–5) and k−j other reviews; skipped products are **not** replaced. The same text set goes to all three models; Jev per-review Nouls are in the same call as holistic. The disjoint-calibration smooth target estimates a random-shown-set conditional probability, whereas controlled sets include each feasible product once: target/model gaps are descriptive and selection/composition can differ. Jev Noul and LLM stated numbers are different readout interfaces, not interchangeable calibrated probabilities. CIs for model means resample products.

| k | positive share | feasible/240 (skipped) | exact-j calibration draws | smooth target [95% CI] | model | valid | mean confidence [95% CI] | model − target |
|---:|---:|---:|---:|---|---:|---|---:|---:|
| 4 | 25% | 228/240 (12) | 1043 | 0.138 [0.113, 0.165] | jev | 228 | 0.264 [0.253, 0.275] | +0.126 |
| 4 | 25% | 228/240 (12) | 1043 | 0.138 [0.113, 0.165] | llama | 228 | 0.083 [0.064, 0.102] | -0.054 |
| 4 | 25% | 228/240 (12) | 1043 | 0.138 [0.113, 0.165] | haiku | 228 | 0.215 [0.206, 0.225] | +0.077 |
| 4 | 75% | 237/240 (3) | 3952 | 0.604 [0.567, 0.641] | jev | 237 | 0.547 [0.534, 0.561] | -0.057 |
| 4 | 75% | 237/240 (3) | 3952 | 0.604 [0.567, 0.641] | llama | 237 | 0.626 [0.599, 0.651] | +0.022 |
| 4 | 75% | 237/240 (3) | 3952 | 0.604 [0.567, 0.641] | haiku | 237 | 0.575 [0.545, 0.597] | -0.028 |
| 8 | 25% | 207/240 (33) | 372 | 0.048 [0.034, 0.066] | jev | 207 | 0.218 [0.208, 0.229] | +0.171 |
| 8 | 25% | 207/240 (33) | 372 | 0.048 [0.034, 0.066] | llama | 207 | 0.106 [0.086, 0.127] | +0.059 |
| 8 | 25% | 207/240 (33) | 372 | 0.048 [0.034, 0.066] | haiku | 207 | 0.191 [0.181, 0.203] | +0.144 |
| 8 | 75% | 233/240 (7) | 2471 | 0.616 [0.576, 0.652] | jev | 233 | 0.576 [0.565, 0.589] | -0.040 |
| 8 | 75% | 233/240 (7) | 2471 | 0.616 [0.576, 0.652] | llama | 233 | 0.698 [0.675, 0.721] | +0.081 |
| 8 | 75% | 233/240 (7) | 2471 | 0.616 [0.576, 0.652] | haiku | 233 | 0.615 [0.592, 0.634] | -0.001 |
| 16 | 25% | 147/240 (93) | 135 | 0.015 [0.010, 0.025] | jev | 147 | 0.202 [0.192, 0.212] | +0.186 |
| 16 | 25% | 147/240 (93) | 135 | 0.015 [0.010, 0.025] | llama | 147 | 0.065 [0.049, 0.084] | +0.050 |
| 16 | 25% | 147/240 (93) | 135 | 0.015 [0.010, 0.025] | haiku | 147 | 0.169 [0.159, 0.181] | +0.154 |
| 16 | 75% | 217/240 (23) | 1460 | 0.628 [0.586, 0.666] | jev | 217 | 0.575 [0.562, 0.588] | -0.053 |
| 16 | 75% | 217/240 (23) | 1460 | 0.628 [0.586, 0.666] | llama | 217 | 0.732 [0.710, 0.752] | +0.104 |
| 16 | 75% | 217/240 (23) | 1460 | 0.628 [0.586, 0.666] | haiku | 217 | 0.638 [0.618, 0.658] | +0.010 |
| 32 | 25% | 72/240 (168) | 41 | 0.005 [0.003, 0.009] | jev | 72 | 0.193 [0.180, 0.204] | +0.188 |
| 32 | 25% | 72/240 (168) | 41 | 0.005 [0.003, 0.009] | llama | 72 | 0.057 [0.036, 0.073] | +0.052 |
| 32 | 25% | 72/240 (168) | 41 | 0.005 [0.003, 0.009] | haiku | 72 | 0.162 [0.145, 0.179] | +0.157 |
| 32 | 75% | 158/240 (82) | 683 | 0.640 [0.593, 0.680] | jev | 158 | 0.576 [0.563, 0.590] | -0.064 |
| 32 | 75% | 158/240 (82) | 683 | 0.640 [0.593, 0.680] | llama | 158 | 0.734 [0.712, 0.757] | +0.094 |
| 32 | 75% | 158/240 (82) | 683 | 0.640 [0.593, 0.680] | haiku | 158 | 0.661 [0.639, 0.683] | +0.022 |

**Primary reported E4 contrast (v3.2 analysis):** confidence against k for the **exactly 75%** controlled sets, compared with the human-label empirical target in the table and figure above. Other confidence contrasts and the 25% arm are secondary. As k increases, different products remain feasible; the table is not a paired fixed-product comparison across all k.

**Common-product robustness (75% share):** restrict all k to products feasible at k=32; these are the same products at k=4,8,16,32. Their review sets are independent per k (not nested).

| k | same products | smooth target | Jev mean [95% product CI] | Llama mean [95% product CI] | Haiku mean [95% product CI] |
|---:|---:|---:|---|---|---|
| 4 | 158 | 0.604 | 0.545 [0.529, 0.563] (n=158) | 0.622 [0.586, 0.654] (n=158) | 0.568 [0.535, 0.598] (n=158) |
| 8 | 158 | 0.616 | 0.569 [0.555, 0.584] (n=158) | 0.695 [0.667, 0.717] (n=158) | 0.593 [0.567, 0.626] (n=158) |
| 16 | 158 | 0.628 | 0.573 [0.561, 0.587] (n=158) | 0.725 [0.699, 0.749] (n=158) | 0.629 [0.599, 0.653] (n=158) |
| 32 | 158 | 0.640 | 0.576 [0.563, 0.590] (n=158) | 0.734 [0.709, 0.757] (n=158) | 0.661 [0.641, 0.684] (n=158) |

### Jev per-review count remedy on disjoint evaluation products

The same human-only calibration logistic rule above is trained **only** on the 741 non-evaluation products; at evaluation its input is k and the fraction of Jev per-review Nouls ≥0.5, never human stars. No Jev output enters training, tuning or model selection. Both outcomes below are the held-out human label. Log loss clips probability to [0.005,0.995]; AUROC is threshold-free. Product-cluster bootstrap 95% CIs, 250 draws. Controlled table pools both shares at each k, clustering duplicate products.

| set | k | valid products | condition rows | method | Brier [95% CI] | log loss [95% CI] | AUROC [95% CI] | target>0.9 products/rows | gate P(pred>0.9 | target>0.9) [95% CI] |
|---|---:|---:|---:|---|---|---|---|---:|---|
| natural | 2 | 240 | 240 | holistic | 0.216 [0.200, 0.231] | 0.620 [0.585, 0.660] | 0.718 [0.661, 0.783] | 0 | NA (zero target-positive rows) |
| natural | 2 | 240 | 240 | per-review rule | 0.227 [0.207, 0.247] | 0.646 [0.602, 0.679] | 0.667 [0.596, 0.725] | 0 | NA (zero target-positive rows) |
| natural | 4 | 240 | 240 | holistic | 0.187 [0.171, 0.211] | 0.556 [0.512, 0.603] | 0.798 [0.748, 0.865] | 0 | NA (zero target-positive rows) |
| natural | 4 | 240 | 240 | per-review rule | 0.190 [0.164, 0.218] | 0.567 [0.505, 0.628] | 0.770 [0.698, 0.838] | 0 | NA (zero target-positive rows) |
| natural | 8 | 240 | 240 | holistic | 0.162 [0.144, 0.181] | 0.501 [0.464, 0.540] | 0.872 [0.828, 0.911] | 44 | 0.000 [0.000, 0.080] |
| natural | 8 | 240 | 240 | per-review rule | 0.161 [0.134, 0.188] | 0.489 [0.425, 0.561] | 0.843 [0.793, 0.890] | 44 | 0.727 [0.607, 0.846] |
| natural | 16 | 240 | 240 | holistic | 0.153 [0.138, 0.167] | 0.478 [0.445, 0.509] | 0.910 [0.867, 0.938] | 43 | 0.000 [0.000, 0.082] |
| natural | 16 | 240 | 240 | per-review rule | 0.130 [0.105, 0.155] | 0.409 [0.344, 0.492] | 0.893 [0.853, 0.932] | 43 | 0.651 [0.480, 0.794] |
| natural | 32 | 240 | 240 | holistic | 0.144 [0.128, 0.161] | 0.460 [0.432, 0.494] | 0.930 [0.899, 0.956] | 61 | 0.000 [0.000, 0.059] |
| natural | 32 | 240 | 240 | per-review rule | 0.108 [0.083, 0.132] | 0.353 [0.290, 0.420] | 0.926 [0.890, 0.954] | 61 | 0.770 [0.650, 0.869] |
| controlled | 4 | 237 | 465 | holistic | 0.253 [0.239, 0.267] | 0.703 [0.672, 0.733] | 0.612 [0.579, 0.643] | 0 | NA (zero target-positive rows) |
| controlled | 4 | 237 | 465 | per-review rule | 0.287 [0.269, 0.307] | 0.809 [0.760, 0.859] | 0.566 [0.538, 0.593] | 0 | NA (zero target-positive rows) |
| controlled | 8 | 235 | 440 | holistic | 0.252 [0.237, 0.269] | 0.703 [0.666, 0.739] | 0.629 [0.593, 0.662] | 0 | NA (zero target-positive rows) |
| controlled | 8 | 235 | 440 | per-review rule | 0.300 [0.277, 0.328] | 0.932 [0.844, 1.011] | 0.588 [0.561, 0.617] | 0 | NA (zero target-positive rows) |
| controlled | 16 | 223 | 364 | holistic | 0.216 [0.202, 0.231] | 0.618 [0.584, 0.650] | 0.709 [0.669, 0.744] | 0 | NA (zero target-positive rows) |
| controlled | 16 | 223 | 364 | per-review rule | 0.257 [0.230, 0.282] | 0.853 [0.749, 0.979] | 0.649 [0.608, 0.695] | 0 | NA (zero target-positive rows) |
| controlled | 32 | 166 | 230 | holistic | 0.216 [0.200, 0.232] | 0.617 [0.582, 0.655] | 0.722 [0.666, 0.775] | 0 | NA (zero target-positive rows) |
| controlled | 32 | 166 | 230 | per-review rule | 0.244 [0.215, 0.273] | 0.823 [0.693, 0.960] | 0.661 [0.614, 0.722] | 0 | NA (zero target-positive rows) |

**Paired Brier improvement at k=32 (remedy minus holistic; negative is better):**
natural: -0.036 [-0.050, -0.021] (n=240 products, 240 rows; 95% product-cluster bootstrap).
controlled: +0.027 [0.009, 0.047] (n=166 products, 230 rows; 95% product-cluster bootstrap).
The Jev per-review rule improves Brier on natural nested sets at k=32, but **worsens it on fixed-share controlled sets**; no general remedy success is claimed.

**Gate caveat:** A zero target>0.9 denominator makes recall at that gate undefined, not 0%. The 25%/75% controlled shares may never cross the 0.9 empirical-target threshold; natural shown sets are also reported to make this explicit. Target and remedy are human-label-trained estimates, not independent normative truth. The evaluation set was balanced by held-out label; its calibration/Brier/log-loss do not estimate the original population prevalence.


### Invalid/excluded readouts and sensitivity (v3.2)

For each exact-share cell, missing/invalid model probabilities are imputed first at 0.5 and then at the smooth empirical target. The latter is a human-data reference, **not a normative answer** (none exists for real text). The same feasible products are used in both scenarios; no silent dropping.

| model | arm | k | valid / feasible | categories for recorded calls | complete-case mean | impute 0.5 | impute empirical target |
|---|---:|---:|---:|---|---:|---:|---:|
| jev | 25% | 4 | 228/228 | {'valid': 228} | 0.264 | 0.264 | 0.264 |
| jev | 25% | 8 | 207/207 | {'valid': 207} | 0.218 | 0.218 | 0.218 |
| jev | 25% | 16 | 147/147 | {'valid': 147} | 0.202 | 0.202 | 0.202 |
| jev | 25% | 32 | 72/72 | {'valid': 72} | 0.193 | 0.193 | 0.193 |
| jev | 75% | 4 | 237/237 | {'valid': 237} | 0.547 | 0.547 | 0.547 |
| jev | 75% | 8 | 233/233 | {'valid': 233} | 0.576 | 0.576 | 0.576 |
| jev | 75% | 16 | 217/217 | {'valid': 217} | 0.575 | 0.575 | 0.575 |
| jev | 75% | 32 | 158/158 | {'valid': 158} | 0.576 | 0.576 | 0.576 |
| llama | 25% | 4 | 228/228 | {'valid': 228} | 0.083 | 0.083 | 0.083 |
| llama | 25% | 8 | 207/207 | {'valid': 207} | 0.106 | 0.106 | 0.106 |
| llama | 25% | 16 | 147/147 | {'valid': 147} | 0.065 | 0.065 | 0.065 |
| llama | 25% | 32 | 72/72 | {'valid': 72} | 0.057 | 0.057 | 0.057 |
| llama | 75% | 4 | 237/237 | {'valid': 237} | 0.626 | 0.626 | 0.626 |
| llama | 75% | 8 | 233/233 | {'valid': 233} | 0.698 | 0.698 | 0.698 |
| llama | 75% | 16 | 217/217 | {'valid': 217} | 0.732 | 0.732 | 0.732 |
| llama | 75% | 32 | 158/158 | {'valid': 158} | 0.734 | 0.734 | 0.734 |
| haiku | 25% | 4 | 228/228 | {'valid': 228} | 0.215 | 0.215 | 0.215 |
| haiku | 25% | 8 | 207/207 | {'valid': 207} | 0.191 | 0.191 | 0.191 |
| haiku | 25% | 16 | 147/147 | {'valid': 147} | 0.169 | 0.169 | 0.169 |
| haiku | 25% | 32 | 72/72 | {'valid': 72} | 0.162 | 0.162 | 0.162 |
| haiku | 75% | 4 | 237/237 | {'valid': 237} | 0.575 | 0.575 | 0.575 |
| haiku | 75% | 8 | 233/233 | {'valid': 233} | 0.615 | 0.615 | 0.615 |
| haiku | 75% | 16 | 217/217 | {'valid': 217} | 0.638 | 0.638 | 0.638 |
| haiku | 75% | 32 | 158/158 | {'valid': 158} | 0.661 | 0.661 | 0.661 |
Clean v3.1 calls: 8097 attempts, 4497 valid distinct controlled calls for 1499 feasible product-condition sets; cost jev $0.1688, llama $2.0783, haiku $2.1868, total $4.4339 (natural + controlled). Failed/invalid records remain in `v31_calls.jsonl`; `controlled_calls.jsonl` is superseded.

### Identical-input test–retest (v3.2)

50 fixed products in the controlled 75%-positive k=8 arm were queried once more per model with byte-identical prompt/question text and request configuration. Agreement means identical numeric readout (not just class); differences may reflect provider/model nondeterminism. Retest CIs are product-bootstrap unless agreement equals 0 or 1, where Wilson bounds apply.

| model | valid paired / 50 | exactly identical [95% CI] | mean absolute shift [95% CI] | max shift |
|---|---:|---:|---:|---:|
| jev | 50/50 | 0.360 [0.240, 0.496] | 0.011 [0.008, 0.014] | 0.040 |
| llama | 50/50 | 0.920 [0.845, 0.980] | 0.016 [0.003, 0.031] | 0.280 |
| haiku | 50/50 | 1.000 [0.929, 1.000] | 0.000 [0.000, 0.000] | 0.000 |
