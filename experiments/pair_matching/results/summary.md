# E2 (pairs): summary of results (spec v3.1/v3.2; supplementary under v3.6)

The numbers below were produced by `analyze.py` from historical raw responses, which are not included. The released aggregate AUROC table is `results/tables/holistic_by_m.csv`.
**n:** the main items are 60 positives + 60 matched hard negatives + 15 easy negatives per roster size m and per vocabulary (names, codes).
**CIs:** 95% item bootstrap within class, 2000 resamples. Where a proportion is exactly 0 or 1, the Wilson interval is used instead.
**Thresholds:** Jev Noul P ≥ 0.5 counts as yes. For LLMs, a stated probability above 50 counts as yes (a stated 50 counts as no; a sensitivity check follows).
**Primary metric (v3.2 analysis):** AUROC of positives against matched hard negatives, by m.

## Task
- Each prompt gives a stated, exhaustive list of 12 disjoint forbidden pairs and a team roster of m people. The question is whether the roster contains any forbidden pair.
- Positives contain exactly one complete pair.
- Hard negatives are matched to positives on a and on the number of fillers, where a is the number of listed people present on the roster. They contain no complete pair.
- Easy negatives contain no listed people.
- Every item is checked by an executable oracle that re-parses the rendered prompt.

## Primary: matched-hard-negative AUROC, m = 4 → m = 16 (names; stated probability for LLMs)

| Model | AUROC m=4 | AUROC m=16 | difference | error direction (m = 8–16) |
|---|---|---|---|---|
| Jev (Noul) | 1.00 [0.99, 1.00] | 0.67 [0.57, 0.77] | −0.33 [−0.42, −0.23] | false alarms (FPR 0.93 at m=16, names) |
| Llama-3.3-70B (Together) | 0.85 [0.78, 0.91] | 0.60 [0.53, 0.67] | −0.25 [−0.33, −0.16] | false alarms |
| Mistral-Small-3.2-24B (DeepInfra) | 0.75 [0.67, 0.82] | 0.56 [0.48, 0.65] | −0.18 [−0.30, −0.06] | false alarms |
| Gemma-3-27B (DeepInfra) | 0.73 [0.66, 0.80] | 0.64 [0.57, 0.71] | −0.09 [−0.19, +0.01] (not significant; codes −0.15 [−0.25, −0.05]) | false alarms (FPR 0.97 at m=16) |
| DeepSeek-V3.2 (DeepInfra) | 0.63 [0.57, 0.69] | 0.47 [0.44, 0.50] | −0.16 [−0.22, −0.10] | **misses** (TPR ≤ 0.25 at every m) |
| Claude Haiku 4.5 | 0.75 [0.68, 0.82] | 0.67 [0.59, 0.75] | −0.08 [−0.19, +0.03] (not significant; codes −0.21 [−0.33, −0.10]) | **misses** (TPR 0.02–0.28, FPR 0.00–0.10, names) |
| Sonnet 5, reasoning reference (20 pos + 20 hard per m) | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0 | none (acc 1.00 [0.91, 1.00]); median 386 output tokens, 3.6 s |

- **Model coverage** (v3.2 item 4) is Jev plus 3 of 5 families in names (Llama, Mistral, DeepSeek), and Jev plus 4 of 5 in codes (Llama, Mistral, Gemma, Haiku). The AUROC fall is significant in both vocabularies only for Jev, Llama and Mistral.
- **Easy test.** Easy-negative accuracy is 1.00 [0.80, 1.00] (n = 15 per cell) for every model and m in names. The one exception is Gemma in codes, which has FPR 0.18 pooled over m (n = 60).
- **Direction of error depends on the model.** Jev, Llama, Mistral and Gemma raise false alarms. Haiku and DeepSeek mostly miss real pairs, answering "no". Haiku's errors are misses, not false alarms.
- **False alarms against unpaired parts** (hard negatives, within-size logistic slope with set-size fixed effects; §4b):
  - Every false-alarm model has a positive slope in both vocabularies. Jev: +0.62 [+0.45, +0.87] log-odds per unpaired listed part (names). Llama: +1.10. Mistral: +0.95.
  - Pooled over m, Jev's FPR climbs with the number of unpaired listed parts (names): 0.00 at 0 (easy) → 0.34 at 2–3 → 0.65 at 4–6 → 0.96 at 7–9 → 1.00 at 10–12.

## Per-piece checks and code remedies (names; 60 + 60 items per cell, all pieces valid)
- **Jev**, which asked all 36 per-piece questions in the same call as the holistic question:
  - Per-row accuracy is ≥ 0.999 and per-part accuracy is 1.000 at every m, in both vocabularies.
  - The per-row OR remedy scores 0.99–1.00 and the per-part join 1.00 [0.97, 1.00] at every m.
  - The join minus holistic accuracy difference, item-paired, is +0.09 [+0.04, +0.14] at m=4 and +0.47 [+0.43, +0.49] at m=16.
- **LLMs**: 12 isolated per-row calls at every m and 24 isolated per-part calls at m = 4 and 16.
  - The rule is that per-row accuracy must be at least 0.95 and OR-remedy accuracy at least 0.90 at every m.
  - "Each piece right" **holds** for Llama, DeepSeek and Haiku. Per-row accuracy is ≥ 0.997 for all three. OR-remedy accuracy is 0.97–1.00.
  - It **does not hold for Mistral**. When one member of a pair is on the roster, Mistral says both are present in 8–18% of rows. Its OR-remedy accuracy is 0.82, 0.90, 0.67 and 0.72 at m = 4, 8, 12, 16.
  - It **does not hold for Gemma at m = 16** (per-row accuracy 0.972, OR remedy 0.83 [0.78, 0.89]).
  - The per-part join is the more robust remedy. At m = 16 it scores 1.00 for Llama, Mistral and Haiku, 0.98 for DeepSeek and 0.95 for Gemma. At m = 4 it scores 0.96–1.00. The join minus holistic difference at m = 16 is +0.40 to +0.49 for every LLM.

## Controls and robustness
- **Jev holistic-only call** (all 1,080 items). It gives the same answers as the shared call:
  - AUROC 0.871 against 0.869.
  - Mean absolute difference in P of 0.030, maximum 0.46.
  - Decision agreement 0.966.
  - Latency 0.21 s against 0.22 s. Input tokens 473 against 1,092.
- **Haiku misses at m = 4 are genuine.**
  - All 2,160 main Haiku outputs are valid except one attempted-reasoning reply, and there are no refusals.
  - On 60 + 60 items at m = 4, TPR is 0.20 without the system prompt and 0.18 with a reworded question.
  - With step-by-step reasoning allowed, TPR is 1.00 and FPR 0.00 (median 294 output tokens).
  - No prompt was changed.
- **Wording.** Two paraphrased templates (T1, T2) were run on 40 + 40 items at m = 4 and 16:
  - Jev's AUROC falls from m = 4 to m = 16 in every template: T0 1.00 → 0.64, T1 0.99 → 0.82, T2 0.98 → 0.56.
  - The LLMs are highly sensitive to wording. On T1, and for several models on T2, AUROC is already about 0.50 at m = 4, so the size effect cannot be tested there. Their failure is broader than the size effect.
- **Jev Choice readout** (two options, 40 + 40 per m): AUROC is 1.00, 0.95, 0.86 and 0.63 at m = 4, 8, 12, 16, matching the Noul (1.00, 0.94, 0.80, 0.64).
- **Test-retest** (100 identical calls per model; Sonnet 80):
  - Decision agreement is 0.99 for Jev, 0.99–1.00 for Llama, Mistral, Gemma and Haiku, and 0.94 for DeepSeek.
  - Jev's P values are not bit-identical: exact agreement 0.32, mean absolute difference 0.027. Across 3,600 per-piece decisions, agreement is 0.9997.
- **Stated 50 counted as yes.** This changes Mistral's names m=8 TPR/FPR from 0.37/0.28 to 0.73/0.63 (43 of the 135 names m=8 answers were exactly 50), but accuracy stays 0.54–0.55 and AUROC is unaffected.
- **Invalid outputs.** Across all primary LLM holistic calls there is one invalid output (Haiku). AUROC is identical under exclusion, imputation at 0.5, imputation at the normative answer, and counting it as wrong.
- **Pinned against unpinned providers** (superseded unpinned runs). AUROC changes by at most 0.05 in any cell, with no systematic direction. Pinning mattered for per-piece validity, though: one unpinned Llama provider (DeepInfra) returned strings of backslashes on 130 per-piece calls. Those keys were re-run pinned to Together.

## Exclusions
Unpinned provider runs and early prompt-format smoke tests are excluded from these estimates. Historical raw responses are not included in this release.

## Spend (ledger, tag onepass-e2)
Anthropic $5.23, OpenRouter $5.60, TypeSafe Jev $0.12.
