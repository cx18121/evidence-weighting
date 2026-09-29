"""Write representative model inputs to results/examples.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

HERE = Path(__file__).resolve().parent
cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases.jsonl")}


def elide(text):
    lines = text.split("\n")
    out, pad = [], 0
    for ln in lines:
        if ln.startswith("Log ") and pad >= 2:
            pad += 1
            continue
        if ln.startswith("Log "):
            pad += 1
        out.append(ln)
    if pad > 2:
        i = max(j for j, ln in enumerate(out) if ln.startswith("Log "))
        out.insert(i + 1, f"[... {pad - 2} more lines 'Log N: <same note>' ...]")
    return "\n".join(out)


def block(title, text):
    return f"**{title}**\n\n```text\n{elide(text)}\n```\n"


S = ["# Example prompts\n\nLong blocks of neutral padding are shortened below.\n",
     f"LLM system prompt (all non-reasoning LLMs, numeric readouts): `{run.SYSTEM}`  \n"
     f"Yes/No readouts (yes/no decision, per-clue): `{run.SYSTEM_YESNO}`  \n"
     f"max_tokens {run.MAX_TOKENS}, temperature 0. Sonnet 5 reasoning: no system prompt, max_tokens {run.SONNET_MAX_TOKENS}.\n"]
E1L = ["e1-r52-food-n4-k3-listed-s0", "e1-r55-reviews-n8-k2-listed-s1", "e1-r70-scanner-n4-k1-listed-s5"]
E1C = ["e1-r52-food-n64-k48-count-s0", "e1-r55-reviews-n16-k10-count-s4", "e1-r70-scanner-n32-k8-count-s2"]
S.append("## Listed independent reports\n")
S += [block(i, run.llm_prompt(cases[i], "holistic")) for i in E1L]
S.append("## Report counts stated in the prompt\n")
S += [block(i, run.llm_prompt(cases[i], "holistic")) for i in E1C]
S.append("## Jev state and questions\n")
for i in (E1L[0], E1C[0], E1L[2]):
    st, qs = run.jev_request(cases[i], "holistic_clues" if "listed" in i else "holistic")
    S.append(block(i + " (state)", st) + "```json\n" + json.dumps(qs, indent=1)[:1500] + "\n```\n")
S.append("## Prompt-length control\n")
S += [block(i, run.llm_prompt(cases[i], "pad")) for i in ("e1-r55-food-n4-k1-listed-s0", "e1-r55-reviews-n16-k12-listed-s1",
                                                           "e1-r55-scanner-n32-k20-listed-s2")]
S.append("## Reversed outcome\n")
S += [block(i, run.llm_prompt(cases[i], "flip")) for i in ("e1-r55-food-n8-k6-listed-s0", "e1-r55-reviews-n4-k1-listed-s1",
                                                            "e1-r55-scanner-n8-k3-listed-s2")]
S.append("## Count questions\n")
S += [block(i, run.llm_prompt(cases[i], "count")) for i in ("e1-r52-food-n8-k5-listed-s3", "e1-r52-reviews-n4-k2-listed-s1",
                                                             "e1-r52-scanner-n8-k2-listed-s4")]
st, qs = run.jev_request(cases["e1-r52-food-n8-k5-listed-s3"], "jev_count")
S.append("Jev questions for e1-r52-food-n8-k5-listed-s3 (same state as above):\n\n```json\n" + json.dumps(qs, indent=1) + "\n```\n")
S.append("## Individual report questions\n")
S += [block(f"{i} report {j}", run.llm_prompt(cases[i], "clue", j)) for i, j in
      (("e1-r52-food-n4-k1-listed-s0", 0), ("e1-r52-reviews-n4-k3-listed-s1", 2), ("e1-r55-scanner-n4-k3-listed-s0", 3))]
S.append("## Yes/No and option-order controls\n")
S += [block(i, run.llm_prompt(cases[i], "yesno")) for i in ("e1-r52-food-n4-k3-listed-s0", "e1-r52-reviews-n8-k5-listed-s1",
                                                             "e1-r52-scanner-n4-k2-listed-s0")]
st, qs = run.jev_request(cases["e1-r52-food-n4-k3-listed-s0"], "choice")
st2, qs2 = run.jev_request(cases["e1-r52-food-n4-k3-listed-s0"], "choice_rev")
S.append("Jev Choice question (and reversed-order variant):\n\n```json\n" + json.dumps(qs, indent=1) + "\n" + json.dumps(qs2, indent=1) + "\n```\n")
S.append("## Reasoning reference: Sonnet 5\n")
S += [block(i, run.llm_prompt(cases[i], "sonnet")) for i in ("e1-r52-food-n4-k3-listed-s0", "e1-r55-reviews-n8-k2-listed-s0",
                                                              "e1-r70-scanner-n4-k1-listed-s0")]
S.append("## One decisive report among weak reports\n")
S += [block(i, run.llm_prompt(cases[i], "holistic")) for i in ("e3-food-n5-d1-early-decisive-s0", "e3-reviews-n5-d0-late-decisive-s1",
                                                                "e3-scanner-n9-d1-late-decisive-s2")]
S += [block(i, run.llm_prompt(cases[i], "holistic")) for i in ("e3-food-n5-d1-none-weak_only-s0", "e3-reviews-n9-d0-none-weak_only-s3",
                                                                "e3-scanner-n5-d1-none-weak_only-s5")]
S += [block(i, run.llm_prompt(cases[i], "holistic")) for i in ("e3-food-n5-d1-early-padding-s0", "e3-reviews-n17-d0-late-padding-s1",
                                                                "e3-scanner-n33-d1-early-padding-s2")]
(HERE / "results" / "examples.md").write_text("\n".join(S))
print("wrote results/examples.md")
