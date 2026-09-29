"""Generate pair-matching items and check their labels against the rendered text.

Each item lists 12 disjoint forbidden pairs and a team of 4, 8, 12, or 16 people.
Positive teams contain one full pair; hard negatives contain no full pair. Within
each seed and team size, the positive and hard-negative sets have the same distribution
of listed-person and filler counts. Easy negatives contain only fillers.
The same items are rendered with first names and badge codes. The oracle parses the
rendered text, then checks both the labels and matching before writing items.jsonl.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
N_RULES = 12
SET_SIZES = (4, 8, 12, 16)
SEEDS = (0, 1, 2)
PER_SEED = {"pos": 20, "hard": 20, "easy": 5}  # per (seed, m): 60 + 60 + 15 per m overall
A_RANGE = {4: (2, 4), 8: (2, 8), 12: (2, 12), 16: (2, 12)}  # inclusive; a <= 12 always

# ---------------------------------------------------------------- identifier pools
_RAW_NAMES = """
Aaron Abigail Adele Adrian Ahmed Aiko Aisha Alan Alba Alexei Alfonso Alice Alina Alma Amara
Ambrose Amelia Amir Anders Andrea Angus Anika Anton Arjun Arlo Astrid Aurora Axel Ayla Beatrix
Benedict Bernard Bianca Bjorn Blythe Boris Brenda Bruno Caleb Camila Candice Carlos Carmen
Cecil Celeste Chandra Chiara Chidi Clara Clement Colette Conrad Cora Cyrus Dalia Damian Daphne
Darius Declan Delia Dennis Diego Dimitri Dolores Dorian  Edgar Edith Eduardo Efua Elena
Elias Eliza Elliot Elspeth Emeka Emil Enzo Esther Ethan Eunice Evelyn Ezra Fabian Farah Felix
Fergus Fiona Florian Freya Gabriel Gareth Gemma Georgia Gideon Gloria Gordon Greta Gustav Hamid
Hannah Harriet Hassan Hector Helga Henrik Hilda Hiroshi Hugo Ibrahim Ida Ignacio Ilse Imani
Ingrid Irene Isaac Isolde Ivan Jamal Janelle Jasper Javier Jiro Joanna Jonas Josefa Julian
Juniper Kamal Karin Kasimir Keiko Kendra Kenji Kiran Klaus Kofi Lachlan Lars Leander Leila
Lena Leon Liam Lidia Linus Lorenzo Lucia Ludmila Magnus Malik Marcus Margot Marisol Marta Mateo
Matilda Maxim Meera Mei Milan Mira Moira Monique Morgan Nadia Naomi Nasser Neha Nelson Nia
Nikolai Nils Nora Octavia Odette Olga Omar Oriana Oscar Otto Pablo Paloma Pascal Petra Philippa
Pierre Priya Quentin Quinn Rafael Rahul Ramona Ravi Rebecca Reuben Rhea Rikard Rita Roberto
Rosalind Rufus Rupert Sabine Salma Samir Sanjay Sasha Selma Seth Shira Silas Simone Sofia Soren
Stefan Stella Sunita Svetlana Tamsin Tariq Tessa Thea Theodore Tobias Tomas Ursula Valentina
Vera Victor Vikram Viola Walter Wanda Wendell Xavier Ximena Yara Yasmin Yusuf Zainab Zara Zoltan
Agnes Albert Alvaro Anselm Ariadne Bartholomew  Brigid Cassius Cosima Cornelius Dagny
Desmond Eamon Elodie Esme Fatima Filippo Gaspard Giselle Hortense Ingmar Jolene  Lavinia
Leopold Lucius  Maximilian Mireille Nerissa Oswald Perpetua Raimund Rosamund Sebastian
Seraphina Thaddeus Ulrich Wilhelmina Yevgeny Zephyrine Arvid Birgit Cyprian Dario Emrys Frida
Gunnar Halima Idris Jovan Kalina Lorcan Maelle Nikhil Orla Paavo Radek Saoirse Tove Umar Vesna
Wolfgang Yannick Zdenek Anouk Bastian Corentin Dunya Eskil Folasade Gerda Hallie Ilario Jarek
Kaveh Liesel Mikkel Nuala Ottilie Pilar Rashid Solveig  Uzma Vadim Wilma Yuki Zeynep
"""


def _build_name_pool():
    names = sorted(set(_RAW_NAMES.split()))
    assert all(re.fullmatch(r"[A-Z][a-z]+", n) for n in names)
    # drop any name that is a substring of another (e.g. "Ida" in "Idaho") so that no
    # identifier can be confused by substring overlap
    low = [n.lower() for n in names]
    keep = [n for n, l in zip(names, low) if not any(l != o and l in o for o in low)]
    return keep


NAME_POOL = _build_name_pool()
# neutral two-character badge codes: letter (no I, O) + digit 2..9
CODE_POOL = [c + d for c in "ABCDEFGHJKLMNPQRSTUVWXYZ" for d in "23456789"]


def _rng(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------- abstract items
def a_design(seed, m, count):
    lo, hi = A_RANGE[m]
    vals = list(range(lo, hi + 1))
    off = seed * count  # continue the cycle across seeds so all a values are covered per m
    return [vals[(off + j) % len(vals)] for j in range(count)]


def make_base_item(seed, m, cls, j, a):
    """Abstract item. Listed person ids 0..23 (rule r = {2r, 2r+1}); fillers 24.."""
    rng = _rng("e2", seed, m, cls, j)
    rules_perm = list(range(N_RULES))
    rng.shuffle(rules_perm)  # display order of rules
    flips = [rng.random() < 0.5 for _ in range(N_RULES)]  # left/right order within each rule
    rule_display = [((2 * r + 1, 2 * r) if flips[r] else (2 * r, 2 * r + 1)) for r in rules_perm]
    if cls == "pos":
        chosen = rng.sample(range(N_RULES), a - 1)  # first one is the complete pair
        pair_rule = chosen[0]
        listed = [2 * pair_rule, 2 * pair_rule + 1] + [2 * r + rng.randrange(2) for r in chosen[1:]]
    elif cls == "hard":
        chosen = rng.sample(range(N_RULES), a)
        pair_rule = None
        listed = [2 * r + rng.randrange(2) for r in chosen]
    elif cls == "easy":
        assert a == 0
        pair_rule, listed = None, []
    else:
        raise ValueError(cls)
    fillers = list(range(24, 24 + (m - len(listed))))
    roster = listed + fillers
    rng.shuffle(roster)
    part_order = list(range(24))
    rng.shuffle(part_order)  # order of the 24 per-part questions (cosmetic for isolated branches)
    return {"seed": seed, "m": m, "cls": cls, "j": j, "a": len(listed), "n_fillers": len(fillers),
            "pair_rule": pair_rule, "rule_display": rule_display, "roster": roster,
            "part_order": part_order}


def base_items():
    items = []
    for seed in SEEDS:
        for m in SET_SIZES:
            avals = a_design(seed, m, PER_SEED["pos"])
            assert PER_SEED["pos"] == PER_SEED["hard"]
            for cls in ("pos", "hard"):
                for j, a in enumerate(avals):
                    items.append(make_base_item(seed, m, cls, j, a))
            for j in range(PER_SEED["easy"]):
                items.append(make_base_item(seed, m, "easy", j, 0))
    for it in items:
        it["base_id"] = f"s{it['seed']}-m{it['m']:02d}-{it['cls']}-{it['j']:02d}"
    return items


# ---------------------------------------------------------------- rendering
VOCAB_TEXT = {
    "names": {"intro": "Team assignment rules. Each pair of people below must not be on the same team.",
              "who": "people"},
    "codes": {"intro": "Team assignment rules. Staff are identified by badge codes. Each pair of staff "
                       "below must not be on the same team.",
              "who": "staff"},
}


def identifier_map(base, vocab):
    n_ids = 24 + base["n_fillers"]
    rng = _rng("ids", vocab, base["base_id"])
    pool = NAME_POOL if vocab == "names" else CODE_POOL
    ids = rng.sample(pool, n_ids)
    return {i: ids[i] for i in range(n_ids)}


def render_state(base, vocab, idmap):
    t = VOCAB_TEXT[vocab]
    lines = [t["intro"],
             f"This list of {N_RULES} forbidden pairs is complete: no other pair of {t['who']} is forbidden.",
             ""]
    for k, (x, y) in enumerate(base["rule_display"], 1):
        lines.append(f"{k}. {idmap[x]} and {idmap[y]}")
    lines += ["", f"Proposed team roster ({base['m']} {t['who']}):"]
    lines += [f"- {idmap[p]}" for p in base["roster"]]
    return "\n".join(lines)


HOLISTIC_Q = ("Does the proposed team roster contain at least one forbidden pair, that is, both "
              "members of one of the listed pairs?")


def row_question(x, y):
    return f"Are both {x} and {y} on the proposed team roster?"


def part_question(x):
    return f"Is {x} on the proposed team roster?"


def render_item(base, vocab):
    idmap = identifier_map(base, vocab)
    state = render_state(base, vocab, idmap)
    rows = [{"key": f"r{k:02d}", "x": idmap[x], "y": idmap[y]}
            for k, (x, y) in enumerate(base["rule_display"])]
    parts = [{"key": f"p{k:02d}", "x": idmap[p]} for k, p in enumerate(base["part_order"])]
    return {"item_id": f"{vocab}-{base['base_id']}", "base_id": base["base_id"], "vocab": vocab,
            "seed": base["seed"], "m": base["m"], "cls": base["cls"], "a": base["a"],
            "n_fillers": base["n_fillers"], "state": state, "rows": rows, "parts": parts,
            "idmap": {str(k): v for k, v in idmap.items()}}


# ---------------------------------------------------------------- executable oracle
_RULE_RE = re.compile(r"^(\d+)\. (\S+) and (\S+)$")
_ROSTER_HDR = re.compile(r"^Proposed team roster \((\d+) (people|staff)\):$")


def parse_state(state):
    """Parse the rendered prompt text back into (rules, roster). Raises on any format error."""
    lines = state.split("\n")
    rules, roster, declared_m, mode = [], [], None, "head"
    for ln in lines:
        if mode == "head":
            mr = _RULE_RE.match(ln)
            if mr:
                if int(mr.group(1)) != len(rules) + 1:
                    raise ValueError("rule numbering")
                rules.append((mr.group(2), mr.group(3)))
                continue
            mh = _ROSTER_HDR.match(ln)
            if mh:
                declared_m, mode = int(mh.group(1)), "roster"
            continue
        if not ln.startswith("- ") or len(ln.split()) != 2:
            raise ValueError(f"bad roster line {ln!r}")
        roster.append(ln[2:])
    if declared_m is None or declared_m != len(roster):
        raise ValueError("roster header count mismatch")
    return rules, roster


def oracle(state):
    """Recompute every label from the rendered text."""
    rules, roster = parse_state(state)
    listed = [p for r in rules for p in r]
    roster_set = set(roster)
    complete = [i for i, (x, y) in enumerate(rules) if x in roster_set and y in roster_set]
    endpoints = [p for p in roster if p in set(listed)]
    rules_touched = {i for i, (x, y) in enumerate(rules) if x in roster_set or y in roster_set}
    return {
        "n_rules": len(rules),
        "rules_disjoint": len(set(listed)) == len(listed),
        "roster_unique": len(roster_set) == len(roster),
        "m": len(roster),
        "n_complete_pairs": len(complete),
        "complete_rows": complete,
        "a": len(endpoints),
        "n_fillers": len(roster) - len(endpoints),
        "n_rules_touched": len(rules_touched),
        "n_unpaired": len(endpoints) - 2 * len(complete),
        "label": int(len(complete) > 0),
        "row_truth": [int(x in roster_set and y in roster_set) for x, y in rules],
        "part_truth": {p: int(p in roster_set) for p in listed},
    }


def validate_item(it):
    """Raise AssertionError if the rendered item violates the design."""
    o = oracle(it["state"])
    assert o["n_rules"] == N_RULES, "need exactly 12 rules"
    assert o["rules_disjoint"], "rules must be disjoint (24 distinct identifiers)"
    assert o["roster_unique"], "duplicate on roster"
    assert o["m"] == it["m"]
    assert o["a"] == it["a"] and o["n_fillers"] == it["n_fillers"]
    assert o["a"] <= N_RULES
    if it["cls"] == "pos":
        assert o["n_complete_pairs"] == 1 and o["n_unpaired"] == it["a"] - 2
        assert o["n_rules_touched"] == it["a"] - 1
    elif it["cls"] == "hard":
        assert o["n_complete_pairs"] == 0 and o["n_unpaired"] == it["a"]
        assert o["n_rules_touched"] == it["a"] and it["a"] >= 2
    elif it["cls"] == "easy":
        assert o["a"] == 0 and o["n_complete_pairs"] == 0
    if it["m"] == 16:
        assert o["n_fillers"] >= 4
    # identifier hygiene: no identifier is a substring of another in the same item
    ids = list(it["idmap"].values())
    assert len(set(ids)) == len(ids)
    low = [s.lower() for s in ids]
    assert not any(x != y and x in y for x in low for y in low), "substring identifiers"
    # rows and parts cover the rule list exactly
    rules, roster = parse_state(it["state"])
    assert [(r["x"], r["y"]) for r in it["rows"]] == rules
    assert sorted(p["x"] for p in it["parts"]) == sorted(p for r in rules for p in r)
    # fillers are never listed
    listed = {p for r in rules for p in r}
    assert sum(p not in listed for p in roster) == it["n_fillers"]
    return o


def validate_matching(items):
    """Per (vocab, seed, m): identical multiset of (a, n_fillers) for pos and hard."""
    groups = {}
    for it in items:
        groups.setdefault((it["vocab"], it["seed"], it["m"], it["cls"]), []).append((it["a"], it["n_fillers"]))
    for (v, s, m, c), vals in groups.items():
        if c == "pos":
            assert Counter(vals) == Counter(groups[(v, s, m, "hard")]), f"unmatched a at {v} s{s} m{m}"


def validate_pairing(items):
    """names and codes renderings of a base item must share structure exactly."""
    by = {}
    for it in items:
        by.setdefault(it["base_id"], {})[it["vocab"]] = it
    for bid, d in by.items():
        assert set(d) == {"names", "codes"}, bid
        n, c = d["names"], d["codes"]
        inv_n = {v: k for k, v in n["idmap"].items()}
        inv_c = {v: k for k, v in c["idmap"].items()}
        on, oc = oracle(n["state"]), oracle(c["state"])
        for key in ("label", "a", "n_fillers", "m", "complete_rows", "row_truth", "n_unpaired"):
            assert on[key] == oc[key], (bid, key)
        rn, qn = parse_state(n["state"])
        rc, qc = parse_state(c["state"])
        assert [(inv_n[x], inv_n[y]) for x, y in rn] == [(inv_c[x], inv_c[y]) for x, y in rc]
        assert [inv_n[x] for x in qn] == [inv_c[x] for x in qc]
        assert [inv_n[p["x"]] for p in n["parts"]] == [inv_c[p["x"]] for p in c["parts"]]


def build_all():
    items = []
    for b in base_items():
        for vocab in ("names", "codes"):
            items.append(render_item(b, vocab))
    for it in items:
        o = validate_item(it)
        it["label"] = o["label"]
        it["n_unpaired"] = o["n_unpaired"]
        it["row_truth"] = o["row_truth"]
        it["part_truth"] = [o["part_truth"][p["x"]] for p in it["parts"]]
    validate_matching(items)
    validate_pairing(items)
    return items


def main():
    items = build_all()
    out = os.path.join(HERE, "items.jsonl")
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    c = Counter((it["vocab"], it["m"], it["cls"]) for it in items)
    print(f"name pool {len(NAME_POOL)}, code pool {len(CODE_POOL)}; wrote {len(items)} validated items to {out}")
    for k in sorted(c):
        print(k, c[k])
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
    print("items.jsonl sha256[:16] =", sha)


# ---------------------------------------------------------------- v3.2: paraphrased templates (subset)
# Two paraphrases of the rules text, roster layout and holistic question. Same base items, same names,
# same rule order and roster order as the main "names" rendering (T0); only the wording changes.
PARA_Q = {
    "T1": "Is any listed pair fully present on this team, with both of its people assigned?",
    "T2": "Does this team include both people from at least one conflict on the list?",
}
_PARA_RULE = {"T1": re.compile(r"^([A-L])\) (\S+) / (\S+)$"),
              "T2": re.compile(r"^- (\S+) cannot work with (\S+)$")}
_PARA_HDR = {"T1": re.compile(r"^People currently assigned to the team \((\d+) in total\): (.+)$"),
             "T2": re.compile(r"^Team members \((\d+)\):$")}


def para_subset(base):
    """40 positives + 40 matched hard negatives per m in {4, 16}: j % 3 != 2, minus seed 2 j >= 18."""
    j = base["j"]
    return (base["m"] in (4, 16) and base["cls"] in ("pos", "hard") and j % 3 != 2
            and not (base["seed"] == 2 and j >= 18))


def render_para(base, tpl):
    idmap = identifier_map(base, "names")  # identical names to the main names rendering
    rules = [(idmap[x], idmap[y]) for x, y in base["rule_display"]]
    roster = [idmap[p] for p in base["roster"]]
    if tpl == "T1":
        lines = ["Staffing constraints. The two people in each pair below may not be assigned to the same team.",
                 f"These {N_RULES} pairs are the only restrictions; every other combination of people is allowed.", ""]
        lines += [f"{chr(65 + k)}) {x} / {y}" for k, (x, y) in enumerate(rules)]
        lines += ["", f"People currently assigned to the team ({len(roster)} in total): " + ", ".join(roster)]
    elif tpl == "T2":
        lines = [f"Conflict list (complete: there are no conflicts beyond these {N_RULES}):"]
        lines += [f"- {x} cannot work with {y}" for x, y in rules]
        lines += ["", f"Team members ({len(roster)}):"]
        lines += [f"{k}. {p}" for k, p in enumerate(roster, 1)]
    else:
        raise ValueError(tpl)
    return {"item_id": f"names{tpl}-{base['base_id']}", "base_id": base["base_id"], "vocab": "names",
            "template": tpl, "seed": base["seed"], "m": base["m"], "cls": base["cls"], "a": base["a"],
            "n_fillers": base["n_fillers"], "state": "\n".join(lines), "question": PARA_Q[tpl],
            "idmap": {str(k): v for k, v in idmap.items()}}


def parse_para(state, tpl):
    rules, roster, declared = [], [], None
    lines = state.split("\n")
    for ln in lines:
        mr = _PARA_RULE[tpl].match(ln)
        if mr:
            g = mr.groups()
            if tpl == "T1" and ord(g[0]) - 65 != len(rules):
                raise ValueError("rule lettering")
            rules.append(g[-2:])
            continue
        mh = _PARA_HDR[tpl].match(ln)
        if mh:
            declared = int(mh.group(1))
            if tpl == "T1":
                roster = mh.group(2).split(", ")
            continue
        if tpl == "T2" and declared is not None:
            mm = re.match(r"^(\d+)\. (\S+)$", ln)
            if not mm or int(mm.group(1)) != len(roster) + 1:
                raise ValueError(f"bad roster line {ln!r}")
            roster.append(mm.group(2))
    if declared is None or declared != len(roster):
        raise ValueError("roster count mismatch")
    return rules, roster


def validate_para(it, main_item):
    rules, roster = parse_para(it["state"], it["template"])
    r0, q0 = parse_state(main_item["state"])
    assert rules == r0 and roster == q0, "paraphrase must keep names and orders"
    assert len(rules) == N_RULES and len({p for r in rules for p in r}) == 2 * N_RULES
    s = set(roster)
    complete = sum(x in s and y in s for x, y in rules)
    assert complete == (1 if it["cls"] == "pos" else 0)
    listed = {p for r in rules for p in r}
    assert sum(p in listed for p in roster) == it["a"]


def build_para(main_items=None):
    main_items = main_items or {it["item_id"]: it for it in build_all()}
    out = []
    for b in base_items():
        if not para_subset(b):
            continue
        for tpl in ("T1", "T2"):
            it = render_para(b, tpl)
            validate_para(it, main_items[f"names-{b['base_id']}"])
            it["label"] = 1 if b["cls"] == "pos" else 0
            out.append(it)
    c = Counter((it["template"], it["m"], it["cls"]) for it in out)
    assert all(v == 40 for v in c.values()) and len(c) == 8, c
    for tpl in ("T1", "T2"):
        for m in (4, 16):
            pa = Counter(it["a"] for it in out if (it["template"], it["m"], it["cls"]) == (tpl, m, "pos"))
            ha = Counter(it["a"] for it in out if (it["template"], it["m"], it["cls"]) == (tpl, m, "hard"))
            assert pa == ha
    return out


def main_para():
    out = build_para()
    path = os.path.join(HERE, "items_para.jsonl")
    with open(path, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} validated paraphrase items to {path}; sha256[:16] =",
          hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and validate pair-matching items.")
    parser.add_argument("--para", action="store_true", help="Generate the paraphrased subset.")
    args = parser.parse_args()
    try:
        main_para() if args.para else main()
    except AssertionError as exc:
        print("VALIDATION FAILED:", exc)
        sys.exit(1)
