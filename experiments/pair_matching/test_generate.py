"""Unit tests for the E2 generator and oracle. Run: python -m pytest -q test_generate.py"""
from collections import Counter

import pytest

import generate as g


@pytest.fixture(scope="module")
def items():
    return g.build_all()


def test_counts(items):
    c = Counter((it["vocab"], it["m"], it["cls"]) for it in items)
    for v in ("names", "codes"):
        for m in g.SET_SIZES:
            assert c[(v, m, "pos")] == 60 and c[(v, m, "hard")] == 60 and c[(v, m, "easy")] == 15


def test_labels_and_oracle_agree(items):
    for it in items:
        o = g.oracle(it["state"])
        assert o["label"] == (1 if it["cls"] == "pos" else 0)
        assert sum(o["row_truth"]) == o["n_complete_pairs"]
        assert sum(it["part_truth"]) == it["a"]


def test_matching_per_m(items):
    for v in ("names", "codes"):
        for m in g.SET_SIZES:
            pa = Counter(it["a"] for it in items if (it["vocab"], it["m"], it["cls"]) == (v, m, "pos"))
            ha = Counter(it["a"] for it in items if (it["vocab"], it["m"], it["cls"]) == (v, m, "hard"))
            assert pa == ha
            lo, hi = g.A_RANGE[m]
            assert set(pa) == set(range(lo, hi + 1))  # a varies over the whole allowed range


def test_deterministic():
    a = g.build_all()
    b = g.build_all()
    assert [x["state"] for x in a] == [x["state"] for x in b]


def test_orders_are_randomized(items):
    # the complete pair's rule is not always in the same display row, and roster orders differ
    rows = Counter(g.oracle(it["state"])["complete_rows"][0] for it in items if it["cls"] == "pos")
    assert len(rows) == 12
    # first roster entry is a listed endpoint in a reasonable share of pos items
    firsts = [g.parse_state(it["state"])[1][0] for it in items if it["cls"] == "pos" and it["m"] == 16]
    assert len(set(firsts)) > 20


def test_oracle_detects_bad_items():
    base = g.make_base_item(0, 8, "hard", 0, 4)
    base["base_id"] = "t"
    it = g.render_item(base, "names")
    g.validate_item(it)
    # corrupt: add the partner of a roster endpoint -> hard negative now has a pair
    rules, roster = g.parse_state(it["state"])
    partner = next(y if x in roster else x for x, y in rules if (x in roster) != (y in roster))
    bad = dict(it, state=it["state"] + f"\n- {partner}")
    with pytest.raises(ValueError):  # header count no longer matches
        g.validate_item(bad)
    bad_state = bad["state"].replace("(8 people)", "(9 people)")
    o = g.oracle(bad_state)
    assert o["n_complete_pairs"] == 1 and o["label"] == 1
    with pytest.raises(AssertionError):
        g.validate_item(dict(it, state=bad_state, m=9))


def test_oracle_detects_non_disjoint_rules():
    state = ("x\n\n1. A2 and B2\n2. A2 and C2\n" + "\n".join(f"{k}. D{k} and E{k}" for k in range(3, 13))
             + "\n\nProposed team roster (2 staff):\n- A2\n- C2")
    o = g.oracle(state)
    assert not o["rules_disjoint"] and o["n_complete_pairs"] == 1


def test_pairing_structure(items):
    g.validate_pairing(items)
    by = {}
    for it in items:
        by.setdefault(it["base_id"], {})[it["vocab"]] = it
    n = by["s0-m16-pos-00"]
    assert n["names"]["state"] != n["codes"]["state"]
    assert n["names"]["label"] == n["codes"]["label"] == 1


def test_identifier_pools():
    assert len(g.NAME_POOL) >= 300
    assert len(g.CODE_POOL) >= 150
    low = [x.lower() for x in g.NAME_POOL]
    assert not any(a != b and a in b for a in low for b in low)


def test_m16_fillers(items):
    assert all(it["n_fillers"] >= 4 for it in items if it["m"] == 16)


def test_paraphrase_items(items):
    main = {it["item_id"]: it for it in items}
    para = g.build_para(main)
    assert len(para) == 320
    for it in para:
        g.validate_para(it, main[f"names-{it['base_id']}"])
        assert it["question"] == g.PARA_Q[it["template"]]
    # templates really differ in wording from the main rendering
    x = [p for p in para if p["base_id"] == "s0-m16-pos-00"]
    assert len({p["state"] for p in x} | {main["names-s0-m16-pos-00"]["state"]}) == 3


def test_paraphrase_oracle_catches_order_change(items):
    main = {it["item_id"]: it for it in items}
    it = next(p for p in g.build_para(main) if p["template"] == "T2")
    lines = it["state"].split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("1. "))
    lines[i], lines[i + 1] = lines[i + 1].replace("2. ", "1. "), lines[i].replace("1. ", "2. ")
    bad = dict(it, state="\n".join(lines))
    with pytest.raises(AssertionError):
        g.validate_para(bad, main[f"names-{it['base_id']}"])
