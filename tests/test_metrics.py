"""Metrics: joint/MATE/MASC + reproducible paired bootstrap."""
from metrics import joint_prf, mate_prf, masc_acc_f1, paired_bootstrap


def test_joint_prf_perfect():
    preds = [[(0, 2, "POS")], [(1, 2, "NEG")]]
    golds = [[(0, 2, "POS")], [(1, 2, "NEG")]]
    r = joint_prf(preds, golds)
    assert r["F1"] == 100.0


def test_joint_vs_mate():
    preds = [[(0, 2, "POS")]]   # right span, wrong polarity
    golds = [[(0, 2, "NEG")]]
    assert joint_prf(preds, golds)["F1"] == 0.0
    assert mate_prf(preds, golds)["F1"] == 100.0  # span correct


def test_masc():
    r = masc_acc_f1(["POS", "NEG", "NEU"], ["POS", "NEG", "POS"])
    assert abs(r["Acc"] - 200 / 3) < 1e-6  # 2/3


def test_bootstrap_reproducible():
    golds = [[(0, 1, "POS")] for _ in range(20)]
    a = [[(0, 1, "POS")] for _ in range(20)]            # perfect
    b = [[(0, 1, "NEG")] for _ in range(20)]            # all wrong polarity
    r1 = paired_bootstrap(a, b, golds, n_samples=200, seed=1)
    r2 = paired_bootstrap(a, b, golds, n_samples=200, seed=1)
    assert r1["p_value"] == r2["p_value"]
    assert r1["significant"] is True
