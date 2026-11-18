"""Regression tests for my_modules/coupling_oscillator_bayes.py.

Reference: save_data/est_model/bayse_est_chb15_06.npy, written by eeg_bayes_est_main.py with the
original (pre-replacement) CouplingOscillator (P = 3, prec_param = 1/0.1, Tstep = 1).

Run:  python -m pytest seizure_EEG/tests   or   python seizure_EEG/tests/test_coupling_oscillator_bayes.py
"""
import filecmp
import os
import sys

import numpy as np

EEG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(EEG_DIR)
sys.path.insert(0, EEG_DIR)
from my_modules.coupling_oscillator_bayes import CouplingOscillator, log_evidence_grid

SAVED = os.path.join(EEG_DIR, "save_data", "est_model", "bayse_est_chb15_06.npy")


def _load():
    return np.load(SAVED, allow_pickle=True).item()


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(b).max()


def test_reproduces_original_saved_outputs():
    d = _load()
    m = CouplingOscillator(d["theta"].T, int(d["Popt"]), 1, 1 / d["fs"], 1 / 1e-1).fit_model()
    assert _rel(m.coeff_cos, d["K1est"]) < 1e-9
    assert _rel(m.coeff_sin, d["K2est"]) < 1e-9
    assert _rel(m.omega[-1], d["omega_est"]) < 1e-9
    assert _rel(m.y_hat, d["dtheta_pred"]) < 1e-9          # per-sample filtered predictions


def test_block_size_does_not_change_outputs():
    x = _load()["theta"].T[:600]
    ref = CouplingOscillator(x, 2, 1, 0.01, 2.0, block_size=1).fit_model()
    for bs in (37, 128, 1000):
        m = CouplingOscillator(x, 2, 1, 0.01, 2.0, block_size=bs).fit_model()
        for key in ("loglike", "y_hat", "omega", "coeff_cos", "coeff_sin"):
            assert _rel(getattr(m, key), getattr(ref, key)) < 1e-10, (bs, key)


def test_evidence_grid_equals_sum_of_one_step_loglike():
    x = _load()["theta"].T[:500]
    P_list, prec_list = [2, 3], [2.0, 0.2]
    ev = log_evidence_grid(x, 0.01, P_list, prec_list)
    for a, P in enumerate(P_list):
        for b, prec in enumerate(prec_list):
            m = CouplingOscillator(x, P, 1, 0.01, prec).fit_model()
            s = m.loglike.sum()
            assert abs(ev["loglike_sum"][a, b] - s) / abs(s) < 1e-10
            k = m.Nosc * (m.Nosc * 2 * P + 1)
            assert np.isclose(ev["BIC"][a, b], k * np.log(m.omega.shape[0]) - 2 * s, rtol=1e-10)


def _uniform_rotation():
    """One oscillator rotating by pi/4 per step: every coupling regressor is identically zero, so
    the model reduces to y_t = omega + noise with omega ~ N(0, 1), whose evidence is analytic."""
    x = ((np.arange(1001) % 8) * (np.pi / 4))[:, None]
    h, n = 0.01, 1000
    c = (np.pi / 4) / h

    def analytic(prec):
        s2 = 1 / prec
        return -0.5 * (n * np.log(2 * np.pi) + (n - 1) * np.log(s2) + np.log(s2 + n)
                       + n * c ** 2 / (s2 + n))
    return x, h, analytic


def test_evidence_grid_matches_analytic_at_high_precision():
    x, h, analytic = _uniform_rotation()
    precs = [2.0, 1e6, 1e10]
    ev = log_evidence_grid(x, h, [1, 2], precs)
    for b, prec in enumerate(precs):
        for a in range(2):
            assert abs(ev["loglike_sum"][a, b] - analytic(prec)) <= 1e-10 * abs(analytic(prec)), (a, prec)


def test_evidence_grid_collinear_columns_matches_analytic():
    """N oscillators with identical phases rotating uniformly: every non-zero regressor of an
    oscillator is a column of ones (1 + P(N-1) of them), so F F^T = m 11^T is rank one."""
    N, P, h, n = 4, 2, 0.01, 1000
    x = np.repeat(((np.arange(n + 1) % 8) * (np.pi / 4))[:, None], N, axis=1)
    c, m = (np.pi / 4) / h, 1 + P * (N - 1)
    for prec in (2.0, 1e6, 1e10):
        s2 = 1 / prec
        ref = -0.5 * N * (n * np.log(2 * np.pi) + (n - 1) * np.log(s2) + np.log(s2 + n * m)
                          + n * c ** 2 / (s2 + n * m))
        got = log_evidence_grid(x, h, [P], [prec])["loglike_sum"][0, 0]
        assert abs(got - ref) <= 1e-10 * abs(ref), (prec, got, ref)


def test_dense_precision_grid_matches_sequential_fits():
    x = _load()["theta"].T[:500]
    precs = np.logspace(np.log10(0.02), np.log10(2.0), 300)
    ev = log_evidence_grid(x, 0.01, [2, 4], precs)
    for a, P in enumerate([2, 4]):
        for b in (0, 137, 299):
            s = CouplingOscillator(x, P, 1, 0.01, precs[b]).fit_model().loglike.sum()
            assert abs(ev["loglike_sum"][a, b] - s) / abs(s) < 1e-10, (P, b)


def test_concurrent_grid_calls_restore_blas_threads():
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        return
    import threading
    x = _load()["theta"].T[:300]
    with threadpool_limits(limits=3, user_api="blas"):
        before = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"]
        out = [None, None]

        def run(j):
            out[j] = log_evidence_grid(x, 0.01, [2, 3], [2.0, 0.2])["loglike_sum"]
        th = [threading.Thread(target=run, args=(j,)) for j in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        after = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"]
    assert before == after, (before, after)
    assert np.array_equal(out[0], out[1])


def test_sequential_fit_matches_analytic():
    x, h, analytic = _uniform_rotation()
    for prec, rtol in ((2.0, 1e-12), (1e6, 1e-10), (1e10, 1e-7)):
        s = CouplingOscillator(x, 2, 1, h, prec).fit_model().loglike.sum()
        assert abs(s - analytic(prec)) <= rtol * abs(analytic(prec)), prec


def test_invalid_arguments_raise():
    x = _load()["theta"].T[:50]
    bad_init = [dict(block_size=0), dict(block_size=-1), dict(block_size=2.5), dict(block_size=True),
                dict(P=0), dict(P=2.0), dict(prec_param=0.0), dict(prec_param=-1.0),
                dict(prec_param=np.inf), dict(h=0.0)]
    for kw in bad_init:
        args = dict(P=2, h=0.01, prec_param=2.0, block_size=128)
        args.update(kw)
        try:
            CouplingOscillator(x, args["P"], 1, args["h"], args["prec_param"], block_size=args["block_size"])
        except ValueError:
            continue
        raise AssertionError(f"accepted {kw}")
    for kw in (dict(chunk=0), dict(chunk=-1), dict(P_list=[0]), dict(P_list=[]),
               dict(prec_param_list=[-1.0]), dict(prec_param_list=[])):
        args = dict(P_list=[2], prec_param_list=[2.0], chunk=2048)
        args.update(kw)
        try:
            log_evidence_grid(x, 0.01, args["P_list"], args["prec_param_list"], chunk=args["chunk"])
        except ValueError:
            continue
        raise AssertionError(f"accepted {kw}")
    for bad_x in (x[:1], x[:, 0], np.zeros((5, 0)), np.where(np.arange(50)[:, None] == 3, np.nan, x)):
        try:
            CouplingOscillator(bad_x, 2, 1, 0.01, 2.0).fit_model()
        except ValueError:
            continue
        raise AssertionError("accepted invalid phases")


def test_sim2_copy_is_identical():
    sim2 = os.path.join(REPO, "numerical_simulations", "sim2", "my_modules", "coupling_oscillator_bayes.py")
    here = os.path.join(EEG_DIR, "my_modules", "coupling_oscillator_bayes.py")
    assert filecmp.cmp(sim2, here, shallow=False)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
