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


def test_sim2_copy_is_identical():
    sim2 = os.path.join(REPO, "numerical_simulations", "sim2", "my_modules", "coupling_oscillator_bayes.py")
    here = os.path.join(EEG_DIR, "my_modules", "coupling_oscillator_bayes.py")
    assert filecmp.cmp(sim2, here, shallow=False)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
