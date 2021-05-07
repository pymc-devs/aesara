import numpy as np
import pytest


scipy = pytest.importorskip("scipy")

from functools import partial

from aesara import tensor as aet
from aesara.compile.mode import get_default_mode
from aesara.configdefaults import config
from aesara.tensor import inplace
from tests import unittest_tools as utt
from tests.tensor.utils import (
    _good_broadcast_unary_chi2sf,
    _good_broadcast_unary_normal,
    _good_broadcast_unary_normal_float,
    _good_broadcast_unary_normal_float_no_complex,
    _good_broadcast_unary_normal_float_no_complex_small_neg_range,
    _good_broadcast_unary_normal_no_complex,
    _grad_broadcast_unary_0_2_no_complex,
    _grad_broadcast_unary_abs1_no_complex,
    _grad_broadcast_unary_normal,
    _grad_broadcast_unary_normal_small_neg_range,
    check_floatX,
    copymod,
    makeBroadcastTester,
    rand_ranged,
    randint_ranged,
    upcast_int8_nfunc,
)


imported_scipy_special = False
mode_no_scipy = get_default_mode()
try:
    import scipy.special
    import scipy.stats

    imported_scipy_special = True
except ImportError:
    if config.mode == "FAST_COMPILE":
        mode_no_scipy = "FAST_RUN"


def scipy_special_gammau(k, x):
    return scipy.special.gammaincc(k, x) * scipy.special.gamma(k)


def scipy_special_gammal(k, x):
    return scipy.special.gammainc(k, x) * scipy.special.gamma(k)


# We can't test it if scipy is not installed!
# Precomputing the result is brittle(it have been broken!)
# As if we do any modification to random number here,
# The input random number will change and the output!
if imported_scipy_special:
    expected_erf = scipy.special.erf
    expected_erfc = scipy.special.erfc
    expected_erfinv = scipy.special.erfinv
    expected_erfcinv = scipy.special.erfcinv
    expected_gamma = scipy.special.gamma
    expected_gammaln = scipy.special.gammaln
    expected_psi = scipy.special.psi
    expected_tri_gamma = partial(scipy.special.polygamma, 1)
    expected_chi2sf = scipy.stats.chi2.sf
    expected_gammainc = scipy.special.gammainc
    expected_gammaincc = scipy.special.gammaincc
    expected_gammau = scipy_special_gammau
    expected_gammal = scipy_special_gammal
    expected_j0 = scipy.special.j0
    expected_j1 = scipy.special.j1
    expected_jv = scipy.special.jv
    expected_i0 = scipy.special.i0
    expected_i1 = scipy.special.i1
    expected_iv = scipy.special.iv
    expected_erfcx = scipy.special.erfcx
    expected_sigmoid = scipy.special.expit
    skip_scipy = False
else:
    expected_erf = []
    expected_erfc = []
    expected_erfcx = []
    expected_erfinv = []
    expected_erfcinv = []
    expected_gamma = []
    expected_gammaln = []
    expected_psi = []
    expected_tri_gamma = []
    expected_chi2sf = []
    expected_gammainc = []
    expected_gammaincc = []
    expected_gammau = []
    expected_gammal = []
    expected_j0 = []
    expected_j1 = []
    expected_jv = []
    expected_i0 = []
    expected_i1 = []
    expected_iv = []
    expected_sigmoid = (
        upcast_int8_nfunc(
            lambda inputs: check_floatX(inputs, np.log1p(np.exp(inputs)))
        ),
    )
    skip_scipy = "scipy is not present"

TestErfBroadcast = makeBroadcastTester(
    op=aet.erf,
    expected=expected_erf,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)
TestErfInplaceBroadcast = makeBroadcastTester(
    op=inplace.erf_inplace,
    expected=expected_erf,
    good=_good_broadcast_unary_normal_float,
    mode=mode_no_scipy,
    eps=2e-10,
    inplace=True,
    skip=skip_scipy,
)

TestErfcBroadcast = makeBroadcastTester(
    op=aet.erfc,
    expected=expected_erfc,
    good=_good_broadcast_unary_normal_float_no_complex,
    grad=_grad_broadcast_unary_normal,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)
TestErfcInplaceBroadcast = makeBroadcastTester(
    op=inplace.erfc_inplace,
    expected=expected_erfc,
    good=_good_broadcast_unary_normal_float_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestErfcxBroadcast = makeBroadcastTester(
    op=aet.erfcx,
    expected=expected_erfcx,
    good=_good_broadcast_unary_normal_float_no_complex_small_neg_range,
    grad=_grad_broadcast_unary_normal_small_neg_range,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)
TestErfcxInplaceBroadcast = makeBroadcastTester(
    op=inplace.erfcx_inplace,
    expected=expected_erfcx,
    good=_good_broadcast_unary_normal_float_no_complex_small_neg_range,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestErfinvBroadcast = makeBroadcastTester(
    op=aet.erfinv,
    expected=expected_erfinv,
    good={
        "normal": [rand_ranged(-0.9, 0.9, (2, 3))],
        "empty": [np.asarray([], dtype=config.floatX)],
    },
    grad=_grad_broadcast_unary_abs1_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestErfcinvBroadcast = makeBroadcastTester(
    op=aet.erfcinv,
    expected=expected_erfcinv,
    good={
        "normal": [rand_ranged(0.001, 1.9, (2, 3))],
        "empty": [np.asarray([], dtype=config.floatX)],
    },
    grad=_grad_broadcast_unary_0_2_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

_good_broadcast_unary_gammaln = dict(
    normal=(rand_ranged(-1 + 1e-2, 10, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(randint_ranged(1, 10, (2, 3)),),
    uint8=(randint_ranged(1, 6, (2, 3)).astype("uint8"),),
    uint16=(randint_ranged(1, 10, (2, 3)).astype("uint16"),),
    uint64=(randint_ranged(1, 10, (2, 3)).astype("uint64"),),
)
_grad_broadcast_unary_gammaln = dict(
    # smaller range as our grad method does not estimate it well enough.
    normal=(rand_ranged(1e-1, 8, (2, 3)),),
)

TestGammaBroadcast = makeBroadcastTester(
    op=aet.gamma,
    expected=expected_gamma,
    good=_good_broadcast_unary_gammaln,
    grad=_grad_broadcast_unary_gammaln,
    mode=mode_no_scipy,
    eps=1e-5,
    skip=skip_scipy,
)
TestGammaInplaceBroadcast = makeBroadcastTester(
    op=inplace.gamma_inplace,
    expected=expected_gamma,
    good=_good_broadcast_unary_gammaln,
    mode=mode_no_scipy,
    eps=1e-5,
    inplace=True,
    skip=skip_scipy,
)

TestGammalnBroadcast = makeBroadcastTester(
    op=aet.gammaln,
    expected=expected_gammaln,
    good=_good_broadcast_unary_gammaln,
    grad=_grad_broadcast_unary_gammaln,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)
TestGammalnInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammaln_inplace,
    expected=expected_gammaln,
    good=_good_broadcast_unary_gammaln,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

_good_broadcast_unary_psi = dict(
    normal=(rand_ranged(1, 10, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(randint_ranged(1, 10, (2, 3)),),
    uint8=(randint_ranged(1, 10, (2, 3)).astype("uint8"),),
    uint16=(randint_ranged(1, 10, (2, 3)).astype("uint16"),),
)

TestPsiBroadcast = makeBroadcastTester(
    op=aet.psi,
    expected=expected_psi,
    good=_good_broadcast_unary_psi,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)
TestPsiInplaceBroadcast = makeBroadcastTester(
    op=inplace.psi_inplace,
    expected=expected_psi,
    good=_good_broadcast_unary_psi,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

_good_broadcast_unary_tri_gamma = _good_broadcast_unary_psi

TestTriGammaBroadcast = makeBroadcastTester(
    op=aet.tri_gamma,
    expected=expected_tri_gamma,
    good=_good_broadcast_unary_psi,
    eps=2e-8,
    mode=mode_no_scipy,
    skip=skip_scipy,
)
TestTriGammaInplaceBroadcast = makeBroadcastTester(
    op=inplace.tri_gamma_inplace,
    expected=expected_tri_gamma,
    good=_good_broadcast_unary_tri_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestChi2SFBroadcast = makeBroadcastTester(
    op=aet.chi2sf,
    expected=expected_chi2sf,
    good=_good_broadcast_unary_chi2sf,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
    name="Chi2SF",
)

TestChi2SFInplaceBroadcast = makeBroadcastTester(
    op=inplace.chi2sf_inplace,
    expected=expected_chi2sf,
    good=_good_broadcast_unary_chi2sf,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
    name="Chi2SF",
)

_good_broadcast_binary_gamma = dict(
    normal=(rand_ranged(1e-2, 10, (2, 3)), rand_ranged(1e-2, 10, (2, 3))),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    int=(randint_ranged(1, 10, (2, 3)), randint_ranged(1, 10, (2, 3))),
    uint8=(
        randint_ranged(1, 6, (2, 3)).astype("uint8"),
        randint_ranged(1, 6, (2, 3)).astype("uint8"),
    ),
    uint16=(
        randint_ranged(1, 10, (2, 3)).astype("uint16"),
        randint_ranged(1, 10, (2, 3)).astype("uint16"),
    ),
    uint64=(
        randint_ranged(1, 10, (2, 3)).astype("uint64"),
        randint_ranged(1, 10, (2, 3)).astype("uint64"),
    ),
)

TestGammaIncBroadcast = makeBroadcastTester(
    op=aet.gammainc,
    expected=expected_gammainc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestGammaIncInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammainc_inplace,
    expected=expected_gammainc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestGammaInccBroadcast = makeBroadcastTester(
    op=aet.gammaincc,
    expected=expected_gammaincc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestGammaInccInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammaincc_inplace,
    expected=expected_gammaincc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestGammaUBroadcast = makeBroadcastTester(
    op=aet.gammau,
    expected=expected_gammau,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestGammaUInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammau_inplace,
    expected=expected_gammau,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestGammaLBroadcast = makeBroadcastTester(
    op=aet.gammal,
    expected=expected_gammal,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestGammaLInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammal_inplace,
    expected=expected_gammal,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

_good_broadcast_unary_bessel = dict(
    normal=(rand_ranged(-10, 10, (2, 3)),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(randint_ranged(-10, 10, (2, 3)),),
    uint8=(randint_ranged(0, 10, (2, 3)).astype("uint8"),),
    uint16=(randint_ranged(0, 10, (2, 3)).astype("uint16"),),
)

_grad_broadcast_unary_bessel = dict(
    normal=(rand_ranged(-10.0, 10.0, (2, 3)),),
)

_good_broadcast_binary_bessel = dict(
    normal=(rand_ranged(-5, 5, (2, 3)), rand_ranged(0, 10, (2, 3))),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    integers=(randint_ranged(-5, 5, (2, 3)), randint_ranged(-10, 10, (2, 3))),
    uint8=(
        randint_ranged(0, 5, (2, 3)).astype("uint8"),
        randint_ranged(0, 10, (2, 3)).astype("uint8"),
    ),
    uint16=(
        randint_ranged(0, 5, (2, 3)).astype("uint16"),
        randint_ranged(0, 10, (2, 3)).astype("uint16"),
    ),
)

_grad_broadcast_binary_bessel = dict(
    normal=(rand_ranged(1, 5, (2, 3)), rand_ranged(0, 10, (2, 3)))
)

TestJ0Broadcast = makeBroadcastTester(
    op=aet.j0,
    expected=expected_j0,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestJ0InplaceBroadcast = makeBroadcastTester(
    op=inplace.j0_inplace,
    expected=expected_j0,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestJ1Broadcast = makeBroadcastTester(
    op=aet.j1,
    expected=expected_j1,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestJ1InplaceBroadcast = makeBroadcastTester(
    op=inplace.j1_inplace,
    expected=expected_j1,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestJvBroadcast = makeBroadcastTester(
    op=aet.jv,
    expected=expected_jv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestJvInplaceBroadcast = makeBroadcastTester(
    op=inplace.jv_inplace,
    expected=expected_jv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)


def test_verify_jv_grad():
    # Verify Jv gradient.
    # Implemented separately due to need to fix first input for which grad is
    # not defined.
    v_val, x_val = _grad_broadcast_binary_bessel["normal"]

    def fixed_first_input_jv(x):
        return aet.jv(v_val, x)

    utt.verify_grad(fixed_first_input_jv, [x_val])


TestI0Broadcast = makeBroadcastTester(
    op=aet.i0,
    expected=expected_i0,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestI0InplaceBroadcast = makeBroadcastTester(
    op=inplace.i0_inplace,
    expected=expected_i0,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestI1Broadcast = makeBroadcastTester(
    op=aet.i1,
    expected=expected_i1,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestI1InplaceBroadcast = makeBroadcastTester(
    op=inplace.i1_inplace,
    expected=expected_i1,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)

TestIvBroadcast = makeBroadcastTester(
    op=aet.iv,
    expected=expected_iv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    skip=skip_scipy,
)

TestIvInplaceBroadcast = makeBroadcastTester(
    op=inplace.iv_inplace,
    expected=expected_iv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    skip=skip_scipy,
)


def test_verify_iv_grad():
    # Verify Iv gradient.
    # Implemented separately due to need to fix first input for which grad is
    # not defined.
    v_val, x_val = _grad_broadcast_binary_bessel["normal"]

    def fixed_first_input_iv(x):
        return aet.iv(v_val, x)

    utt.verify_grad(fixed_first_input_iv, [x_val])


TestSigmoidBroadcast = makeBroadcastTester(
    op=aet.sigmoid,
    expected=expected_sigmoid,
    good=_good_broadcast_unary_normal_no_complex,
    eps=1e-8,
)

TestSigmoidInplaceBroadcast = makeBroadcastTester(
    op=inplace.sigmoid_inplace,
    expected=expected_sigmoid,
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal,
    eps=1e-8,
    inplace=True,
)


class TestSigmoid:
    def setup_method(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(aet.sigmoid, [np.random.rand(3, 4)])


_good_broadcast_unary_softplus = dict(
    copymod(
        _good_broadcast_unary_normal_no_complex,
        without=["uint8", "uint16", "big_scalar"],
    ),  # numpy function overflows with uint16.
    uint8=[
        np.arange(0, 89, dtype="uint8")
    ],  # the range is different in new added uint8.
    int8=[np.arange(-127, 89, dtype="int8")],
)

expected_sofplus = upcast_int8_nfunc(
    lambda inputs: check_floatX(inputs, np.log1p(np.exp(inputs)))
)

TestSoftplusBroadcast = makeBroadcastTester(
    op=aet.softplus,
    expected=expected_sofplus,
    good=_good_broadcast_unary_softplus,
    eps=1e-8,
)

TestSoftplusInplaceBroadcast = makeBroadcastTester(
    op=inplace.softplus_inplace,
    expected=expected_sofplus,
    good=_good_broadcast_unary_softplus,
    grad=_grad_broadcast_unary_normal,
    eps=1e-8,
    inplace=True,
)


class TestSoftplus:
    def setup_method(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(aet.softplus, [np.random.rand(3, 4)])

    def test_accuracy(self):
        # Test all aproximations are working (cutoff points are -37, 18, 33.3)
        x_test = np.array([-40.0, -17.5, 17.5, 18.5, 40.0])
        y_th = aet.softplus(x_test).eval()
        y_np = np.log1p(np.exp(x_test))
        np.testing.assert_allclose(y_th, y_np, rtol=10e-10)
