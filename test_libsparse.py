import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
from hypothesis import given, settings, seed
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, floats, integers, lists, tuples
from pympler import asizeof

import libsparse as sp
sp.hide_DensityWarning = True

MIN_VALUE = -1e10
MAX_VALUE = 1e10
SHAPE = (50, 50)


@composite
def shape_generator(draw):
    """
    Author Simon Glennemeier-Marke
    """
    unequal = draw(tuples(integers(min_value=2, max_value=100),
                          integers(min_value=2, max_value=100)).filter(lambda x: x[0] != x[1]))
    equal = draw(tuples(integers(min_value=2, max_value=100),
                        integers(min_value=2, max_value=100)).filter(lambda x: x[0] == x[1]))
    shape = (equal, unequal)[np.random.choice([0, 1])]

    return shape

@given(int_array=arrays(np.int, shape=SHAPE, elements=integers(min_value=MIN_VALUE, max_value=MAX_VALUE)),
       float_array=arrays(np.float, shape=SHAPE, elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)))
def test_nonzero_csr_construct(int_array, float_array):
    """
    Author: Simon Glennemeier-Marke
    """
    int_zero = np.alltrue(int_array == np.zeros_like(int_array))
    float_zero = np.alltrue(float_array == np.zeros_like(float_array))
    if int_zero or float_zero:
        with pytest.raises(sp.AllZeroError):
            assert sp.sparse(int_array)
            assert sp.sparse(float_array)
    else:
        assert np.alltrue(sp.sparse(int_array).toarray() == scipy.sparse.construct.csr_matrix(
            tuple(sp.sparse(int_array).CSR.values())).toarray())
        assert np.alltrue(sp.sparse(float_array).toarray() == scipy.sparse.construct.csr_matrix(
            tuple(sp.sparse(float_array).CSR.values())).toarray())


def test_AllZeroError():
    """
    Author: Simon Glennemeier-Marke
    """
    with pytest.raises(sp.AllZeroError):
        assert sp.sparse(np.zeros((10, 10)))


@given(in1=arrays(np.float, shape=SHAPE, elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)).filter(lambda x: np.alltrue(x != np.zeros_like(x))))
def test_matrix_algebra(in1):
    """
    Author: Simon Glennemeier-Marke
    """
    sp1 = sp.sparse(in1)
    # Transposition
    assert np.allclose(in1.transpose(), sp1.T().toarray())


@settings(deadline=None)
@given(in1=arrays(np.float, shape=SHAPE, elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)),
       in2=arrays(np.float, shape=SHAPE, elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)))
def test_matrix_matrix_algebra(in1, in2):
    """
    Author: Simon Glennemeier-Marke
    """
    in1_zeros = np.alltrue(in1 == np.zeros_like(in1))
    in2_zeros = np.alltrue(in2 == np.zeros_like(in2))
    if in1_zeros or in2_zeros:
        with pytest.raises(sp.AllZeroError):
            sp.sparse(in1)
            sp.sparse(in2)
    else:
        sp1 = sp.sparse(in1)
        sp2 = sp.sparse(in2)
        cond1 = np.allclose(in1+in2, sp1.__add__(sp2).toarray())
        # print("cond1 =", cond1)
        assert cond1
        cond2 = np.allclose(in1+in2, (sp1+sp2).toarray())
        # print("cond2 =", cond2)
        assert cond2
        cond3 = np.allclose(in1-in2, sp1.__sub__(sp2).toarray())
        # print("cond3 =", cond3)
        assert cond3
        cond4 = np.allclose(in1-in2, (sp1-sp2).toarray())
        # print("cond4 =", cond4)
        assert cond4
        cond5 = np.allclose(in1@in2, sp1.__matmul__(sp2).toarray())
        # print("cond5 =", cond5)
        assert cond5
        cond6 = np.allclose(in1@in2, (sp1@sp2).toarray())
        # print("cond6 =", cond6)
        assert cond6


@settings(deadline=None)
@given(t_mat=arrays(np.float, shape=SHAPE, elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE, allow_infinity=False, allow_nan=False)).filter(lambda x: np.alltrue(x != np.zeros_like(x))),
       t_vec=arrays(np.float, shape=SHAPE[0], elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE, allow_infinity=False, allow_nan=False)).filter(lambda x: np.alltrue(x != np.zeros_like(x))))
def test_matrix_vector_algebra(t_mat, t_vec):
    sp1 = sp.sparse(t_mat)
    with pytest.raises(sp.ShapeError):
        assert np.allclose(t_mat+t_vec, sp1.__add__(t_vec))
        assert np.allclose(t_mat+t_vec, (sp1+t_vec))
        assert np.allclose(t_mat-t_vec, sp1.__sub__(t_vec))
        assert np.allclose(t_mat-t_vec, (sp1-t_vec))
    assert np.allclose(t_mat@t_vec,  sp1.__matmul__(t_vec))
    assert np.allclose(t_mat @ t_vec, (sp1 @ t_vec))


@given(in1=arrays(np.float, shape=tuples(integers(min_value=2, max_value=50), integers(min_value=2, max_value=50)), elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)).filter(lambda x: np.alltrue(x != np.zeros_like(x))))
def test_quadratic(in1):
    """
    Author: Simon Glennemeier-Marke
    """
    np_bool = (in1.shape[0] == in1.shape[1])
    sp_bool = sp.quadratic(sp.sparse(in1))
    assert np_bool == sp_bool
    with pytest.raises(AttributeError):
        assert sp.quadratic("I cannot have a shape attribute")


def test_mem_overhead():
    """
    Author: Simon Glennemeier-Marke
    """
    size_sparse = []
    size_numpy = []
    for N in range(10, 1000, 10):
        mat = sp.random_banded(N, N//4)
        size_numpy.append(asizeof.asizeof(mat))
        size_sparse.append(asizeof.asizeof(sp.sparse(mat)))
    fig = plt.figure()
    name = "mem_overhead"
    fig: plt.Figure
    ax1 = fig.add_subplot(1, 1, 1)
    ax1: plt.Axes
    ax1.plot(range(len(size_numpy)), size_numpy, label="numpy")
    ax1.plot(range(len(size_sparse)), size_sparse, label="sparse")
    ax1.set_title(name)
    ax1.set_xlabel("N")
    ax1.set_ylabel("bytes")
    ax1.legend()
    plt.savefig(name + ".png")
    assert True


def test_mem_efficiency():
    """
    Author: Simon Glennemeier-Marke
    """
    size_sparse = []
    size_numpy = []
    density = np.delete(np.arange(0, 1, 0.05), 0)

    for rho in density:
        mat = scipy.sparse.random(1000, 1000, density=rho)
        size_numpy.append(asizeof.asizeof(mat.toarray()))
        size_sparse.append(asizeof.asizeof(sp.sparse(mat.toarray())))
    fig = plt.figure()
    name = "mem_efficiency"
    fig: plt.Figure
    ax1 = fig.add_subplot(1, 1, 1)
    ax1: plt.Axes
    ax1.plot(density, size_numpy, label="numpy")
    ax1.plot(density, size_sparse, label="sparse")
    ax1.set_title(name)
    ax1.set_xlabel("density")
    ax1.set_ylabel("bytes")
    ax1.legend()
    plt.savefig(name + ".png")
    assert True


# MIN_VALUE = -1e8
# MAX_VALUE = 1e8


@composite
def system_generator(draw):
    n = draw(integers(min_value=20, max_value=100))
    mat = scipy.sparse.random(n, n, density=draw(
        floats(min_value=0.05, max_value=0.20))).toarray() + np.eye(n)
    vec = draw(arrays("float64", shape=(n,), elements=floats(
        min_value=MIN_VALUE, max_value=MAX_VALUE)))

    assert mat.shape[0] == mat.shape[1]
    return mat, vec


@settings(deadline=None)
@given(insys=system_generator())
def test_linsys(insys):
    matrices, vectors = insys
    sys = sp.linsys(sp.sparse(matrices), vectors)

    sp_sol_scipy = sys.solve(method='scipy')
    sp_sol_lu = sys.solve(method='lu')

    symm_mat = np.eye(matrices.shape[0])+(matrices+np.transpose(matrices))/2
    symm_sys = sp.linsys(sp.sparse(symm_mat), vectors)
    sp_sol_cg = symm_sys.solve(method='cg')

    true_sol = scipy.linalg.solve(matrices, vectors)
    true_sol_cg = scipy.linalg.solve(symm_mat, vectors)

    assert np.allclose(sp_sol_scipy, true_sol)
    assert np.allclose(sp_sol_lu, true_sol)
    assert np.allclose(sp_sol_cg, true_sol_cg, rtol=1e-2)


if __name__ == "__main__":
    pass
