import libsparse as sp
import sys
from pympler import asizeof
# import progressbar
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import lists, integers, floats, tuples, composite
from hypothesis.extra.numpy import arrays

MIN_VALUE = 1
MAX_VALUE = 100
SHAPE = (50, 50)


@composite
def shape_generator(draw):
    unequal = draw(tuples(integers(min_value=2, max_value=100),
                          integers(min_value=2, max_value=100)).filter(lambda x: x[0] != x[1]))
    equal = draw(tuples(integers(min_value=2, max_value=100),
                        integers(min_value=2, max_value=100)).filter(lambda x: x[0] == x[1]))
    shape = (equal, unequal)[np.random.choice([0, 1])]

    return shape


@given(int_array=arrays("int32", shape=SHAPE, elements=integers(min_value=MIN_VALUE, max_value=MAX_VALUE)),
       float_array=arrays("float64", shape=SHAPE, elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)))
def test_csr_construct(int_array, float_array):
    """
    Author: Simon Glennemeier-Marke
    """
    assert np.alltrue(sp.sparse(int_array).toarray() == scipy.sparse.construct.csr_matrix(
        tuple(sp.sparse(int_array).CSR.values())).toarray())
    assert np.alltrue(sp.sparse(float_array).toarray() == scipy.sparse.construct.csr_matrix(
        tuple(sp.sparse(float_array).CSR.values())).toarray())


def test_zeros_valueerror():
    """
    Author: Simon Glennemeier-Marke
    """
    with pytest.raises(ValueError):
        assert sp.sparse(np.zeros((10, 10)))


@given(in1=arrays("float64", shape=SHAPE, elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)))
def test_matrix_algebra(in1):
    """
    Author: Simon Glennemeier-Marke
    """
    sp1 = sp.sparse(in1)
    # Transposition
    assert np.allclose(in1.transpose(), sp1.T().toarray())


def test_matrix_matrix_algebra(in1, in2):
    sp1 = sp.sparse(in1)
    sp2 = sp.sparse(in2)
    assert np.allclose(in1+in2, sp1.__add__(sp2))
    assert np.allclose(in1-in2, sp1.__sub__(sp2))
    assert np.allclose(in1@in2,  sp1.__mul__(sp2))


@given(in1=arrays("float64", shape=shape_generator(), elements=floats(min_value=MIN_VALUE, max_value=MAX_VALUE)))
def test_quadratic(in1):
    np_bool = (in1.shape[0] == in1.shape[0])
    sp_bool = sp.quadratic(sp.sparse(in1))
    assert np_bool == sp_bool


def test_mem_overhead():
    """
    Author: Simon Glennemeier-Marke
    """
    size_sparse = []
    size_numpy = []
    # bar1 = progressbar.ProgressBar(min_value=10, max_value=1000)
    for N in range(10, 1000, 10):
        mat = sp.random_banded(N, N//4)
        size_numpy.append(asizeof.asizeof(mat))
        size_sparse.append(asizeof.asizeof(sp.sparse(mat)))
        # print(N, size_numpy[-1]//1000, "kB", size_sparse[-1]//1000, "kB")
        # bar1.update(N)
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

    # bar2 = progressbar.ProgressBar(min_value=0, max_value=1)
    for rho in density:
        mat = scipy.sparse.random(100, 100, density=rho).toarray()
        size_numpy.append(asizeof.asizeof(mat))
        size_sparse.append(asizeof.asizeof(sp.sparse(mat)))
        # print(rho, str(size_numpy[-1])+" B", str(size_sparse[-1])+" B")
        # bar2.update(rho)
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


@settings(deadline=500)
@given(
    insys=system_generator()
)
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
