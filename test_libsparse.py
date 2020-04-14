import libsparse
import sys
from pympler import asizeof
import progressbar
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import lists, integers, floats, tuples

VALUE_RANGE = (-100, 100)


@given(int_array=lists(lists(integers(min_value=VALUE_RANGE[0], max_value=VALUE_RANGE[1]))),
       float_array=lists(lists(integers(min_value=VALUE_RANGE[0], max_value=VALUE_RANGE[1]))))
def test_csr_construct(int_array, float_array):
    """
    Author: Simon Glennemeier-Marke
    """
    assert np.alltrue(sp.sparse(int_array).toarray() == scipy.sparse.construct.csr_matrix(
        tuple(sp.sparse(int_array).CSR.values())).toarray())
    assert np.alltrue(sp.sparse(float_array).toarray() == scipy.sparse.construct.csr_matrix(
        tuple(sp.sparse(float_array).CSR.values())).toarray())


def test_mem_overhead():
    size_sparse = []
    size_numpy = []
    bar1 = progressbar.ProgressBar(min_value=10, max_value=1000)
    for N in range(10, 1000, 10):
        mat = libsparse.random_banded(N, N//4)
        size_numpy.append(asizeof.asizeof(mat))
        size_sparse.append(asizeof.asizeof(libsparse.sparse(mat)))
        # print(N, size_numpy[-1]//1000, "kB", size_sparse[-1]//1000, "kB")
        bar1.update(N)
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
    size_sparse = []
    size_numpy = []
    density = np.arange(0, 1, 0.1)
    bar2 = progressbar.ProgressBar(min_value=0, max_value=1)
    for rho in density:
        mat = scipy.sparse.random(100, 100, density=rho).toarray()
        size_numpy.append(asizeof.asizeof(mat))
        size_sparse.append(asizeof.asizeof(libsparse.sparse(mat)))
        # print(rho, str(size_numpy[-1])+" B", str(size_sparse[-1])+" B")
        bar2.update(rho)
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


if __name__ == "__main__":
    # test_mem_overhead()
    # test_mem_efficiency()
    pass
