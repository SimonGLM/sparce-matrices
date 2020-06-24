import numpy as np
import scipy
import time
import libsparse as sp
import pytest


def timer(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        f = func(args, kwargs)
        t2 = time.time()
        print(f"{func.__name__} took {t2-t1} seconds to complete")
        return f
    return wrapper


@timer
def blank(*args, **kwargs):
    in1, in2 = np.ones((50, 50))*2.22044605e-6, np.ones((50, 50))*2.22044605e-6
    sp1 = sp.sparse(in1)
    sp2 = sp.sparse(in2)
    cond11 = np.allclose(in1+in2, sp1.__add__(sp2).toarray())
    cond12 = np.allclose(in1+in2, sp1.__add__(in1).toarray())
    print("cond11 =", cond11, " and cond12 =", cond12)
    # assert cond11 and cond12
    cond21 = np.allclose(in1+in2, (sp1+sp2).toarray())
    cond22 = np.allclose(in1+in2, (sp1+in1).toarray())
    print("cond21 =", cond21, " and cond22 =", cond22)
    # assert cond21 and cond22
    cond31 = np.allclose(in1-in2, sp1.__sub__(sp2).toarray())
    cond32 = np.allclose(in1-in2, sp1.__sub__(in1).toarray())
    print("cond31 =", cond31, " and cond32 =", cond32)
    # assert cond31 and cond32
    cond41 = np.allclose(in1-in2, (sp1-sp2).toarray())
    cond42 = np.allclose(in1-in2, (sp1-in1).toarray())
    print("cond41 =", cond41, " and cond42 =", cond42)
    # assert cond41 and cond42
    cond51 = np.allclose(in1@in2, sp1.__matmul__(sp2).toarray())
    cond52 = np.allclose(in1@in2, sp1.__matmul__(in1).toarray())
    print("cond51 =", cond51, " and cond52 =", cond52)
    # assert cond51 and cond52
    cond61 = np.allclose(in1@in2, (sp1@sp2).toarray())
    cond62 = np.allclose(in1@in2, (sp1@in1).toarray())
    print("cond61 =", cond61, " and cond62 =", cond62)
    # assert cond61 and cond62


if __name__ == "__main__":
    sp.hide_DensityWarning = True

    blank()
    # a = np.random.random(size=(10, 20))
    # # a = np.ones((5, 5))
    # # a[1, 1] = 1e-14
    # # print(a)
    # # a *= ~np.isclose(a, np.zeros_like(a))
    # # print(a)
    # b = sp.sparse(a)
    # print(b[5, 5000])
    pass
