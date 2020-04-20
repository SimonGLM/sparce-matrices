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
    in1, in2 = np.ones((50, 50))*2.22044605e-10, np.ones((50, 50))*2.22044605e-10
    in1_zeros = np.allclose(in1, np.zeros_like(in1))
    in2_zeros = np.allclose(in2, np.zeros_like(in2))
    if in1_zeros or in2_zeros:
        with pytest.raises(sp.AllZeroError):
            sp.sparse(in1)
            sp.sparse(in2)
        return
    sp1 = sp.sparse(in1)
    sp2 = sp.sparse(in2)
    # assert np.allclose(in1+in2, sp1.__add__(sp2).toarray(), sp1.__add__(in1).toarray())
    # assert np.allclose(in1+in2, (sp1+sp2).toarray(), (sp1+in1).toarray())
    # assert np.allclose(in1-in2, sp1.__sub__(sp2).toarray(), sp1.__sub__(in1).toarray())
    # assert np.allclose(in1-in2, (sp1-sp2).toarray(), (sp1-in1).toarray())
    assert np.allclose(in1@in2,  sp1.__matmul__(sp2).toarray(), sp1.__matmul__(in1).toarray())
    # assert np.allclose(in1@in2, (sp1@sp2).toarray(), (sp1@in1).toarray())
    print("success")


if __name__ == "__main__":
    blank()
    # a = np.ones((5, 5))
    # a[1, 1] = 1e-14
    # print(a)
    # a *= ~np.isclose(a, np.zeros_like(a))
    # print(a)
    # res = sp.sparse(a)
    pass
