import libsparse as sp
import numpy as np
import scipy
if __name__ == "__main__":
    mat = sp.random_banded(5, 3)
    assert np.alltrue(sp.sparse(mat).toarray() == scipy.sparse.construct.csr_matrix(tuple(sp.sparse(mat).CSR.values())).toarray()))
    pass
