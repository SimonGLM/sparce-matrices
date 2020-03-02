'''
Sparse
======
This library implements sparse matrices.

`import libsparse as sp`


Sparse matricies are matricies which have relativley few non-zero entries.

Therefore it is inefficient to store them as a complete array object.

Contents
--------
sp.sparse(array: np.ndarray):
> The 'sparse' object provided by this library will implement the 'Compressed Sparse Row' format.
> The CSR format stores all values in a list and only the column index and where new rows begin.

sp.random_banded(size: int, num_diags: int):
> Creates, displays and returns a sparse `size`×`size` matrix which is banded.
> I.e. it only has non-zero values on num_diags centered diagonals.


This library is part of a project done as an end-term assignment in the 'Scientific Programming'-course at the Justus-Liebig-University in Giessen, Germany.
'''

import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
rng = np.random.default_rng()


class sparse(object):
    # TODO NEEDSDOC
    '''
    Author: Simon Glennemeier-Marke & Henrik Spielvogel

    A sparse array object
    =====================

    This object is the general sparse matrix object

    Arguments:
    > array : np.ndarray of arbitrary size
    '''

    def __init__(self, array: np.ndarray):
        self.INCOMING = array if (type(array) == np.ndarray) else np.asfarray(array)
        self.sparsity = 1 - np.count_nonzero(self.INCOMING)/self.INCOMING.size
        self._choose_scheme(self.INCOMING)

    def __repr__(self):
        return '<sparse matrix of shape {} and sparsity {:.2f}>'.format(self.INCOMING.shape, self.sparsity)

    def __mul__(self, other):
        return self.matvec(other)

    def construct_CSR(self, array):
        # TODO NEEDSDOC
        '''
        Author: Simon Glennemeier-Marke

        Constructs a CSR form of a given array.

        Args:
        > `INCOMING` :  sparse numpy array

        Returns:
        > self.CSR :  dict containing the CSR object
        '''
        csr = {'AVAL': [], 'JCOL': [], 'IROW': [0]}
        for j, col in enumerate(array):
            for i, el in enumerate(col):
                if el != 0:
                    csr['AVAL'].append(el)
                    csr['JCOL'].append(i)
                continue
            csr['IROW'].append(len(csr['AVAL']))

        return csr

    def _choose_scheme(self, array):
        # "_method" means python won't import this method with wildcard import "from lib import * "
        '''
        Author: Simon Glennemeier-Marke

        Decide which storage scheme to use based on matrix density.

        Args:
        > array : np.ndarray
        '''
        if 1 > self.sparsity >= .5:
            self.CSR = self.construct_CSR(array)
        elif .5 > self.sparsity > 0:
            print('NotImplementedError: falling back to implemented methods')
            self.CSR = self.construct_CSR(array)
        else:
            raise ValueError('Sparisty should be in open interval (0,1), but is {:.3f}'.format(self.sparsity))

        pass

    # TODO: Needs class methods for gaussian elimination etc...

    def matvec(self, vec):
        '''Author: Henrik Spielvogel

        Calculates the matrix-vector product of a sparse matrix with `vec`.

        Args:
        > `vec` :  list or array of same length as matrix

        Returns:
        > outvec :  np.ndarray

        '''

        n = self.INCOMING.shape[0]
        vec = vec if type(vec) == np.ndarray else np.array(vec)
        outvec = []

        if vec.shape[0] == n:
            for i in range(n):
                val = 0
                for j in np.arange(self.CSR['IROW'][i], self.CSR['IROW'][i+1]):
                    val += vec[self.CSR['JCOL'][j]] * self.CSR['AVAL'][j]
                outvec.append(val)
        else:
            raise ValueError('Shape of vec must be ({},), but is {}.'.format(n, vec.shape))

        return np.array(outvec)


def random_banded(size, num_diags):
    # TODO NEEDSDOC
    '''
    Author: Simon Glennemeier-Marke

    Create quadratic banded sparse matrix of dimension 'size' with 'num_diags' diagonals

    '''
    mat = scipy.sparse.diags([rng.uniform(0, 1, size=size) for i in range(num_diags)], range((-num_diags+1)//2, (num_diags+1)//2), shape=(size, size)).toarray()
    return scipy.sparse.eye(size)+(mat+np.transpose(mat))/2


if __name__ == "__main__":
    # TESTING
    N = 1000
    a = sparse(np.eye(N))  # random_banded(N, 2))
    vector = [rng.integers(-10, 10) for i in range(N)]

    from timeit import default_timer as timer
    t0 = timer()
    b = a*vector
    t1 = timer()
    c = np.dot(a.INCOMING, vector)
    t2 = timer()
    print(np.allclose(b, c))
    print("matvec took {}s and numpy {}s".format(t1-t0, t2-t1))

    pass
