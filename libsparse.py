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
> Creates, displays and returns a sparse `size`Ãƒâ€”`size` matrix which is banded.
> I.e. it only has non-zero values on num_diags centered diagonals.


This library is part of a project done as an end-term assignment in the 'Scientific Programming'-course at the Justus-Liebig-University in Giessen, Germany.
'''

import matplotlib.pyplot as plt
import scipy.sparse
import scipy
import numpy as np
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
        self.ARRAY = array if (type(array) == np.ndarray) else np.asfarray(array)
        self.sparsity = 1 - np.count_nonzero(self.ARRAY)/self.ARRAY.size
        self.shape = self.ARRAY.shape
        self.T = lambda: sparse(self.ARRAY.T)
        self._choose_scheme(self.ARRAY)

    def __repr__(self):
        return '<sparse matrix of shape {} and sparsity {:.2f}>'.format(self.shape, self.sparsity)

    def __mul__(self, other):
        if type(other) != sparse:
            # Matrix-Vector product
            return self.vdot(other)
        else:
            # Matrix-Matrix product
            return self.dot(other)

    def construct_CSR(self, array):
        '''
        Author: Simon Glennemeier-Marke

        Constructs a CSR form of a given array.

        Args:
        > `ARRAY` :  sparse numpy array

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

    def _choose_scheme(self, array: np.ndarray):
        # "_method" means python won't import this method with wildcard import "from lib import * "
        '''
        Author: Simon Glennemeier-Marke

        Decide which storage scheme to use based on matrix density.

        Args:
        > `array` : np.ndarray
        '''
        if 1 > self.sparsity >= .5:
            self.CSR = self.construct_CSR(array)
        elif .5 > self.sparsity >= 0:
            print('NotImplementedError: falling back to implemented methods')
            self.CSR = self.construct_CSR(array)
        else:
            raise ValueError('Sparisty should be in half-open interval [0,1), but is {:.3f}'.format(self.sparsity))

        pass

    # TODO: Needs class methods for gaussian elimination etc...

    def vdot(self, vec: np.ndarray):
        '''
        Author: Henrik Spielvogel

        Calculates the matrix-vector product of a sparse matrix with `vec`.

        Args:
        > `vec` :  list or array of same length as matrix

        Returns:
        > outvec :  np.ndarray

        '''

        n = self.shape[0]
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

    def dot(self, other):
        '''
        Author: Simon Glennemeier-Marke

        To avoid confusion about commutativity,
        use:

        >>> A * B

        `dot` computes the right sided dot product
        > `A.dot(B)` = " `A` * `B` "

        Returns:
        --------
        > <class sparse> of multiplied matrices
        '''
        if type(other) != sparse:
            raise TypeError("Argument has to be {}, not {}".format(type(self), type(other)))
        if self.shape[1] != other.shape[0]:
            raise AttributeError(
                'Shapes do not match {},{}'.format(self.shape, other.shape))
        result = np.zeros((self.shape[0], other.shape[1]))
        for i in range(1, self.shape[0]+1):
            for j in range(1, other.shape[1]+1):
                row = lookup(self, i=i)
                col = lookup(other, j=j)
                result[i-1, j-1] = sum([r*c for r, c in zip(row, col)])
        return sparse(result)


def lookup(array: sparse, i=None, j=None):
    '''
    Author: Simon Glennemeier-Marke

    General utility function for sparse arrays.
    If `lookup` is called with only one of `i` or `j`, it will call recursively get
    all elements of one row/column with index `i`/`j` and return a it as a list.
    See examples.

    Arguments:
    ----------
    > `array`: Input array of <class sparse>
    > `i`: Index of row
    > `j`: Index of column

    Examples
    --------
     >>> rng = np.random.default_rng()
     >>> a = sparse(rng.integers(0,4,(3,4)))
     >>> a.ARRAY
     array([[1, 2, 2, 0],
            [0, 0, 1, 2],
            [3, 0, 3, 0]])

     >>> lookup(a,i=1,j=2)
     2

     >>> lookup(a,i=1)
     [1, 0, 3]

     >>> lookup(a,j=3)
     [3, 0, 3, 0]"
    '''

    if i != None and j != None:
        if i == 0 or j == 0:
            raise IndexError('Indices count from 1')
        i -= 1
        j -= 1
        array: sparse
        slice_ = slice(array.CSR['IROW'][i], array.CSR['IROW'][i+1])
        if j in array.CSR['JCOL'][slice_]:
            j_index = array.CSR['IROW'][i]+array.CSR['JCOL'][slice_].index(j)
            return array.CSR['AVAL'][j_index]
        else:
            return 0
    if i != None and j == None:
        # Retrun row at `i`
        return [lookup(array, i, k) for k in range(1, array.shape[1]+1)]
    if i == None and j != None:
        # Return col at `JCOL`
        return [lookup(array, k, j) for k in range(1, array.shape[0]+1)]


def random_banded(size, num_diags):
    # TODO NEEDSDOC
    '''
    Author: Simon Glennemeier-Marke

    Create quadratic banded sparse matrix of dimension 'size' with 'num_diags' diagonals

    '''
    mat = scipy.sparse.diags([rng.uniform(0, 1, size=size) for i in range(num_diags)],
                             range((-num_diags+1)//2, (num_diags+1)//2), shape=(size, size)).toarray()
    return scipy.sparse.eye(size)+(mat+np.transpose(mat))/2


if __name__ == "__main__":
    rng: np.random.Generator  # hint for IntelliSense
    N = 10
    # a = sparse(np.eye(N))
    # a = sparse(random_banded(N, 2))
    a = sparse(rng.integers(-10, 10, (N, N-3)))
    # b = sparse(np.eye(N))
    # b = sparse(random_banded(N, 2))
    b = sparse(rng.integers(-5, 5, (N-3, N)))
    csp = a*b
    cnp = np.dot(a.ARRAY, b.ARRAY)
    print(np.allclose(csp.ARRAY, cnp))

    # vector = rng.integers(-10, 10, N)
    # from timeit import default_timer as timer
    # t0 = timer()
    # b = a*vector
    # t1 = timer()
    # c = np.dot(a.ARRAY, vector)
    # t2 = timer()
    # print(np.allclose(b, c))
    # print("matvec took {}s and numpy {}s".format(t1-t0, t2-t1))

    pass
