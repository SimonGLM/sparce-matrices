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
> Creates, displays and returns a sparse `size`x`size` matrix which is banded.
> I.e. it only has non-zero values on num_diags centered diagonals.


This library is part of a project done as an end-term assignment in the 'Scientific Programming' course at the Justus-Liebig-University in Giessen, Germany.
'''

import matplotlib.pyplot as plt
import scipy.sparse
import scipy
import numpy as np
rng = np.random.default_rng()
np.set_printoptions(edgeitems=8,linewidth=180)

class sparse(object):
    # TODO NEEDSDOC
    '''
    Author: Simon Glennemeier-Marke & Henrik Spielvogel

    A sparse array object
    =====================

    This object is the general sparse matrix object.
    It implements the 'Compressed Sparse Row' format, storing only non-zero values.

    Sparse objects can be subscripted to ask for an element, a column vector and a row vector.

    Subscripting:
    -------------
    >>> a = sparse([[1,2],[3,4]])
    >>> a[1,1]
    1
    >>> a[None,1]
    [1.0, 3.0]
    >>> a[1,None]
    [1.0, 2.0]

    Arguments:
    > array : np.ndarray of arbitrary size
    '''

    def __init__(self, array: np.ndarray):
        temp = array if (type(array) == np.ndarray) else np.asfarray(array)
        self.sparsity = 1 - np.count_nonzero(temp)/temp.size
        self.shape = temp.shape
        self.quadratic = bool(self.shape[0] == self.shape[1])
        self.T = lambda: sparse(self.ARRAY.T)
        self._choose_scheme(temp)
        del temp
        self.N = self.shape[0] if self.quadratic else None

    def __repr__(self):
        return '<sparse matrix of shape {} and sparsity {:.2f}>'.format(self.shape, self.sparsity)

    def __mul__(self, other):
        if type(other) != sparse:
            # Matrix-Vector product
            return self.vdot(other)
        else:
            # Matrix-Matrix product
            return self.dot(other)

    def __getitem__(self, key):
        '''
        Author: Simon Glennemeier-Marke

        `__getitem__` is called using subscripting.

        If either the first or second index is `None`, we return the whole vector of that index.

        Examples:
        ---------
        >>> a = sp.sparse([[1,2],[3,4]])
        >>> a[1,1]
        1.0
        >>> a[2,1]
        3.0
        >>> a[None,2]
        [2.0, 4.0]
        '''
        if not all([type(key[i]) == int or key[i] == None for i, el in enumerate(key)]):
            raise TypeError('Argument has to be type int or None')
        if len(key) > 2:
            raise IndexError('Index out of range.')
        i, j = key
        if i != None and j != None:
            if i == 0 or j == 0:
                raise IndexError('Indices count from 1.')
            i -= 1
            j -= 1
            slice_ = slice(self.CSR['IROW'][i], self.CSR['IROW'][i+1])
            if j in self.CSR['JCOL'][slice_]:
                j_index = self.CSR['IROW'][i]+self.CSR['JCOL'][slice_].index(j)
                return self.CSR['AVAL'][j_index]
            else:
                return 0
        if i != None and j == None:
            # Retrun row at `i`
            return [self[i, k] for k in range(1, self.shape[1]+1)]
        if i == None and j != None:
            # Return col at `JCOL`
            return [self[k, j] for k in range(1, self.shape[0]+1)]

    def __setitem__(self, key, value):
        '''
        Author: Simon Glennemeier-Marke

        `__setitem__`  is called using subscripting and assignment

        Arguments:
        ----------
        > `key` : tuple, Array indices where to set the new value (row major order)

        > `value` : int or float, Value to assign the specified element to

        Examples:
        ---------
        >>> a = sparse([[1,2],[3,4]])
        >>> a[1,1] = 6
        >>> a.ARRAY
        array([[6.0, 2.0]
               [3.0, 4.0]])
        '''
        if type(value) != int and type(value) != float:
            raise TypeError('Value is of type {}, but needs to be int or float.'.format(type(value)))
        if type(value) == int:
            value = float(value)
        if len(key) != 2:
            raise IndexError('Index has to be tuple.')
        i, j = key
        if i == 0 or j == 0:
            raise IndexError('Indices count from 1.')
        i -= 1
        j -= 1
        slice_ = slice(self.CSR['IROW'][i], self.CSR['IROW'][i+1])
        if j in self.CSR['JCOL'][slice_]:  # Value exists, just needs to be overwritten
            index = self.CSR['IROW'][i]+self.CSR['JCOL'][slice_].index(j)
            self.CSR['AVAL'][index] = value
        else:  # Value doesn't exist, needs to be inserted into CSR
            new_index = self.CSR['IROW'][i]+j
            self.CSR['AVAL'].insert(new_index, value)
            self.CSR['JCOL'].insert(new_index, j)
            for k in range(i+1, len(self.CSR['IROW'])):
                self.CSR['IROW'][k] += 1

    def construct_CSR(self, array):
        '''
        Author: Simon Glennemeier-Marke

        Constructs a CSR form of a given array.

        Args:
        > `array` :  sparse numpy array

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

    def toarray(self):
        array = np.zeros(self.shape)
        for i, row in enumerate(array):
            for j, el in enumerate(row):
                array[i][j] = self[i+1, j+1] 
        return array

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
                'Shapes do not match {}, {}'.format(self.shape, other.shape))
        result = np.zeros((self.shape[0], other.shape[1]))
        for i in range(1, self.shape[0]+1):
            for j in range(1, other.shape[1]+1):
                row = self[i, None]
                col = other[None, j]
                result[i-1, j-1] = sum([r*c for r, c in zip(row, col)])
        return sparse(result)

    def LU_decomp(self):
        if not self.quadratic:
            raise AssertionError('LU decomposition is not possible for non-quadratic matrices.')
        import copy
        L = sparse(np.eye(self.N))
        U = copy.deepcopy(self)
        for i in range(1, self.N):
            for k in range(i+1, N):
                L[k, i] = float(U[k, i] / U[i, i])
                for j in range(i, N+1):
                    U[k, j] = float(U[k, j]-L[k, i]*U[i, j])
        return L, U


def random_banded(size, num_diags):
    # TODO NEEDSDOC
    '''
    Author: Simon Glennemeier-Marke

    Create symmetric banded matrix of dimension `size` with `num_diags` diagonals.

    '''
    mat = scipy.sparse.diags([rng.uniform(0, 1, size=size) for i in range(num_diags)],
                             range((-num_diags+1)//2, (num_diags+1)//2), shape=(size, size)).toarray()
    return scipy.sparse.eye(size)+(mat+np.transpose(mat))/2


if __name__ == "__main__":
    rng: np.random.Generator  # hint for IntelliSense
    N = 10
    # a = sparse(np.eye(N))
    # a = sparse(random_banded(N, 2))
    a = sparse(rng.integers(-10, 10, (N, N)))
    # a = scipy.sparse.rand(50, 50, 0.2)

    # b = sparse(np.eye(N))
    # b = sparse(random_banded(N, 2))
    # b = sparse(rng.integers(-5, 5, (N-3, N)))
    # b = scipy.sparse.rand(50, 50, 0.2)
    # csp = a*b
    # cnp = np.dot(a.toarray(), b.toarray())
    # print(np.allclose(csp.toarray(), cnp))
    L, U = a.LU_decomp()

    pass
