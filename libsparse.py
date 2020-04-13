'''
Sparse
======
This library implements sparse matrices.

`import libsparse as sp`


Sparse matrices are matrices which have relatively few non-zero entries.

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
import scipy.linalg
import scipy.sparse.linalg
import scipy.sparse
import scipy
import numpy as np
rng = np.random.default_rng()
np.set_printoptions(edgeitems=8, linewidth=180)


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

    def __init__(self, array):
        temp = array if (type(array) == np.ndarray) else np.array(array)
        self.sparsity = 1 - np.count_nonzero(temp)/temp.size
        self.shape = temp.shape
        self.T = self.transpose
        self._choose_scheme(temp)
        del temp
        self.N = self.shape[0] if quadratic(self) else None

    def __repr__(self):
        return '<sparse matrix of shape {} and sparsity {:.2f}>'.format(self.shape, self.sparsity)

    def __mul__(self, other):
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
            slice_ = slice(self.CSR['IROW'][i], self.CSR['IROW'][i+1])
            if j in self.CSR['JCOL'][slice_]:
                j_index = self.CSR['IROW'][i]+self.CSR['JCOL'][slice_].index(j)
                return self.CSR['AVAL'][j_index]
            else:
                return 0
        if i != None and j == None:
            # Retrun row at `i`
            return [self[i, k] for k in range(self.shape[1])]
        if i == None and j != None:
            # Return col at `JCOL`
            return [self[k, j] for k in range(self.shape[0])]

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
        >>> a.toarray()
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
        for _, col in enumerate(array):
            for i, el in enumerate(col):
                if el != 0:
                    csr['AVAL'].append(el)
                    csr['JCOL'].append(i)
                continue
            csr['IROW'].append(len(csr['AVAL']))

        return csr

    def construct_CSR_fast(self, array):
        '''
        Author: Simon Glennemeier-Marke

        Faster version of construct_CSR.

        Regular implementation is O(n^2), where as this is O(n).

        This is achieved by only iterating over the rows and filling it all at once.
        In addition, we now use numpy methods which are a lot faster.

        Args:
        > `array` :  sparse numpy array

        Returns:
        > self.CSR :  dict containing the CSR object
        '''
        array: np.ndarray
        jcol = np.array([], dtype=np.int32)
        aval = np.array([], dtype=np.float)
        irow = np.array([0], dtype=np.int32)
        for row in array:
            row: np.ndarray
            indices = np.nonzero(row)[0]
            jcol = np.append(jcol, indices)
            aval = np.append(aval, np.take(row, indices))
            irow = np.append(irow, len(aval))
        csr = {'AVAL': list(aval), 'JCOL': list(jcol), 'IROW': list(irow)}
        return csr

    def toarray(self):
        '''
        Author: Simon Glennemeier-Marke

        Converts sparse object to numpy array using scipy sparse methods

        Returns:
        --------
        > `np.ndarray` : Fullsize array of self
        '''
        return scipy.sparse.csr_matrix((self.CSR['AVAL'], self.CSR['JCOL'], self.CSR['IROW'])).toarray()

    def transpose(self):
        '''
        Author: Simon Glennemeier-Marke

        Return the transposed version of self.

        Returns:
        --------
        > `sp.sparse` : Transposed sparse object of self
        '''
        return sparse(np.transpose(self.toarray()))

    def check_posdef(self):
        '''
        Author: Henrik Spielvogel

        Checks if matrix is positive definite.


        Returns:
        --------
        > `bool` : True if self is positive definite
        '''
        evals = scipy.sparse.linalg.eigs(self.toarray())
        return np.alltrue(evals[0] > 0)

    def _choose_scheme(self, array):
        '''
        Author: Simon Glennemeier-Marke

        Decide which storage scheme to use based on matrix density.

        Args:
        > `array` : np.ndarray
        '''
        if 1 > self.sparsity >= .5:
            # self.CSR = self.construct_CSR(array)
            self.CSR = self.construct_CSR_fast(array)
        elif .5 > self.sparsity >= 0:
            print('NotImplementedError: falling back to implemented methods')
            # self.CSR = self.construct_CSR(array)
            self.CSR = self.construct_CSR_fast(array)
        else:
            raise ValueError('Sparsity should be in half-open interval [0,1), but is {:.3f}'.format(self.sparsity))

        pass

    def _vdot(self, vec: np.ndarray):
        '''
        Author: Henrik Spielvogel

        Calculates the matrix-vector product of a sparse matrix with `vec`.

        Args:
        > `vec` :  list or array of same length as matrix

        Returns:
        > outvec :  np.ndarray

        '''

        vec = vec if type(vec) == np.ndarray else np.array(vec)
        n = len(vec)
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

        Compute the dot product of a matrix and either another matrix or a vector

        If other is a vector we call `self._vdot(other)`.
        This is to enhance comability with numpy methods of matrix multiplication.

        Operator overloading:
        ---------------------
        >>> A * B

        Returns:
        --------
        > <class 'sparse'> of multiplied matrices
        > or
        > <class 'numpy.ndarray'> in case of a multiplication with a vector
        '''
        if type(other) != sparse and len(other.shape) == 1:  # check for vector
            return self._vdot(other)
        if type(other) == np.ndarray:  # check for ndarray
            return self.toarray()@other

        if self.shape[1] != other.shape[0]:
            raise AttributeError(
                'Shapes do not match {}, {}'.format(self.shape, other.shape))

        result = np.zeros((self.shape[0], other.shape[1]))
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                row = self[i, None]
                col = other[None, j]
                result[i, j] = sum([r*c for r, c in zip(row, col)])
        return sparse(result)

    def show(self):
        '''
        Author: Simon Glennemeier-Marke
        '''
        data = self.toarray()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data, interpolation='nearest')
        fig.colorbar(cax)
        plt.show()


def lu_factor(array):
    '''
    Author: Simon Glennemeier-Marke

    Arguments:
    ----------
    > `array` : Input array to be LU factorized

    Returns:
    --------
    > `P` : Permutation matrix
    > `L` : Lower triangular
    > `U` : Upper triangular
    '''
    if not quadratic(array):
        raise AssertionError('LU decomposition is not possible for non-quadratic matrices.')
    N = array.shape[0]
    P = np.eye(N)
    L = np.eye(N)
    U = np.zeros(array.shape)
    for i in range(N):
        for k in range(i, N):
            val = 0
            for j in range(i):
                val += L[i, j] * U[j, k]
            U[i, k] = array[i, k] - val
        for k in range(i, N):
            val = 0
            for j in range(i):
                val += L[k, j] * U[j, i]
            if U[i, i] == 0:
                raise ZeroDivisionError
            L[k, i] = (array[k, i] - val)/U[i, i]
    return P, L, U


def quadratic(array):
    '''Author: Simon Glennemeier-Marke'''
    return bool(array.shape[0] == array.shape[1])


def random_banded(size, num_diags):
    '''
    Author: Simon Glennemeier-Marke

    Create symmetric banded matrix of dimension `size` with `num_diags` diagonals.

    Returns:
    --------
    > np.ndarray
    '''
    mat = scipy.sparse.diags([rng.uniform(0, 1, size=size) for i in range(num_diags)],
                             range((-num_diags+1)//2, (num_diags+1)//2), shape=(size, size)).toarray()
    return np.array(scipy.sparse.eye(size)+(mat+np.transpose(mat))/2)


class linsys(object):
    '''
    Author: Henrik Spielvogel

    A linear system of the form Ax=b
    ================================

    This object creates linear systems of equations of the form Ax=b.
    It implements different methods for solving these systems considering the sparsity of the given matrix A.

    Arguments:
    ----------
    > `A` : sp.sparse or np.ndarray
    > `b` : 1D-list or np.ndarray
    '''

    def __init__(self, A, b):
        if type(A) not in [sparse, np.ndarray]:
            raise TypeError('Matrix A has to be of type sp.sparse or np.ndarray')
        self.matrix = A
        self.target_vector = b if (type(b) == np.ndarray) else np.array(b)
        self.shape = A.shape
        self.N = A.shape[0]

    def __repr__(self):
        return '<linsys of dim: {} >'.format(self.N)

    def lu_solve(self):
        '''
        Author: Henrik Spielvogel

        Solving a dense linear system using LU-Decomposition without pivoting

        Returns:
        --------
        > `sol`: np.ndarray solution vector x of the linear system Ax=b
        '''

        mat = self.matrix
        vec = self.target_vector
        y = np.zeros_like(vec)
        sol = y

        # LU-Decomposition
        P, L, U = lu_factor(mat)

        # forward substitution
        for i in range(self.N):
            y[i] = (vec[i] - y.dot(L[i]))/L[i, i]
        # back substitution
        for i in range(self.N, 0, -1):
            sol[i-1] = (y[i-1] - U[i-1, i:].dot(sol[i:])) / U[i-1, i-1]

        return sol

    def cg_solve(self, init_guess=None, TOL=1e-8):
        '''
        Author: Henrik Spielvogel

        Solving a dense linear system using the conjugate-gradient-method

        Returns:
        --------
        > `sol`: np.ndarray solution vector x of the linear system Ax=b
        '''
        n = self.N
        mat = self.matrix
        vec = self.target_vector

        if init_guess is None:
            x = np.ones(n)
        elif type(init_guess) == list and len(init_guess) == n:
            x = np.array(init_guess)
        elif type(init_guess) == np.ndarray and init_guess.shape[0] == n:
            x = init_guess
        else:
            raise ValueError('init_guess must be list or np.ndarray of length {}.'.format(n))

        r = mat.dot(x) - vec
        p = -r
        r_norm = r.dot(r)

        for i in range(2*n):
            z = mat.dot(p)
            alpha = r_norm / p.dot(z)
            x += alpha * p
            r += alpha * z
            r_norm_next = r.dot(r)
            beta = r_norm_next / r_norm
            r_norm = r_norm_next

            if r_norm_next < TOL:
                print('CG-Method converged after {} iterations.'.format(i))
                break
            p = beta * p - r

        sol = x

        return sol

    def solve(self, sparse=False, method='scipy'):
        mat = self.matrix
        vec = self.target_vector

        implemented = ['scipy', 'lu', 'cg']
        if method not in implemented:
            raise NotImplementedError('Method `{}` unknown. Implemented methods are {}'.format(method, implemented))

        if sparse:
            # needs routine for solving sparse systems
            raise NotImplementedError
        else:
            if method == 'scipy':
                sol = scipy.linalg.solve(mat, vec)
            elif method == 'lu':
                sol = self.lu_solve()
            elif method == 'cg':
                sol = self.cg_solve()
        return sol


if __name__ == "__main__":
    rng: np.random.Generator  # hint for IntelliSense
    N = 10
    a = np.random.randint(-3, 3, (N, N))
    a = np.eye(N)+((a+np.transpose(a))/2)

    mat1 = A = np.array([[-3,  1, -1,  0, -1],
                         [0,  1,  0,  1,  0],
                         [-1, -1, -3, -1,  0],
                         [0,  1, -1,  2,  0],
                         [0,  1, -1,  1, -1]], dtype=np.float_)
    vec1 = np.array([-1, -2, 5, -2, -2], dtype=np.float_)

    mat2 = random_banded(N, N//10)
    vec2 = np.random.rand(N)

    from timeit import default_timer as timer
    sys = linsys(mat2, vec2)

    start1 = timer()
    sol1 = sys.solve(method='scipy')
    t1 = timer()-start1

    start2 = timer()
    sol2 = sys.solve(method='lu')
    t2 = timer() - start2

    start3 = timer()
    sol3 = sys.cg_solve(init_guess=np.random.rand(sys.N-1))
    t3 = timer() - start3

    print('Scipy took {:1.5f} seconds'.format(t1))
    print('LU-Decomp took {:1.5f} seconds'.format(t2))
    print('Conjugate-Gradient-Method took {:1.5f} seconds'.format(t3))

    print('scipy == LU: {}'.format(np.allclose(sol1, sol2)))
    print('scipy == CG: {}'.format(np.allclose(sol1, sol3, atol=1e-4)))

    pass


'''
TODO Tasks:
  > CG-Method for sparse
  > Gaussian elimination

'''
