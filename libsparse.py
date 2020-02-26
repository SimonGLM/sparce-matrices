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

    def construct_CSR(self, array):
        # TODO NEEDSDOC
        '''
        Author: Simon Glennemeier-Marke

        Constructs a CSR form of a given array.

        Args:
        > 'INCOMING' :  sparse numpy array

        Returns:
        > self.CSR :  dict containing the CSR object
        '''
        csr = {'AVAL': [], 'JCOL': [], 'IROW': []}
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


def random_banded(size, num_diags):
    # TODO NEEDSDOC
    '''Author: Simon Glennemeier-Marke'''
    mat = scipy.sparse.diags([rng.uniform(0, 1, size=size) for i in range(num_diags)], range((-num_diags+1)//2, (num_diags+1)//2), shape=(size, size)).toarray()
    return scipy.sparse.eye(size)+(mat+np.transpose(mat))/2


if __name__ == "__main__":
    # TESTING
    a = sparse(random_banded(50, 5))
    print(a)
    plt.matshow(a.INCOMING)
    plt.colorbar()
    plt.show()
    pass
