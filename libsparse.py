import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
rng = np.random.default_rng()


class sparse(object):
    # NEEDSDOC
    '''
    Author: Simon Glennemeier-Marke & Henrik Spielvogel

    A sparse array object
    =====================

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
        # NEEDSDOC
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
        '''
        if 1 > self.sparsity >= .5:
            self.CSR = self.construct_CSR(array)
        elif .5 > self.sparsity > 0:
            raise NotImplementedError
        else:
            raise ValueError('Sparisty should be in open interval (0,1), but is {:.3f}'.format(self.sparsity))

        pass

    # TODO: Needs class methods for gaussian elimination etc...


if __name__ == "__main__":
    # TESTING
    size = 64
    array = scipy.sparse.block_diag([rng.uniform(0, 3, size=(3, 3))for i in range(10)]).toarray()
    a = sparse(array)
    print(a)
    plt.matshow(a.INCOMING)
    plt.show()
    pass
