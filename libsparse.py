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
        self.CSR = {'AVAL': [], 'JCOL': [], 'IROW': []}  # This might change as it might be replaced by a dedicated CSR object...
        self._choose_scheme()

    def __repr__(self):
        return '<sparse matrix of shape {} and sparsity {:.2f}>'.format(self.INCOMING.shape, self.sparsity)

    def construct_CSR(self):
        # NEEDSDOC
        '''
        Author: Simon Glennemeier-Marke

        Constructs a CSR form of a given array.
        Args:
        > 'INCOMING' :  sparse numpy array

        Returns:
        > self.CSR :  dict containing the CSR object
        '''
        for j, col in enumerate(self.INCOMING):
            for i, el in enumerate(col):
                if el != 0:
                    self.CSR['AVAL'].append(el)
                    self.CSR['JCOL'].append(i)
                continue
            self.CSR['IROW'].append(len(self.CSR['AVAL']))
        pass

    def _choose_scheme(self):
        # "_method" means python won't import this method with wildcard import "from lib import * "
        '''
        Author: Simon Glennemeier-Marke

        Decide which storage scheme to use based on matrix density.
        '''
        if self.sparsity > .5:
            self.construct_CSR()
        elif self.sparsity > 0:
            raise NotImplementedError
        else:
            raise ValueError('Sparisty should be in interval (0,1), but is {:.3f}'.format(self.sparsity))

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
