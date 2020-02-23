class sparce(object):
    def __init__(self,ndsparce):
        # NEEDSDOC
        '''Author: Simon Glennemeier-Marke & Henrik Spielvogel'''
        self.INCOMING=ndsparce
        self.ParseToCSR(self.INCOMING)
        self.CSR={'AVAL':[],'JCOL':[],'IROW':[]} # This might change as it might be replaced by a dedicated CSR object...
    
    def ParseToCSR(self,INCOMING):
        # NEEDSDOC
        '''
        Author: Simon Glennemeier-Marke
        
        Constructs a CSR form of a given array.
        Args:
        > 'INCOMING' :  sparce numpy array
        
        Returns:
        > self.CSR :  dict containing the CSR object
        '''
        # TODO: Construct CSR in here
        
        pass

    # TODO: Needs class methods for gaussian elimination
        
if __name__ == "__main__":
    pass