from libsparse import *

if __name__ == "__main__":
    # Do nothing...
    a = np.random.randint(0, 5, (3, 3))
    b = np.random.randint(0, 5, (3, 3))
    spa = CSR(a)
    spb = CSR(b)
    print(spa.toarray())
    print(spb.toarray())
    print(type(spa*spb))
    pass
