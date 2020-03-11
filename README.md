# sparse-matrices
This is the repository for an end term assignment for the 'Scientific Programming' course at the Justus-Liebig-University, Giessen.

[Cheat sheet](https://www.markdownguide.org/cheat-sheet/#basic-syntax)



# Optimiziation
Let's take a look at how we can accalerate the construction of our CSR arrays. The following 
<details>
    <summary>Output of timing measurements</summary>

    ```python
    $ python -m line_profiler libsparse.py.lprof
    Timer unit: 1e-06 s

    Total time: 148.352 s
    File: libsparse.py
    Function: construct_CSR at line 168

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       168                                           
       169                                               def construct_CSR(self, array):
       [...]
       181         1          3.0      3.0      0.0          csr = {'AVAL': [], 'JCOL': [], 'IROW': [0]}
       182     10001      18618.0      1.9      0.0          for j, col in enumerate(array):
       183 100010000   65355397.0      0.7     44.1              for i, el in enumerate(col):
       184 100000000   82901563.0      0.8     55.9                  if el != 0:
       185     29998      26134.0      0.9      0.0                      csr['AVAL'].append(el)
       186     29998      16687.0      0.6      0.0                      csr['JCOL'].append(i)
       187     29998      13125.0      0.4      0.0                  continue
       188     10000      20265.0      2.0      0.0              csr['IROW'].append(len(csr['AVAL']))
       189                                           
       190         1          1.0      1.0      0.0          return csr

    Total time: 1.50618 s
    File: libsparse.py
    Function: construct_CSR_fast at line 192

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       192                                               def construct_CSR_fast(self, array):
       [...]
       209                                                   array: np.ndarray
       210                                                   jcol = np.array([])
       211         1         13.0     13.0      0.0          aval = np.array([])
       212         1          2.0      2.0      0.0          irow = np.array([0])
       213         1          4.0      4.0      0.0          for row in array:
       214     10001      12535.0      1.3      0.8              row: np.ndarray
       215                                                       indices = np.nonzero(row)[0]
       216     10000     770523.0     77.1     51.2              jcol = np.append(jcol, indices)
       217     10000     231029.0     23.1     15.3              aval = np.append(aval, np.take(row, indices))
       218     10000     287930.0     28.8     19.1              irow = np.append(irow, len(aval))
       219     10000     201464.0     20.1     13.4          csr = {'AVAL': list(aval), 'JCOL': list(jcol),     'IROW': list(irow)}
       220         1       2684.0   2684.0      0.2          return csr
    ```
</details>