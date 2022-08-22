import numpy as np
import random as rand

cpdef chooseActions(double [:,:] probas):
    cdef int i,j
    cdef int rows,columns
    cdef float cumul
    #print(probas.shape)
    result = np.empty((probas.shape[0],1), dtype = np.dtype('i'))
    cdef int[:,:] actions = result
    cdef double r = rand.random()
    rows = probas.shape[0]
    columns = probas.shape[1]
    for i in range(rows):
        cumul = 0
        for j in range(columns):
            cumul+=probas[i,j]
            if cumul>r:
                break
        actions[i,0]=j
    return result
