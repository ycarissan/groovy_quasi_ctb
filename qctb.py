"""
Module for QCTB evaluation
"""

import pytest
import numpy as np
import scipy.linalg

B_styrene=np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
])
E_Huckel_styrene = 10.42

def get_eigenValues_and_Functions_sorted_by_reverse_eigenValues(M):
    eigV, eigF = scipy.linalg.eigh(M)
    idx = eigV.argsort()[::-1]
    eigV = eigV[idx]
    eigF = eigF[:,idx]
    return (eigV, eigF)

class qctb():
    def __init__(self, B):
        H = np.block([[np.zeros(B.shape), B],[B.T, np.zeros(B.shape)]])
        eigValH, eigFuncH = get_eigenValues_and_Functions_sorted_by_reverse_eigenValues(H)
        self.data={
            "B": B,
            "H": H,
            "eigValH": eigValH,
            "eigFuncH": eigFuncH,
        }
        return

    def get_Huckel_eigenValues(self):
        return self.data["eigValH"]

    def get_Huckel_eigenFunctions(self):
        return self.data["eigFuncH"]

def test_000_Huckel_styrene_Total_Energy():
    thrs = 0.1
    QCTB_styrene = qctb(B_styrene)
    nOcc=len(B_styrene[0])
    E_tot = sum([ 2 * e for e in QCTB_styrene.get_Huckel_eigenValues()[0:nOcc] ])
    assert np.abs(E_tot - E_Huckel_styrene) < thrs

def main():
    test_styrene = qctb(B_styrene)
    return

if __name__ == "__main__":
    main()
