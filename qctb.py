"""
Module for QCTB evaluation
"""

import pytest
import numpy as np
import scipy.linalg

B_benzene=np.array([
    [1, 1, 0],
    [0, 1 ,1],
    [1, 0 ,1]
])

B_styrene=np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
])

E_Huckel_benzene = 8
E_Huckel_styrene = 10.42

n_unpaired_electrons_benzene_ref = 0.025
n_unpaired_electrons_styrene_ref = 0.069

def get_eigenValues_and_Functions_sorted_by_reverse_eigenValues(M):
    eigV, eigF = scipy.linalg.eigh(M)
    idx = eigV.argsort()[::-1]
    eigV = eigV[idx]
    eigF = eigF[:,idx]
    return (eigV, eigF)

class qctb():
    """
    QCTB class contatins all info about the molecule.
    At contruction, all calculations are done and stored in the dict data.
    """
    def __init__(self, B):
        """
        QCB Constructor with the B matrix. This matrix indicates the connection
        between starred and unstarred atoms in the molecule.
        """

        DELTA = 7/24

        H = np.block([[np.zeros(B.shape), B],[B.T, np.zeros(B.shape)]])
        eigValH, eigFuncH = get_eigenValues_and_Functions_sorted_by_reverse_eigenValues(H)

#        H_alpha = np.block([[  DELTA * np.identity(len(B[0])), B],[B.T, -DELTA * np.identity(len(B[0]))]])
#        H_beta  = np.block([[ -DELTA * np.identity(len(B[0])), B],[B.T,  DELTA * np.identity(len(B[0]))]])

        P_U = np.power(DELTA, 4) * np.linalg.matrix_power(
            np.linalg.matrix_power(H, 2) + np.power(DELTA, 2) * np.identity(len(H[0])),
            -2
        )
        eigValP_U, eigFuncP_U = scipy.linalg.eigh(P_U)

        self.data={
            "B": B,
            "H": H,
            "eigValH": eigValH,
            "eigFuncH": eigFuncH,
            "P_U": P_U,
            "eigValP_U": eigValP_U,
            "eigFuncP_U": eigFuncP_U,
        }
        return

    def get_Huckel_eigenValues(self):
        return self.data["eigValH"]

    def get_Huckel_eigenFunctions(self):
        return self.data["eigFuncH"]

    def get_Density_of_Unpaired_electrons(self):
        return self.data["P_U"]

def test_001_Huckel_benzene_Total_Energy():
    """
    Test the Huckel total energy of benzene
    """

    thrs = 0.1

    QCTB_benzene = qctb(B_benzene)
    nOcc=len(B_benzene[0])
    E_tot = sum([ 2 * e for e in QCTB_benzene.get_Huckel_eigenValues()[0:nOcc] ])
    assert np.abs(E_tot - E_Huckel_benzene) < thrs

def test_002_Huckel_styrene_Total_Energy():
    """
    Test the Huckel total energy versus a value computed with hulis online for the styrene molecule
    """

    thrs = 0.1

    QCTB_styrene = qctb(B_styrene)
    nOcc=len(B_styrene[0])
    E_tot = sum([ 2 * e for e in QCTB_styrene.get_Huckel_eigenValues()[0:nOcc] ])
    assert np.abs(E_tot - E_Huckel_styrene) < thrs

def test_003_QCTB_benzene_N_unpaired_electrons():
    """
    Test the QCTB number of unparied electrons for the benzene molecule
    """

    thrs = 0.01

    QCTB_benzene = qctb(B_benzene)
    P_U = QCTB_benzene.get_Density_of_Unpaired_electrons()

    assert np.abs(np.trace(P_U) - n_unpaired_electrons_benzene_ref) < thrs

def test_004_QCTB_styrene_N_unpaired_electrons():
    """
    Test the QCTB number of unparied electrons for the styrene molecule
    """

    thrs = 0.01

    QCTB_styrene = qctb(B_styrene)
    P_U = QCTB_styrene.get_Density_of_Unpaired_electrons()

    assert np.abs(np.trace(P_U) - n_unpaired_electrons_styrene_ref) < thrs

def main():
    print("Running tests")
    pytest.main()
    return

if __name__ == "__main__":
    main()
