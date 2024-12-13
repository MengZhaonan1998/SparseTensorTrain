import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from pyscf import gto, scf

from utils import TensorSparseStat
from tensortrain_id import TT_IDPRRLDU
from tensortrain_svd import TT_SVD

def view_tensor_2d(eri_tensor):
    # View the eri tensor in 2D way
    shape = eri_tensor.shape
    eri2d = tl.reshape(eri_tensor, [shape[0] * shape[1], shape[2] * shape[3]])
    plt.imshow(eri2d, interpolation='none')
    plt.show()

def disp_eritt_info(eri_tensor, eri_tt, remark = None):
    # Display information of the eri tensor and the corresponding tensor train
    print(f"=== INFO DISPLAY ({remark})===")
    
    print("--- Sparsity of the eri tensor ---")
    TensorSparseStat([eri_tensor])
    
    recon_eri = tl.tt_to_tensor(eri_tt)
    diff = np.abs(recon_eri - eri_tensor)
    error = np.max(diff)
    print("--- Reconstruction error ---")
    print(f"Absolute max error: {error}")
    print(f"Relative max error: {error / np.max(np.abs(eri_tensor))}")
    
    print("--- Sparsity of the eri tensor train ---")
    TensorSparseStat(eri_tt)    
    return

def eri_test1():
    # Electron repulsion integral (eri) tensor test 1
    # Define the molecule
    mol = gto.Mole()
    mol.atom = 'Li 0 0 0; H 1.5 0 0'
    mol.basis = 'sto-3g'  # Very minimal basis
    mol.build()
    
    # Compute full ERI tensor
    eri_tensor = mol.intor('int2e')
    
    # Print some information about the ERI tensor
    print("ERI Tensor Shape:", eri_tensor.shape)
    print("ERI Tensor Data Type:", eri_tensor.dtype)
    # The indices represent (i,j|k,l) in physicist's notation
    print("Example ERI element (0,0|0,0):", eri_tensor[0,0,0,0])
    
    # TT-ID based on PRRLDU
    maxdim = 30
    cutoff = 1e-6
    eri_tt_id = TT_IDPRRLDU(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_id, "TT-IDPRRLDU")
    
    # TT-SVD
    cutoff = 1e-6
    eri_tt_svd = TT_SVD(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_svd, "TT-SVD")
    return

def eri_test2():
    # Electron repulsion integral (eri) tensor test 2
    # Define the molecule
    mol = gto.Mole()
    mol.atom = '''
    O  0.0  0.0  0.0
    H  0.7  0.0  0.0
    H  0.0  0.7  0.0
    '''
    mol.basis = 'cc-pvdz'  # Choose a basis set
    mol.build()

    # Create a Hartree-Fock object
    mf = scf.RHF(mol)

    # Compute the ERI tensor
    eri_tensor = mol.intor('int2e')

    # Print some information about the ERI tensor
    print("ERI Tensor Shape:", eri_tensor.shape)
    print("ERI Tensor Data Type:", eri_tensor.dtype)
    # The indices represent (i,j|k,l) in physicist's notation
    print("Example ERI element (0,0|0,0):", eri_tensor[0,0,0,0])

    # TT-ID based on PRRLDU
    maxdim = 300
    cutoff = 1e-8
    eri_tt_id = TT_IDPRRLDU(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_id, "TT-IDPRRLDU")
    
    # TT-SVD
    cutoff = 1e-7
    eri_tt_svd = TT_SVD(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_svd, "TT-SVD")
    return

eri_test1()
#eri_test2()