import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from pyscf import gto, scf

from utils import TensorSparseStat
from tensorly.contrib.decomposition import tensor_train_cross
from tensortrain_id import TT_IDPRRLDU
from tensortrain_svd import TT_SVD

def view_tensor_2d(eri_tensor):
    # View the eri tensor in 2D way
    # Reshape the 4D tensor to 2D for visualization
    eri_2d = np.log10(np.abs(eri_tensor.reshape(eri_tensor.shape[0]*eri_tensor.shape[1], eri_tensor.shape[2]*eri_tensor.shape[3]) + 1e-20))

    plt.figure(figsize=(10,8))
    plt.imshow(eri_2d, cmap='viridis', aspect='auto')
    plt.title('ERI Tensor Sparsity Visualization (Log Scale)')
    plt.colorbar(label='Log10(Absolute Integral Value)')
    plt.tight_layout()
    plt.show()
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

def d2s_data_output(eri_tensor):
    # Take a dense-format eri tensor as input and output the sparse format
    
    
    
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
    
    # TT-Cross
    rank = [1 for i in range(5)]
    for i in range(3):
        rank[i+1] = eri_tt_svd[i].shape[2]
    eri_tt_cross = tensor_train_cross(eri_tensor, rank, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_cross, "TT-CROSS")
    return

def eri_test2():
    # Electron repulsion integral (eri) tensor test 2
    # Sodium chloride molecule with a slightly larger, but still minimal basis
    mol = gto.Mole()
    mol.atom = '''
    Na 0.0 0.0 0.0
    Cl 2.4 0.0 0.0
    '''
    mol.basis = 'sto-3g'  # Minimal basis set
    mol.charge = 0
    mol.spin = 0    
    mol.build()
    
    # Compute full ERI tensor
    eri_tensor = mol.intor('int2e')
    
    # Print some information about the ERI tensor
    print("ERI Tensor Shape:", eri_tensor.shape)
    print("ERI Tensor Data Type:", eri_tensor.dtype)
    # The indices represent (i,j|k,l) in physicist's notation
    print("Example ERI element (0,0|0,0):", eri_tensor[0,0,0,0])
    
    # TT-ID based on PRRLDU
    maxdim = 100
    cutoff = 1e-6
    eri_tt_id = TT_IDPRRLDU(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_id, "TT-IDPRRLDU")
    
    # TT-SVD
    cutoff = 1e-6
    eri_tt_svd = TT_SVD(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_svd, "TT-SVD")
    
    # TT-Cross
    rank = [1 for i in range(5)]
    for i in range(3):
        rank[i+1] = eri_tt_svd[i].shape[2]
    eri_tt_cross = tensor_train_cross(eri_tensor, rank, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_cross, "TT-CROSS")
    return

def eri_test3():
    # Electron repulsion integral (eri) tensor test 3
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
    cutoff = 1e-6
    eri_tt_id = TT_IDPRRLDU(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_id, "TT-IDPRRLDU")
    
    # TT-SVD
    cutoff = 1e-6
    eri_tt_svd = TT_SVD(eri_tensor, maxdim, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_svd, "TT-SVD")
    
    # TT-Cross
    rank = [1 for i in range(5)]
    for i in range(3):
        rank[i+1] = eri_tt_svd[i].shape[2]
    eri_tt_cross = tensor_train_cross(eri_tensor, rank, cutoff)
    disp_eritt_info(eri_tensor, eri_tt_cross, "TT-CROSS")
    return

#eri_test1()
#eri_test2()
eri_test3()