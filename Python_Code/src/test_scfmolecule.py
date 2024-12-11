from pyscf import gto, ao2mo
import numpy as np
import h5py
def view(h5file, dataname='eri_mo'):
    with h5py.File(h5file) as f5:
        print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
mo1 = np.random.random((mol.nao_nr(), 10))
mo2 = np.random.random((mol.nao_nr(), 8))
mo3 = np.random.random((mol.nao_nr(), 6))
mo4 = np.random.random((mol.nao_nr(), 4))
eri1 = ao2mo.kernel(mol, mo1)
pass