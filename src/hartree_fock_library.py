import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import trange, tqdm
import numpy as np
from scipy import sparse, linalg
from math import factorial, sqrt
import matplotlib.pyplot as plt



def slater_determinants_combined(C_n, C_p, fock_basis):
    """
    This function computes the Hartree-Fock Slater determinant
    C_n: [single particle dimension per neutrons, number of neutrons]  -- neutron orbitals
    C_p: [single particle dimension per protons, number of protons]  -- proton orbitals
    fock_basis: [dimension of the fock space, single particle dimension]  -- full occupation basis (neutrons + protons)

    Returns:
        psi: [F]  -- Slater determinant amplitudes
    """
    F, M = fock_basis.shape
    M_half = M // 2 # half for neutrons and half for protons
    N_n = C_n.shape[1] # number of neutrons
    N_p = C_p.shape[1] # number of protons

    psi = torch.zeros(F, dtype=C_n.dtype) # initialize psi

    for i in range(F):
        occ = fock_basis[i]  # [M]

        occ_n = torch.nonzero(occ[:M_half]).squeeze() # neutron occupations
        occ_p = torch.nonzero(occ[M_half:]).squeeze()+M_half # proton occupations

        Cn_sub = C_n[occ_n, :]  # shape [N_n, N_n] compute the submatrix for the minors
        Cp_sub = C_p[occ_p, :]  # shape [N_p, N_p]

        if Cn_sub.shape[0] != N_n or Cp_sub.shape[0] != N_p:
            # Skip invalid configurations (e.g., wrong number of particles)
            continue

        # compute determinants for the minors as the compontent of the Hartree Fock wavefunction
        det_n = torch.det(Cn_sub) 
        det_p = torch.det(Cp_sub)
        psi[i] = det_n * det_p


    return psi  # [F]

def slater_determinants_only_neutrons(C_n, fock_basis):
    """
    This functions does the same as slater_determinants_combined but only for neutrons
    C_n: [M_half, N_n]  -- neutron orbitals
    fock_basis: [F, M]  -- full occupation basis (neutrons + protons)

    Returns:
        psi: [F]  -- Slater determinant amplitudes
    """
    F, M = fock_basis.shape
    M_half = M // 2
    N_n = C_n.shape[1]


    psi = torch.zeros(F, dtype=C_n.dtype)

    for i in range(F):
        occ = fock_basis[i]  # basis element, slater determinant
        occ_n = torch.nonzero(occ[:M_half]).squeeze() # neutron occupations
        Cn_sub = C_n[occ_n, :]  # shape [N_n, N_n] get the minor
        det_n = torch.det(Cn_sub)
        psi[i] = det_n 

    return psi  # [F]



def gram_schmidt(V):
    """
    Perform Gram-Schmidt orthogonalization on the set of vectors V.
    
    Parameters:
        V (numpy.ndarray): A 2D numpy array where each column is a vector.
        
    Returns:
        numpy.ndarray: A 2D numpy array where each column is an orthonormal vector.
    """
    # Number of vectors
    num_vectors = V.shape[1]
    # Dimension of each vector
    dim = V.shape[0]
    
    # Initialize an empty array for the orthogonal vectors
    Q = np.zeros((dim, num_vectors))
    
    for i in range(num_vectors):
        # Start with the original vector
        q = V[:, i]
        
        # Subtract the projection of q onto each of the previously calculated orthogonal vectors
        for j in range(i):
            q = q - np.dot(Q[:, j], V[:, i]) * Q[:, j]
        
        # Normalize the resulting vector
        Q[:, i] = q / np.linalg.norm(q)
    
    return Q

class HFEnergyFunctionalNuclear(nn.Module):
    def __init__(self, h_vec, V_dict, num_neutrons, num_protons, neutron_indices, proton_indices,m_values:np.ndarray=None,multiplier_m_values:Optional[float]=0.):
        """
        Initializes the Hartree-Fock energy functional for a nuclear system with neutrons and protons.
        
        :param h_vec: single particle energies.
        :param V_dict: two body interaction dictionary indexed by (a,b,c,d) -> value.
        :param num_neutrons: number of valence neutrons.
        :param num_protons: number of valence protons.
        :param neutron_indices: index of the initial neutron orbital.
        :param proton_indices: index of the initial proton orbital.
        """
        super().__init__()
        self.h = h_vec  # [M]
        self.M = h_vec.shape[0]
        self.Nn = num_neutrons
        self.Np = num_protons

        self.proton_idx = proton_indices
        # neutron orbitals go from 0 to proton_idx-1
        # proton orbitals go from proton_idx to M-1
        if num_protons!=0:
            self.V_tensor = torch.zeros((self.M, self.M, self.M, self.M), dtype=h_vec.dtype)
            for (a, b, c, d), val in V_dict.items():
                self.V_tensor[a, b, c, d] = val
        else:
            self.V_tensor = torch.zeros((self.M//2, self.M//2, self.M//2, self.M//2), dtype=h_vec.dtype)
            for (a, b, c, d), val in V_dict.items():
                if  a<self.M//2 and b<self.M//2 and c<self.M//2  and d<self.M//2: 
                    self.V_tensor[a, b, c, d] = val
        self.A_n = nn.Parameter(torch.randn(self.proton_idx, self.Nn,dtype=h_vec.dtype))
        # generate A_p only if there are protons
        if num_protons!=0:
            self.A_p = nn.Parameter(torch.randn(self.proton_idx, self.Np,dtype=h_vec.dtype))


        # constrain for M=0
        if m_values is not None:
            self.m_tensor=torch.tensor(m_values,dtype=h_vec.dtype)
            self.multiplier_m_values=multiplier_m_values
        else:
            self.m_tensor=None
    def forward(self):
        # get local orthonormal orbitals for neutrons and protons
        C_n_local, _ = torch.linalg.qr(self.A_n)
        # generate C_p only if there are protons
        if self.Np!=0:
            C_p_local, _ = torch.linalg.qr(self.A_p)
            C_n = torch.zeros((self.M, self.Nn), dtype=C_n_local.dtype, device=C_n_local.device)
        else:
             C_n_local=C_n_local[ :self.M//2]
             C_n = torch.zeros((self.M//2, self.Nn), dtype=C_n_local.dtype, device=C_n_local.device)
        if self.Np!=0:
            C_p = torch.zeros((self.M, self.Np), dtype=C_p_local.dtype, device=C_p_local.device)
        
        # local to full space embedding
        C_n[:self.proton_idx, :] = C_n_local
        # generate rho matrices if there are protons
        if self.Np!=0:
            C_p[self.proton_idx:, :] = C_p_local
            self.rho_p = C_p @ C_p.T
        self.rho_n = C_n @ C_n.T
        # compute energy in case of protons
        if self.Np!=0:
            
            E1 = torch.dot(self.h, torch.diagonal(self.rho_n + self.rho_p))
            E2 = (
            0.5 * torch.einsum('abcd,ca,db->', self.V_tensor, self.rho_n, self.rho_n) +
            0.5 * torch.einsum('abcd,ca,db->', self.V_tensor, self.rho_p, self.rho_p) +
            torch.einsum('abcd,ca,db->', self.V_tensor, self.rho_n, self.rho_p)
            )
            self.C_p=C_p.clone()
            
            if self.m_tensor is not None:
                E_constrain=self.multiplier_m_values*(torch.sum(self.rho_n*self.m_tensor)**2+torch.sum(self.rho_p*self.m_tensor)**2)
            else:
                E_constrain=0.
        # compute energy in case of only neutrons
        else:
            E1 = torch.dot(self.h[:self.M//2], torch.diagonal(self.rho_n ))
            E2 = (
            0.5 * torch.einsum('abcd,ca,db->', self.V_tensor, self.rho_n, self.rho_n) 
            )
            if self.m_tensor is not None:
                E_constrain=self.multiplier_m_values*(torch.sum(self.rho_n*self.m_tensor)**2)
            else:
                E_constrain=0.
            
        self.C_n=C_n.clone()        

        print(f"E1: {E1.item()}, E2: {E2.item()}, E_constrain: {E_constrain.item()}")
        return E1 + E2 + E_constrain


    def build_mixed_fock_matrix(self):
        """
        Builds a single Fock matrix including mixed neutron-proton contributions.
        Assumes self.rho_n and self.rho_p are full MxM density matrices.
        """
        M = self.rho_n.shape[0]
        h_mat = torch.diag(self.h[:M])

        # Single Fock including both species
        F = h_mat.clone()
        total_rho = self.rho_n.clone()
        if self.rho_p is not None:
            total_rho += self.rho_p

        # Contract two-body term
        F += torch.einsum("acbd,dc->ab", self.V_tensor, total_rho)

        # Hermitian symmetrization
        F = 0.5 * (F + F.T)

        return F

    def build_fock_matrices_factorized(self):
        """
        Builds neutron and proton Fock matrices including cross-species contributions.
        Assumes self.rho_n and self.rho_p are embedded in full MxM space.
        """
        M = self.rho_n.shape[0]
        h_mat = torch.diag(self.h[:M])

        # --- Neutron Fock ---
        F_n = h_mat.clone()
        # nn term
        F_n += 0.5 * torch.einsum("acbd,dc->ab", self.V_tensor, self.rho_n)
        # np cross term
        if self.rho_p is not None:
            F_n +=  0.5*(torch.einsum("acbd,dc->ab", self.V_tensor, self.rho_p)+torch.einsum("acbd,ac->bd", self.V_tensor, self.rho_p))
        # Hermitian symmetrization
        F_n = 0.5 * (F_n + F_n.T)

        # --- Proton Fock ---
        F_p = None
        if self.rho_p is not None:
            F_p = h_mat.clone()
            # pp term
            F_p += 0.5 * torch.einsum("acbd,dc->ab", self.V_tensor, self.rho_p)
            # pn cross term
            F_p +=0.5*( torch.einsum("acbd,dc->ab", self.V_tensor, self.rho_n)+torch.einsum("acbd,ac->bd", self.V_tensor, self.rho_n))
            F_p = 0.5 * (F_p + F_p.T)

        # Return blocks corresponding to neutrons/protons if needed
        return F_n[:M//2, :M//2], F_p[M//2:, M//2:] if F_p is not None else None


    # -------------------------
    # Transform integrals with unitary U_can
    # -------------------------
    def transform_integrals_using_fock_rotation(self,U, h, V):
        """
        Transforms one-body and two-body integrals using a unitary transformation U that describes the fock rotation.
        
        :param U: The fock rotation unitary (MxM) as a numpy matrix
        :param h: single particle energy matrix (MxM) as a numpy matrix
        :param V: two-body interaction tensor (MxMxMxM) as a numpy array
        :return: transformed one-body Hamiltonian h_p (MxM), two-body Hamiltonian V_p (MxMxMxM)
        """
        # U: full MxM unitary that maps new_alpha <- old_b  (i.e. a^dag_alpha = sum_b U[alpha,b] c^dag_b)
        # transforms: h' = U @ h @ U^H
        h_p = U @ h @ U.conj().T
        # two-step contraction for V'_{pqrs} = sum_{abcd} U_{p a} U_{q b} V_{abcd} U*_{r c} U*_{s d}
        # do in numpy with einsum (not memory optimized but fine for small M)
        V_p = np.einsum("pa,qb,abcd,rc,sd->pqrs", U, U, V, U.conj(), U.conj())
        return h_p, V_p




def build_fock_matrix(h_mat, V_tensor, rho):
    """
    Computes the Fock matrix given one-body Hamiltonian, two-body interaction tensor, and density matrix.
    
    :param h_mat: single particle energy matrix (M,M) as a numpy matrix
    :param V_tensor: two-body interaction tensor (M,M,M,M) as a numpy array
    :param rho: density matrix (M,M) as a numpy matrix
    :return: Fock matrix (M,M) as a numpy matrix
    """
    # F_ab = h_ab + sum_cd V_acbd * rho_dc
    # here V_tensor shape (M,M,M,M), rho shape (M,M)
    F = h_mat.copy()
    M = h_mat.shape[0]
    # einsum way (numpy)
    F += 0.5*np.einsum("acbd,dc->ab", V_tensor, rho)
    # ensure Hermitian
    F = 0.5 * (F + F.T.conj())
    return F