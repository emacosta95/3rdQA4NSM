from NSMFermions.hamiltonian_utils import FermiHubbardHamiltonian
from NSMFermions.nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState,J2operator
import numpy as np
import torch
from typing import Dict
import scipy
from NSMFermions.qml_models import AdaptVQEFermiHubbard
from NSMFermions.qml_utils.train import Fit
from NSMFermions.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
from NSMFermions.utils_quasiparticle_approximation import QuasiParticlesConverterOnlynnpp,QuasiParticlesConverter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

file_name='data/usdb.nat'

SPS=SingleParticleState(file_name=file_name)
twobody_matrix,_=get_twobody_nuclearshell_model(file_name=file_name)
print(SPS.energies.shape)


nparts=[(0,2),(0,4),(0,6),(0,8),(0,10),(2,2),(2,4),(2,6),(2,8),(2,10),(4,4),(4,6),(4,8),(4,10),(6,6),(6,8),(6,10),(8,8),(8,10),(10,10)]
titles=[r'$^{18}$O',r'$^{20}$O',r'$^{22}$O',r'$^{24}$O',r'$^{26}$O',r'$^{20}$Ne',r'$^{22}$Ne',r'$^{24}$Ne',r'$^{26}$Ne',r'$^{28}$Ne',r'$^{24}$Mg',r'$^{26}$Mg',r'$^{28}$Mg',r'$^{30}$Mg',r'$^{28}$Si',r'$^{30}$Si',r'$^{32}$Si',r'$^{32}$S',r'$^{34}$S',r'$^{36}$Ar']

size_a=SPS.energies.shape[0]//2
size_b=SPS.energies.shape[0]//2

title=r'$^{18}$O'

# Compute the J^2 value
#J2Class=J2operator(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,single_particle_states=SPS.state_encoding,j_square_filename=file_name+'_j2',symmetries=[SPS.total_M_zero])

#Quadrupole Operator
energy_errors=[]
abs_energy_errors=[]
fidelities=[]
for idx,npart in enumerate(nparts):
    nparticles_a,nparticles_b=npart
    title=titles[idx]
    
    # compute the NSM Hamiltonian
    NSMHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
    print('size=',size_a+size_b,size_b)
    NSMHamiltonian.get_external_potential(external_potential=SPS.energies[:size_a+size_b])
    print(NSMHamiltonian.external_potential.shape)
    NSMHamiltonian.get_twobody_interaction_optimized(twobody_dict=twobody_matrix)
    NSMHamiltonian.get_hamiltonian()

    gpu_hamiltonian=(NSMHamiltonian.hamiltonian)
    egs,psi0=eigsh(gpu_hamiltonian,k=1,which='SA')

    print(egs)
    QPC=QuasiParticlesConverter()

    QPC.initialize_shell(state_encoding=SPS.state_encoding)


    #just for the basis
    QPC.get_the_basis_matrix_transformation(basis=NSMHamiltonian.basis)
    
    hamiltonian_qq=QPC.particles2quasiparticles @ NSMHamiltonian.hamiltonian @ QPC.particles2quasiparticles.T
    
    gpu_hamiltonian_qq=(hamiltonian_qq)
    egs_qq,psi_q=eigsh(gpu_hamiltonian_qq,k=1,which='SA')
    
    psi_q=QPC.particles2quasiparticles.T @ psi_q[:,0]
    delta=np.abs((egs_qq[0]-egs[0])/egs[0])
    energy_errors.append(delta)
    abs_energy_errors.append(np.abs(egs_qq[0]-egs[0]))
    
    fidelity=(psi_q[:].dot(psi0[:,0])).conjugate()*(psi_q[:].dot(psi0[:,0]))
    fidelities.append(1-fidelity.item())
    
    
fig, ax = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns

### ---------- First plot: Δ_Q E ---------- ###
values = energy_errors

# Get unique proton and neutron numbers
protons = sorted(set(p for n, p in nparts))
neutrons = sorted(set(n for n, p in nparts))

protons=[0,2,4,6]
neutrons=[0,2,4,6,8,10]

p_idx = {p: i for i, p in enumerate(protons)}
n_idx = {n: i for i, n in enumerate(neutrons)}

# Create and fill matrix [protons, neutrons]
matrix = np.full((len(protons), len(neutrons)), np.nan)
for (n, p), val in zip(nparts, values):
    matrix[p_idx[p], n_idx[n]] = val

# Plot heatmap
cax = ax[0].imshow(matrix, cmap='viridis', origin='lower')

# Axes
ax[0].set_xticks(np.arange(len(neutrons)))
ax[0].set_yticks(np.arange(len(protons)))
ax[0].set_xticklabels(neutrons, fontsize=14)
ax[0].set_yticklabels(protons, fontsize=14)
ax[0].set_xlabel(r"$N_n$", fontsize=30)
ax[0].set_ylabel(r"$Z_p$", fontsize=30)
ax[0].set_title(r'$\Delta_Q E$', fontsize=30)

# Overlay values and titles
for (n, p), title, val in zip(nparts, titles, values):
    i = p_idx[p]
    j = n_idx[n]
    if not np.isnan(val):
        ax[0].text(j, i - 0.2, f"{title}", ha='center', va='center', color='white', fontsize=20, weight='bold')
        ax[0].text(j, i + 0.3, f"{val:.2f}", ha='center', va='center', color='white', fontsize=20)

# Colorbar
# cbar = fig.colorbar(cax, ax=ax[0])
# cbar.set_label(r'$\Delta_Q E$', fontsize=20)
# cbar.ax.tick_params(labelsize=16)


### ---------- Second plot: 1 - F_Q ---------- ###
values = fidelities

# Create and fill matrix [protons, neutrons]
matrix = np.full((len(protons), len(neutrons)), np.nan)
for (n, p), val in zip(nparts, values):
    matrix[p_idx[p], n_idx[n]] = val

# Plot heatmap
cax = ax[1].imshow(matrix, cmap='viridis', origin='lower')

# Axes
ax[1].set_xticks(np.arange(len(neutrons)))
ax[1].set_yticks(np.arange(len(protons)))
ax[1].set_xticklabels(neutrons, fontsize=20)
ax[1].set_yticklabels(protons, fontsize=20)
ax[1].set_xlabel(r"$N_n$", fontsize=30)
ax[1].set_ylabel(r"$Z_p$", fontsize=30)
ax[1].set_title(r'$1-F_Q$', fontsize=30)

# Overlay values and titles
for (n, p), title, val in zip(nparts, titles, values):
    i = p_idx[p]
    j = n_idx[n]
    if not np.isnan(val):
        ax[1].text(j, i - 0.2, f"{title}", ha='center', va='center', color='white', fontsize=20, weight='bold')
        ax[1].text(j, i + 0.3, f"{val:.2f}", ha='center', va='center', color='white', fontsize=20)

# # Colorbar
# cbar = fig.colorbar(cax, ax=ax[1])
# cbar.set_label(r'$1-F_Q$', fontsize=20)
# cbar.ax.tick_params(labelsize=16)

# Colorbar for Δ_Q E
# cbar = fig.colorbar(cax, ax=ax[0], fraction=0.046, pad=0.04)
# cbar.set_label(r'$\Delta_Q E$', fontsize=20)
# cbar.ax.tick_params(labelsize=16)

# ...
# Colorbar for 1 - F_Q
# cbar = fig.colorbar(cax,  fraction=0.046, pad=0.04)
# #cbar.set_label(r'$1-F_Q$', fontsize=20)
# cbar.ax.tick_params(labelsize=16)
cbar = fig.colorbar(fig, ax=[ax[0], ax[1]], location="left", fraction=0.046, pad=0.08)
cbar.set_label(r'Color scale', fontsize=20)
cbar.ax.tick_params(labelsize=16)
# Final touches
ax[0].tick_params(labelsize=16)
ax[1].tick_params(labelsize=16)
plt.tight_layout()
plt.show()