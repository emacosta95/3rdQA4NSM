#### Imports

from NSMFermions.hamiltonian_utils import FermiHubbardHamiltonian
from NSMFermions.nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState
import numpy as np
import torch
from typing import Dict
import scipy
from NSMFermions.qml_models import AdaptVQEFermiHubbard
from NSMFermions.qml_utils.train import Fit
from NSMFermions.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
from NSMFermions.utils_quasiparticle_approximation import QuasiParticlesConverter,HardcoreBosonsBasis,QuasiParticlesConverterOnlynnpp
from src.brillouin_wigner_utils import full_brillouin_wigner_method,computation_pauli_blockade,full_brillouin_wigner_method_pauliblockade,brillouin_wigner_method_hf_ansatz
from scipy.sparse import lil_matrix
from NSMFermions.utils_quasiparticle_approximation import HardcoreBosonsBasis
from scipy.sparse.linalg import norm
import random
from src.hartree_fock_library import HFEnergyFunctionalNuclear,build_fock_matrix
import torch.optim as optim
from scipy.sparse import csr_matrix,identity
import pickle


#### hyperapameters

file_name='data/usdb.nat' #select the file with the single particle energies and two-body matrix elements
# initialize the class that handles single particle states
SPS=SingleParticleState(file_name=file_name)
# load the twobody matrix as a dictionary
twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)
# get the dimension of each single particle basis
size_a=SPS.energies.shape[0]//2
size_b=SPS.energies.shape[0]//2

# set the number of particles
nparticles=[(2,2),(4,2),(6,2),(8,2),(10,2),(4,4),(4,8),(4,10),(6,10),(8,8),(8,10),(10,10)]
iteration_list=[100]*6+[180]*6

### the reference values
energies_exact=[]
energies_qq=[]

# full bw method
accuracies_fbw=[]
states_fbw=[]
hamiltonians_fbw=[]
energies_fbw=[]

# truncated bw method
energies_truncated_bw=[]
fidelities_truncated_bw=[]
operator_fidelities_bw=[]


# hf bw method
energies_hf_bw=[]
fidelities_hf_bw=[]
accuracies_hfbw=[]
operator_fidelities_hfbw=[]


for index_run,nparts in enumerate(nparticles):
    
    nsteps_iteration=iteration_list[index_run]
    # get the number of neutron and protons
    nparticles_a=nparts[0]
    nparticles_b=nparts[1]
    
    # initialize the nuclear shell model hamiltonian
    NSMHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
    # set the single particle energy part as an external potential
    NSMHamiltonian.get_external_potential(external_potential=SPS.energies[:size_a+size_b])
    # set the two-body interaction
    NSMHamiltonian.get_twobody_interaction_optimized(twobody_matrix)
    # compute the matrix representation of the hamiltonian
    NSMHamiltonian.get_hamiltonian()
    # compute eigenvectors and eigenvalues in the active configuration interaction
    eigvals_aci,eigvecs_aci=NSMHamiltonian.get_spectrum(n_states=1)

    print(eigvals_aci)
    # save the exact energy
    energies_exact.append(eigvals_aci[0])

    print(NSMHamiltonian.hamiltonian.shape)

    ######## get the quasiparticle basis
    
    # Initialize the quasiparticle class
    QPC=QuasiParticlesConverterOnlynnpp()
    # intialize the quasiparticle modes
    QPC.initialize_shell(state_encoding=SPS.state_encoding)


    # compute the quasiparticle basis and the matrix transformation QP -> ACI
    QPC.get_the_basis_matrix_transformation(basis=NSMHamiltonian.basis)
    # check the basis reduction
    
    #### get the terms for the brilloui-wigner method
    hamiltonian_qq=QPC.particles2quasiparticles @ NSMHamiltonian.hamiltonian @ QPC.particles2quasiparticles.T
    hamiltonian_rr=QPC.particles2restofstates @ NSMHamiltonian.hamiltonian @ QPC.particles2restofstates.T
    hamiltonian_qr=QPC.particles2quasiparticles @ NSMHamiltonian.hamiltonian @ QPC.particles2restofstates.T
    hamiltonian_rq=QPC.particles2restofstates @ NSMHamiltonian.hamiltonian @ QPC.particles2quasiparticles.T
        
    e,_=eigsh(hamiltonian_qq,k=1,which='SA')
    # save energies_qq
    energies_qq.append(e[0])
    
    #### compute the full brillouin-wigner method 
    psiq_fbw,energy_fbw,hamiltonian_fbw,list_infedelities_fbw,list_errors_fbw,_=full_brillouin_wigner_method([hamiltonian_qq,hamiltonian_qr,hamiltonian_rq,hamiltonian_rr],threshold=10**-3,eigvals_aci=eigvals_aci,nsteps_iteration=nsteps_iteration)
    states_fbw.append(psiq_fbw)
    hamiltonians_fbw.append(hamiltonian_fbw)
    energies_fbw.append(energy_fbw)
    accuracies_fbw.append(list_errors_fbw)
    #### compute the truncated brillouin-wigner method
    
    #### first of all we get the hamiltonian in the 2+2 sector
    # initialize the nuclear shell model hamiltonian
    Hamiltonian2bodysector=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=2,nparticles_b=2,symmetries=[SPS.total_M_zero])
    # set the single particle energy part as an external potential
    Hamiltonian2bodysector.get_external_potential(external_potential=SPS.energies[:size_a+size_b])
    # set the two-body interaction
    Hamiltonian2bodysector.get_twobody_interaction_optimized(twobody_matrix)
    # compute the matrix representation of the hamiltonian
    Hamiltonian2bodysector.get_hamiltonian()


    
    #### then, we get the quasiparticle subspace
    # Initialize the quasiparticle class
    QPC2body=QuasiParticlesConverterOnlynnpp()
    # intialize the quasiparticle modes
    QPC2body.initialize_shell(state_encoding=SPS.state_encoding)


    # compute the quasiparticle basis and the matrix transformation QP -> ACI
    QPC2body.get_the_basis_matrix_transformation(basis=Hamiltonian2bodysector.basis)
    # check the basis reduction

    # hamiltonians in the 2b sectors    
    hamiltonian_qq_2b=QPC2body.particles2quasiparticles @ Hamiltonian2bodysector.hamiltonian @ QPC2body.particles2quasiparticles.T
    hamiltonian_qr_2b=QPC2body.particles2quasiparticles @ Hamiltonian2bodysector.hamiltonian @ QPC2body.particles2restofstates.T
    hamiltonian_rq_2b=QPC2body.particles2restofstates @ Hamiltonian2bodysector.hamiltonian @ QPC2body.particles2quasiparticles.T
    hamiltonian_rr_2b=QPC2body.particles2restofstates @ Hamiltonian2bodysector.hamiltonian @ QPC2body.particles2restofstates.T

    # get the quantities for the pauli blockade, at the zeroth order we consider n_i and rho^(2)_00 of H_q^(0)
    _,psi0_qq=eigsh(hamiltonian_qq,k=1,which='SA')
    psi0_qq=psi0_qq[:,0]
    #pauli_blockade_density_matrix,sp_density=computation_pauli_blockade(basis=NSMHamiltonian.basis,QPC=QPC,psi_qq=psi0_qq)
    # get the sp density as the prob of having a clean transition
    sp_density=np.zeros(size_a+size_b)
    # for neutron
    sp_density[:size_a]=(nparticles_a-1)/(size_a)
    # and proton
    sp_density[size_a:]=(nparticles_b-1)/(size_b)
    
    #### we compute the 2 qbody H_RQ taking into account the pauli blocking.
    # we initialize it creating the lil_matrix
    hamiltonian_qr_2b_pauliblockade=lil_matrix((QPC2body.quasiparticle_basis.shape[0],QPC2body.rest_basis.shape[0]))

    for index_qbasis,qb in enumerate(QPC2body.quasiparticle_basis):
        # we extract the quasiparticle indices of each basis element
        quasiparticle_indices=[]
        idxs=np.nonzero(qb)[0]
        for idx in idxs:
            quasiparticle_indices.append(QPC2body.couples[idx][0])
            quasiparticle_indices.append(QPC2body.couples[idx][1])
        for index_restbasis,rb in enumerate(QPC2body.rest_basis):
            # we do this only for the non-zero element of H_QR
            if hamiltonian_qr_2b[index_qbasis,index_restbasis]!=0.:
                # extract the indices of the R basis element
                rest_indices=np.nonzero(rb)[0]
                # we get the final indices of the transition (if there is one)
                added_indices = sorted(set(rest_indices) - set(quasiparticle_indices))  # added
            
                if len(added_indices)==1:
                    # if the transition is effective single body you multiply the coefficient by a 1-n_a, the prob of having that state 0
                    constrain=np.prod(1-sp_density[added_indices])
                
                if len(added_indices)==2:
                    # if the transition is two-body, you multiply the coefficient with a \rho_00_ab, the prob of having both states 0
                    constrain=np.prod(1-sp_density[added_indices])#pauli_blockade_density_matrix[added_indices[0],added_indices[1]]
                # we mask the H_QR by using this coefficient
                hamiltonian_qr_2b_pauliblockade[index_qbasis,index_restbasis]=constrain*hamiltonian_qr_2b[index_qbasis,index_restbasis]
                #hamiltonian_rq_2b_pauliblockade[index_restbasis,index_qbasis]=np.prod(pauli_blocking_vector_removed)*hamiltonian_rq_2b[index_restbasis,index_qbasis]
    # H_RQ is just the transverse of H_QR
    hamiltonian_rq_2b_pauliblockade=hamiltonian_qr_2b_pauliblockade.T
    
    # the pauli blockade is also embodied in the H_RR
    hamiltonian_rr_masked=hamiltonian_rr_2b.copy()

    for index_restbasis_a,rba in enumerate(QPC2body.rest_basis):
        # we get the indices of the first rest basis
        rest_indices_a=np.nonzero(rba)[0]
        for index_restbasis_b,rbb in enumerate(QPC2body.rest_basis):
            # we consider only non-zero coefficients
            if hamiltonian_rr_2b[index_restbasis_a,index_restbasis_b]!=0 and index_restbasis_b!=index_restbasis_a:                
                rest_indices_b=np.nonzero(rbb)[0]
                # we consider only the final indices of the transition
                added_indices = sorted(set(rest_indices_b) - set(rest_indices_a))  # added
                # single transition
                if len(added_indices)==1:
                    constrain=np.prod(1-sp_density[added_indices])
                # two-body transition                
                if len(added_indices)==2:
                    constrain=np.prod(1-sp_density[added_indices]) #pauli_blockade_density_matrix[added_indices[0],added_indices[1]]
                
                # mask the H_RR hamiltonian
                hamiltonian_rr_masked[index_restbasis_a,index_restbasis_b]=constrain*hamiltonian_rr_2b[index_restbasis_a,index_restbasis_b].copy()
                
    #### compute the truncated brillouin-wigner method 
    _,_,_,_,_,delta_hamiltonian=full_brillouin_wigner_method_pauliblockade([hamiltonian_qq_2b,hamiltonian_qr_2b_pauliblockade,hamiltonian_qr_2b,hamiltonian_rq_2b_pauliblockade,hamiltonian_rr_2b,hamiltonian_rr_masked],threshold=10**-3,eigvals_aci=eigvals_aci,nsteps_iteration=nsteps_iteration)
    
    # extract the two body transitions of the \Delta H_QQ
    twobody_quasiparticle_effective_interaction={}
    # we get the indices of the 2 body quasiparticle states
    for q,base_q in enumerate(QPC2body.quasiparticle_basis):
        for p,base_p in enumerate(QPC2body.quasiparticle_basis):
            a_q,b_q=np.nonzero(base_q)[0]
            a_p,b_p=np.nonzero(base_p)[0]
            q_max=np.max([a_q,b_q])
            q_min=np.min([a_q,b_q])
            p_max=np.max([a_p,b_p])
            p_min=np.min([a_p,b_p])
            twobody_quasiparticle_effective_interaction[(a_q,b_q,a_p,b_p)]=delta_hamiltonian[q,p]

    #### we compute the effective interaction by using the Hardcore boson space

    # we introduce a class to build up quasiparticle operators on the quasiparticle basis
    # it works as the FermiHubbardHamiltonian class, the only difference is the basis of the single site operator that corresponds with the S+ S- format, instead of c^dag c
    HBB=HardcoreBosonsBasis(QPC.quasiparticle_basis)

    # we compute the Delta H using the dictionary in the new basis
    
    tbar=tqdm(enumerate(twobody_quasiparticle_effective_interaction.keys()))
    delta_hamiltonian_qq=0.
    for _,key in tbar:        
        a,b,c,d=key
        delta_hamiltonian_qq+=HBB.adag_adag_a_a_matrix_optimized(a,b,c,d)*twobody_quasiparticle_effective_interaction[key]
    
    # then we compute the energy corrections
    tot_hamiltonian_qq_truncated=hamiltonian_qq+delta_hamiltonian_qq
    values,psiq_truncated=eigsh(tot_hamiltonian_qq_truncated,k=1)
    e=values[0]
    psiq_truncated=psiq_truncated[:,0]
    print(e)
    print('Energy error of the truncated BW method=',np.abs((e-eigvals_aci[0])/eigvals_aci[0]),'\n')
    
    fidelity=np.abs(np.vdot(psiq_fbw,psiq_truncated))**2
    fidelities_truncated_bw.append(fidelity)
    energies_truncated_bw.append(e)
    distance_hf=norm(hamiltonian_fbw- tot_hamiltonian_qq_truncated,ord='fro')/norm(hamiltonian_fbw,ord='fro')
    operator_fidelities_bw.append(distance_hf)
    
    #### In the last part of the analysis we explore the HF method
    # first, we get the HF for the NSM system to extract the energy


    
    
    # fix the seed for the initialization of the HF coefficients
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # compute the m_values for the symmetry restirction of the HF method
    # we compute the m values array
    m_values=[]
    for sp in SPS.state_encoding:
        n,l,j,m,t,tz=sp
        m_values.append(m)

    m_values=np.array(m_values)

    
    # Extract the HF energy of the system ###########################
    
    # we initialize the Hartree-Fock model as a pytorch module
    model=HFEnergyFunctionalNuclear(h_vec=torch.tensor(SPS.energies,dtype=torch.double),V_dict=twobody_matrix,num_neutrons=nparticles_a,num_protons=nparticles_b,neutron_indices=0,proton_indices=size_a,m_values=m_values,multiplier_m_values=0)
    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # training loop
    num_steps = 600
    # to get more info about this just go to the pytorch documentation
    for step in range(num_steps):
        optimizer.zero_grad()
        energy = model()  # calls forward()
        energy.backward()
        optimizer.step()

        if step % 20 == 0 or step == num_steps - 1:
            # see how it goes
            print(f"Step {step:4d} | Energy = {energy.item():.6f}")
            
    energy_hf=energy.item()
    
    # Extract the orbitals for computing the density matrix in the rest space of the 2 quasiparticle sector
    ###############################################################
    
    # we initialize the Hartree-Fock model as a pytorch module
    model_twobody=HFEnergyFunctionalNuclear(h_vec=torch.tensor(SPS.energies,dtype=torch.double),V_dict=twobody_matrix,num_neutrons=2,num_protons=2,neutron_indices=0,proton_indices=size_a,m_values=m_values,multiplier_m_values=0)
    # initialize the optimizer
    optimizer = optim.Adam(model_twobody.parameters(), lr=0.01)
    
    # training loop
    num_steps = 600
    # to get more info about this just go to the pytorch documentation
    for step in range(num_steps):
        optimizer.zero_grad()
        energy_twobody = model_twobody()  # calls forward()
        energy_twobody.backward()
        optimizer.step()

        if step % 20 == 0 or step == num_steps - 1:
            # see how it goes
            print(f"Step {step:4d} | Energy = {energy_twobody.item():.6f}")
    
    # we compute the HF wavefunction in the 2-2 particle sector
    psi_hf=np.zeros(Hamiltonian2bodysector.basis.shape[0])
    # run over the basis element
    for idx_rb,rb in enumerate(Hamiltonian2bodysector.basis):
        occ_idx=np.nonzero(rb)[0]
        proton_idxs=occ_idx[occ_idx >= size_a]
        neutron_idxs=occ_idx[occ_idx < size_a]
        # compute the minors
        phi_neutron=model_twobody.hf_component_from_indices(occ_idx=neutron_idxs,species='neutron')
        # compute the minors
        phi_proton=model_twobody.hf_component_from_indices(occ_idx=proton_idxs,species='proton')
        # get the component
        psi_hf[idx_rb]=phi_neutron*phi_proton
    # normalization factor
    psi_hf=psi_hf/np.linalg.norm(psi_hf)
    # get the density matrix, necessary for the BW-HF method
    density_matrix = csr_matrix(np.einsum('a,b->ab',psi_hf,psi_hf.conj()))
    # reduce the density matrix to the R space, with rho_RR
    density_matrix_rr=QPC2body.particles2restofstates @ density_matrix @ QPC2body.particles2restofstates.T

    # get the H_RR term as the identity for the 1/(E-E_hf)
    hamiltonian_rr_2b_hf=energy_hf*identity(QPC2body.rest_basis.shape[0])
    print('energy_hf=',energy_hf,'\n')
    psiq_hfbw,energy_hfbw,hamiltonian_hfbw,list_infedelities_hfbw,list_errors_hfbw,_=brillouin_wigner_method_hf_ansatz(hamiltonians_hf=[hamiltonian_qq_2b,hamiltonian_qr_2b,hamiltonian_rq_2b,hamiltonian_rr_2b_hf],hamiltonian_qq=hamiltonian_qq,density_matrix_rr=density_matrix_rr,QPC2body=QPC2body,QPC=QPC,threshold=10**-3,eigvals_aci=eigvals_aci,nsteps_iteration=nsteps_iteration)
    
    accuracies_hfbw.append(list_errors_hfbw)
    energies_hf_bw.append(energy_hfbw)
    fidelity=np.abs(np.vdot(psiq_fbw,psiq_hfbw))**2
    fidelities_hf_bw.append(fidelity)
    distance_hf=norm(hamiltonian_fbw- hamiltonian_hfbw,ord='fro')/norm(hamiltonian_fbw,ord='fro')
    operator_fidelities_hfbw.append(distance_hf)

    data = dict(
        nparticles=nparticles,
        energies_exact=energies_exact,
        energies_qq=energies_qq,
        energies_truncated_bw=energies_truncated_bw,
        energies_fbw=energies_fbw,
        accuracies_fbw=accuracies_fbw,
        fidelities_truncated_bw=fidelities_truncated_bw,
        accuracies_hfbw=accuracies_hfbw,
        energies_hfbw=energies_hf_bw,
        fidelities_hfbw=fidelities_hf_bw,
        operator_fidelities_bw=operator_fidelities_bw,
        operator_fidelities_hfbw=operator_fidelities_hfbw,
    )

    with open('data/bw_calculations.pkl', 'wb') as f:
        pickle.dump(data, f)