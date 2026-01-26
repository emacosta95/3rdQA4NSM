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
from scipy.sparse.linalg import eigsh


def full_brillouin_wigner_method(hamiltonians,threshold,eigvals_aci,nsteps_iteration):
    """Apply the Brillouin-Wigner method using the set of hamiltonians considered

    Args:
        hamiltonians (List[csr_matrix]): LIST of the H_QQ, H_QR, H_RQ and H_RR hamiltonians
        threshold (float): minimum value of Delta e= E^(N_iteration)-E^(N_iteration-1) at which the loop stops. It has to be positive
        eigvals_aci (np.ndarray): list of eigenvalues of the nsm hamiltonian to get an accuracy of the error in the energy. 

    Returns:
        psi_qq (np.ndarray): ground state of the effective hamiltonian
        e (float): energy estimated by the BW method
        hamiltonian_tot_qq_full_bw (csr_matrix): effective hamiltonian
        infidelities_full_brillouinwigner_method List[float]: 1-F of psi_qq with respect to the gs of H_QQ^(0) at each iteration
        history_errors_full_brillouinwigner_method List[float]: error with respect to the exact gs energy at each iteration
        delta_hamiltonian (np.ndarray): contribution of all the corrections in the H_QQ.
    """
    target_eigenvalue=0

    
    hamiltonian_qq=hamiltonians[0]
    hamiltonian_qr=hamiltonians[1]
    hamiltonian_rq=hamiltonians[2]
    hamiltonian_rr=hamiltonians[3]
    
    # start computing all the correction terms necessary for the Brillouin-Wigner perturbation theory

        
    # then we compute the energy corrections
    tot_hamiltonian=hamiltonian_qq
    values,psiq_order0=eigsh(hamiltonian_qq,k=1)
    e=values[0]
    e_old=200
    # we fix the delta_e_iteration to enter the while loop
    delta_e_iteration=1.0

    approximations=[]
    print('h_qq value=',e)
    single_term = hamiltonian_rq  # Start with initial term
    for i in trange(nsteps_iteration):
        if i > 0:
            # INSTEAD OF DIRECTLY
            single_term =hamiltonian_rr @ single_term  # Efficient update
        update=(hamiltonian_qr @ single_term)

        approximations.append(update)  # Store result
        
    history_errors_full_brillouinwigner_method=[]
    interaction_terms_full_brillouinwigner_method=[]
    infidelities_full_brillouinwigner_method=[]
    #for i in trange(nsteps_iteration):
    i=0 # we want to count the number of iterations
    while(delta_e_iteration>threshold):    
        tot_hamiltonian=hamiltonian_qq
        delta_hamiltonian=0.
        for j in range(i):
            delta_hamiltonian=delta_hamiltonian+approximations[j]/e**(j+1)
        interaction_terms_full_brillouinwigner_method.append(delta_hamiltonian)
        values,psiq_orderi=eigsh(hamiltonian_qq+delta_hamiltonian,k=1)
        hamiltonian_tot_qq_full_bw=hamiltonian_qq+delta_hamiltonian
        # compute the energy for this order
        e=values[0]
        # compute the energy difference
        delta_e_iteration=e_old-e.copy()
        # now you can update the old energy
        e_old=e.copy()
        fidelity=np.abs(np.vdot(psiq_order0,psiq_orderi))**2
        infidelities_full_brillouinwigner_method.append(1-fidelity)
        history_errors_full_brillouinwigner_method.append(np.abs((e-eigvals_aci[target_eigenvalue])/eigvals_aci[target_eigenvalue]))
        i+=1
        print(f'Iteration {i}: delta_e={delta_e_iteration}')
    print(e)
    print(np.abs((e-eigvals_aci[target_eigenvalue])/eigvals_aci[target_eigenvalue]),'index=',i)
    n_steps_bw=i
    
    
    return psiq_orderi,e,hamiltonian_tot_qq_full_bw,infidelities_full_brillouinwigner_method,history_errors_full_brillouinwigner_method,delta_hamiltonian


def full_brillouin_wigner_method_pauliblockade(hamiltonians,threshold,eigvals_aci,nsteps_iteration):
    """Apply the truncated Brillouin-Wigner method using the set of hamiltonians considered

    Args:
        hamiltonians (List[csr_matrix]): LIST of the H_QQ, H_QR pauli blockade, H_QR standard, H_RQ, H_RR and H_RR pauli_blockade hamiltonians
        threshold (float): minimum value of Delta e= E^(N_iteration)-E^(N_iteration-1) at which the loop stops. It has to be positive
        eigvals_aci (np.ndarray): list of eigenvalues of the nsm hamiltonian to get an accuracy of the error in the energy. 

    Returns:
        psi_qq (np.ndarray): ground state of the effective hamiltonian
        e (float): energy of the brillouin wigner method
        hamiltonian_tot_qq_full_bw (csr_matrix): effective hamiltonian
        infidelities_full_brillouinwigner_method List[float]: 1-F of psi_qq with respect to the gs of H_QQ^(0) at each iteration
        history_errors_full_brillouinwigner_method List[float]: error with respect to the exact gs energy at each iteration
        delta_hamiltonian (np.ndarray): contribution of all the corrections in the H_QQ.
    """
    target_eigenvalue=0
    
    hamiltonian_qq=hamiltonians[0]
    hamiltonian_qr_pauliblockade=hamiltonians[1]
    hamiltonian_qr=hamiltonians[2]
    hamiltonian_rq_pauliblockade=hamiltonians[3]
    hamiltonian_rr=hamiltonians[4]
    hamiltonian_rr_pauliblockade=hamiltonians[5]
    
    # start computing all the correction terms necessary for the Brillouin-Wigner perturbation theory

        
    # then we compute the energy corrections
    tot_hamiltonian=hamiltonian_qq
    values,psiq_order0=eigsh(hamiltonian_qq,k=1)
    e=values[0]
    e_old=200
    # we fix the delta_e_iteration to enter the while loop
    delta_e_iteration=1.0

    approximations=[]
    print('h_qq value=',e)
    single_term = hamiltonian_rq_pauliblockade # Start with initial term
    for i in trange(nsteps_iteration):
        
        # in order to get the pauliblockade at each virtual process transition we need to consider
        # this order of matrix multiplication
        if i ==1:
            # H_QR pauli_blockade H_RR pauli_blockade H_RQ
            single_term =hamiltonian_rr @ single_term  # Efficient update
        
        if i>1:
            # like H_QR pb H_RR pb H_RR pb H_RQ (n-1 pb for n H_RR)
            single_term=hamiltonian_rr_pauliblockade @ single_term
        
        if i==0:
            # at order zero we have H_QR pauli_blockade H_RQ
            update=(hamiltonian_qr @ single_term)
        else:
            # at n-th order we have H_QR pauli_blockade H_RR pauli_blockade ... H_RR pauli_blockade H_RQ
            update=hamiltonian_qr_pauliblockade @ single_term

        approximations.append(update)  # Store result
        
    history_errors_full_brillouinwigner_method=[]
    interaction_terms_full_brillouinwigner_method=[]
    infidelities_full_brillouinwigner_method=[]
    #for i in trange(nsteps_iteration):
    i=0 # we want to count the number of iterations
    while(delta_e_iteration>threshold):    
        tot_hamiltonian=hamiltonian_qq
        delta_hamiltonian=0.
        for j in range(i):
            delta_hamiltonian=delta_hamiltonian+approximations[j]/e**(j+1)
        interaction_terms_full_brillouinwigner_method.append(delta_hamiltonian)
        values,psiq_orderi=eigsh(hamiltonian_qq+delta_hamiltonian,k=1)
        hamiltonian_tot_qq_full_bw=hamiltonian_qq+delta_hamiltonian
        # compute the energy for this order
        e=values[0]
        # compute the energy difference
        delta_e_iteration=e_old-e.copy()
        # now you can update the old energy
        e_old=e.copy()
        fidelity=np.abs(np.vdot(psiq_order0,psiq_orderi))**2
        infidelities_full_brillouinwigner_method.append(1-fidelity)
        history_errors_full_brillouinwigner_method.append(np.abs((e-eigvals_aci[target_eigenvalue])/eigvals_aci[target_eigenvalue]))
        i+=1
        print(f'Iteration {i}: delta_e={delta_e_iteration}')
    print(e)
    print(np.abs((e-eigvals_aci[target_eigenvalue])/eigvals_aci[target_eigenvalue]),'index=',i)
    n_steps_bw=i
    
    
    return psiq_orderi,e,hamiltonian_tot_qq_full_bw,infidelities_full_brillouinwigner_method,history_errors_full_brillouinwigner_method,delta_hamiltonian



def brillouin_wigner_method_hf_ansatz(hamiltonians_hf,hamiltonian_qq,density_matrix_rr,QPC2body,QPC,threshold,eigvals_aci,nsteps_iteration):
    """Computation of the BW method using the HF ansatz

    Args:
        hamiltonians_hf (_type_): _description_
        hamiltonian_qq (_type_): _description_
        density_matrix_rr (_type_): _description_
        QPC2body (_type_): _description_
        QPC (_type_): _description_
        threshold (_type_): _description_
        eigvals_aci (_type_): _description_
        nsteps_iteration (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    target_eigenvalue=0
    # start computing all the correction terms necessary for the Brillouin-Wigner perturbation theory

    hamiltonian_qq_2b=hamiltonians_hf[0]
    hamiltonian_qr_2b=hamiltonians_hf[1]
    hamiltonian_rq_2b=hamiltonians_hf[2]
    hamiltonian_rr_2b=hamiltonians_hf[3]
    
    
    
        
    # then we compute the energy corrections as in all the previous cases
    values,psiq_order0=eigsh(hamiltonian_qq_2b,k=1)
    e=values[0]
    approximations_as_dictionary=[]
    approximations_as_matrix=[]

    single_term = hamiltonian_rq_2b   # Start with initial term
    for i in trange(nsteps_iteration):
        if i >0:
            ########################################################################## GET THIS
            diagonal_rr=hamiltonian_rr_2b
            single_term =diagonal_rr @ single_term  # Efficient update
            ################################################################
            
        update=(hamiltonian_qr_2b @ density_matrix_rr  @ single_term)
        # for each correction we build the 2-body quasiparticle coefficients
        twobody_quasiparticle={}
        for q,base_q in enumerate(QPC2body.quasiparticle_basis):
            for p,base_p in enumerate(QPC2body.quasiparticle_basis):
                a_q,b_q=np.nonzero(base_q)[0]
                a_p,b_p=np.nonzero(base_p)[0]
                q_max=np.max([a_q,b_q])
                q_min=np.min([a_q,b_q])
                p_max=np.max([a_p,b_p])
                p_min=np.min([a_p,b_p])
                # we adopt these coefficients in the full quasiparticle MB basis to compute the corrections
                twobody_quasiparticle[(a_q,b_q,a_p,b_p)]=update[q,p]
                
        # we save these terms as dictionary
        approximations_as_dictionary.append(twobody_quasiparticle)
        approximations_as_matrix.append(update)
    
    # we get the Hardcore boson basis related to the Quasiparticle basis of the NSM Hamiltonian
    HBB=HardcoreBosonsBasis(QPC.quasiparticle_basis)
    approximations_truncated_uniform=[]
    # we compute all the interaction terms in this new basis
    tbar=tqdm(enumerate(approximations_as_dictionary))
    for _,interaction_dictionary in tbar:
        operator=0.
        tbar2=tqdm(interaction_dictionary.keys())
        for key in tbar2:
            a,b,c,d=key
            operator+=HBB.adag_adag_a_a_matrix_optimized(a,b,c,d)*interaction_dictionary[key]
        approximations_truncated_uniform.append(0.5*(operator+operator.T))    
    
    # start computing all the correction terms necessary for the Brillouin-Wigner method as in the other cases
    tot_hamiltonian=hamiltonian_qq
    values,psiq_order0=eigsh(hamiltonian_qq,k=1)
    e=values[0]
    delta_e_iteration=1.0
    e_old=200
        
    history_errors_hf_brillouinwigner_method=[]
    interaction_terms_hf_brillouinwigner_method=[]
    infidelities_hf_brillouinwigner_method=[]
    i=0
    while((delta_e_iteration)>1e-3):    
        tot_hamiltonian=hamiltonian_qq
        delta_hamiltonian=0.
        for j in range(i):
            delta_hamiltonian=delta_hamiltonian+approximations_truncated_uniform[j]/e**(j+1)
        interaction_terms_hf_brillouinwigner_method.append(delta_hamiltonian)
        values,psiq_orderi_hf=eigsh(hamiltonian_qq+delta_hamiltonian,k=1)
        hamiltonian_tot_qq_hf=hamiltonian_qq+delta_hamiltonian
        #update the energy difference
        e=values[0]
        delta_e_iteration=(e_old-e)
        e_old=e.copy()
        #
        fidelity=np.abs(np.vdot(psiq_order0,psiq_orderi_hf))**2
        infidelities_hf_brillouinwigner_method.append(1-fidelity)
        history_errors_hf_brillouinwigner_method.append(np.abs((e-eigvals_aci[target_eigenvalue])/eigvals_aci[target_eigenvalue]))
        print('iteration=',i,' delta_e=',delta_e_iteration,'energy=',e)
        i+=1

    n_steps_bwi=i
    print('Hartree Fock BW energy error=',np.abs((e-eigvals_aci[target_eigenvalue])/eigvals_aci[target_eigenvalue]),'index=',i,'\n')
    
    return psiq_orderi_hf,e,hamiltonian_tot_qq_hf,infidelities_hf_brillouinwigner_method,history_errors_hf_brillouinwigner_method,delta_hamiltonian



def computation_pauli_blockade(basis,QPC,psi_qq):
    """Compute the observables for the implementation of the pauli blockade terms in the truncation Brillouin-Wigner method

    Args:
        basis (csr_matrix): many-body basis of the full NSM hamiltonian
        QPC (QuasiparticlePairing class): the class related to the quasiparticle pairing system associated to the NSM hamiltonian 
        psi_qq (np.ndarray): quasiparticle pairing state

    Returns:
        rho_vacuum (np.ndarray): |00><00| component of the reduced two-body density matrix in the full fermionic basis
        single_particle_density: one-particle density n_i in the full fermionic single particle basis
    """

    # extend the psi_q in the full hilber space
    psi_qq_gs_full_space=QPC.particles2quasiparticles.T @ psi_qq[:]
    psi_qq_gs_full_space/=np.linalg.norm(psi_qq_gs_full_space)

    # initialize the two quantities
    single_particle_density=np.zeros(basis.shape[1])
    rho_vacuum=np.zeros((basis.shape[1],basis.shape[1]))
    # get the modulo square of the psi_q
    psi_mod_square=psi_qq_gs_full_space.conj()*psi_qq_gs_full_space
    for a in range(basis.shape[1]):
        # compute n_i
        single_particle_density[a]=np.sum(psi_mod_square*(basis[:,a]))
        for b in range(basis.shape[1]):
            # compute \rho^(2)_00=<\psi| (1-n)(1-n)|\psi>
            value_vacuum=np.sum(psi_mod_square*(1-basis[:,a])*((1-basis[:,b])))
            rho_vacuum[a,b]=value_vacuum
            
    return rho_vacuum,single_particle_density


