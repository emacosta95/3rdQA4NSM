import qiskit
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import Dict
from qiskit import transpile







def get_hamiltonian(t_onebody:Dict, n_sites:int):

    # List of Hamiltonian terms as 3-tuples containing
    # (1) the Pauli string,
    # (2) the qubit indices corresponding to the Pauli string,
    # (3) the coefficient.
    XX_tuples=[]
    YY_tuples=[]
    Z_tuples=[]
    I_tuples=[]
    for (i,j),t_val in t_onebody.items():
    
        if i==j:
            Z_tuples.append(("Z", [i], -0.5*t_val))
            I_tuples.append(('I',[i],0.5*t_val)) 
        else:
            XX_tuples.append( ("XX", [i, j], 0.25*t_val) )
            YY_tuples.append(("YY", [i, j], 0.25*t_val) )
            
    # We create the Hamiltonian as a SparsePauliOp, via the method
    # `from_sparse_list`, and multiply by the interaction term.
    hamiltonian = SparsePauliOp.from_sparse_list([*XX_tuples, *YY_tuples,*Z_tuples,*I_tuples], num_qubits=n_sites)
    return hamiltonian.simplify()



def get_shots_qiskit(circuit_x, circuit_y, circuit_z, shots=1000,backend=None):

    circuits = {
        "X": circuit_x,
        "Y": circuit_y,
        "Z": circuit_z
    }

    results = {}

    for basis, qc in circuits.items():
        # circuits should be already transpiled for the backend
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        results[basis] = counts
        print(f"Results for basis {basis}: {counts}")
    return results


def obtain_frequencies_qiskit(results, shots):
    frequencies = {}
    labels = ['X', 'Y', 'Z']

    for label in labels:
        counts = results[label]   # Qiskit counts: {'010': 123, ...}

        bitstrings = list(counts.keys())
        values = list(counts.values())
        measures = {}

        for i in range(len(bitstrings)):
            # Convert '0','1' → eigenvalues -1,+1
            arr = np.array([2*(int(b) - 0.5) for b in bitstrings[i]])
            measures[tuple(arr)] = values[i] / shots

        frequencies[label] = measures

    return frequencies

# Deprecated function
def compute_energy(frequencies, t_onebody,n_qubits,renormalization_factor=None):
    energy_component_z = 0.
    energy_component_y = 0.
    energy_component_x = 0.

    for (i, j), value in t_onebody.items():
        if i != j:    
            freq = frequencies['Y']
            for key in freq.keys():
                energy_component_y += 0.25 * value * key[n_qubits-1-i] * key[n_qubits-1-j] * freq[key]
                

            freq = frequencies['X']
            for key in freq.keys():
                energy_component_x += 0.25 * value * key[n_qubits-1-i] * key[n_qubits-1-j] * freq[key]        

        
        elif i == j:
            # convert projector from Z+I to I-Z
            freq = frequencies['Z']
            for key in freq.keys():
                energy_component_z += 0.5 * value * (key[n_qubits-1-i] + 1) * freq[key]
    
    # we add this renormalization factor to perform an approximate restoration of the symmetry sector
    print(energy_component_x,energy_component_y,energy_component_z)
    if renormalization_factor is not None:
        print('bla bla')
        energy_component_y /=  renormalization_factor
        energy_component_x /=  renormalization_factor

    print(energy_component_x,energy_component_y,energy_component_z)

    return energy_component_x + energy_component_y + energy_component_z


def expectation_from_counts(counts, pauli):
    """
    Compute ⟨P⟩ from measurement counts in the correct basis.
    
    counts: dict like {'010': 120, '111': 880}
    pauli: string like 'IXZ'
    """
    shots = sum(counts.values())
    exp = 0.0
    # since the bitstring are in 1,0 we use the parity to get the +1,-1 counting
    for bitstring, c in counts.items():
        # something that we were missing from our previous implementation
        parity = 0 # we start with 1 as eigenvalue
        for i, p in enumerate(pauli):
            if p != 'I': # if p is not identity we get a X,Y,Z operator with eigenvalues +1, -1. 
                if bitstring[i] == '1': # if the state related to the operator is 1 we flip the eigenvalue such that it becomes -1
                    parity ^= 1
                    # in qiskit the -1 eigenvalue is represented by the bit 1 (they are weird)
        exp += ((-1) ** parity) * c / shots

    return exp



def ionq_energy_expectation(circuits, hamiltonian, backend, shots=1000,nparticle_sector=None):
    """
    Compute ⟨H⟩ on IonQ backend using X/Y/Z basis circuits.

    circuits: [circuit_x, circuit_y, circuit_z]
    hamiltonian: SparsePauliOp
    backend: IonQ backend (simulator or hardware)
    shots: number of shots
    """
    
    circuit_x, circuit_y, circuit_z = circuits
    


    # Run
    job_x = backend.run(circuit_x, shots=shots)
    job_y = backend.run(circuit_y, shots=shots)
    job_z = backend.run(circuit_z, shots=shots)

    result_x = job_x.result()
    result_y = job_y.result()
    result_z = job_z.result()

    counts_x = result_x.get_counts()
    counts_y = result_y.get_counts()
    counts_z = result_z.get_counts()
    
    # we add the option of restricting the Symmetry sector by renormalizing the energy components in X and Y
    p_sector=None
    if nparticle_sector is not None:
        filtered_counts_z={b: c for b, c in counts_z.items() if b.count('1') == nparticle_sector}
        p_sector=sum(filtered_counts_z.values())/shots
        counts_z=filtered_counts_z.copy()
    energy_z = 0.0
    energy_xy = 0.0
    for pauli, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        coeff = coeff.real

        # Identity term
        if set(pauli) == {'I'}:
            energy_z += coeff
            continue

            
        # Determine which basis this Pauli string needs
        if all(p in ['I', 'Z'] for p in pauli):
            exp_val = expectation_from_counts(counts_z, pauli)
            energy_z += coeff * exp_val

        elif all(p in ['I', 'X'] for p in pauli):
            exp_val = expectation_from_counts(counts_x, pauli)
            energy_xy += coeff * exp_val

        elif all(p in ['I', 'Y'] for p in pauli):
            exp_val = expectation_from_counts(counts_y, pauli)
            energy_xy += coeff * exp_val
        else:
            raise ValueError(
                f"Pauli term {pauli} mixes X/Y/Z. "
                "This simple implementation assumes separate X, Y, Z groups."
            )

    if p_sector is not None:
        energy_xy_renormalized =energy_xy/ p_sector

        return energy_z,energy_xy,energy_xy_renormalized, p_sector

    else:
        return energy_z+energy_xy,energy_z,energy_xy