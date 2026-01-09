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