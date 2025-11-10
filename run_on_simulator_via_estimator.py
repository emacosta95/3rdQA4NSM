import numpy as np
import json
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler,Estimator, Options,RuntimeOptions
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.primitives import StatevectorEstimator
# --------------------------
# Load one-body data
# --------------------------
data_onebody = np.load('data/matrix_elements_h_eff_2body/one_body_nn_p.npz')
keys = data_onebody['keys']
values = data_onebody['values']
n_qubits = 3

t_onebody = {}
for a, key in enumerate(keys):
    i, j = key
    t_onebody[(i, j)] = values[a]

# Build target Hamiltonian
from src.qiskit_utils import get_hamiltonian
hamiltonian_q = get_hamiltonian(t_onebody, n_qubits)

# Driver Hamiltonian
coupling_term = -8.4321
Z_tuples = [("Z", [0], -0.5*coupling_term)]
I_tuples = [('I', [0], 0.5*coupling_term)]
hamiltonian_driver = SparsePauliOp.from_sparse_list([*Z_tuples, *I_tuples], num_qubits=n_qubits)

# --------------------------
# Time evolution circuit
# --------------------------
time_steps = 30
tf = 3
time = np.linspace(0, tf, time_steps)
dt = tf / time_steps
driver = 1 - time/tf

circuit_time_evolution = QuantumCircuit(hamiltonian_q.num_qubits)
circuit_time_evolution.x([0])  # initial state

for n, t in enumerate(time):
    hamiltonian_t = (driver[n])*hamiltonian_driver + (1-driver[n])*hamiltonian_q
    exp_H_t = PauliEvolutionGate(hamiltonian_t, time=dt, synthesis=SuzukiTrotter(order=1))
    circuit_time_evolution.append(exp_H_t, range(hamiltonian_q.num_qubits))




# --------------------------
# Connect to IBM Quantum
# --------------------------

backend = BasicSimulator()
print("Running on:", backend.name)

estimator =StatevectorEstimator()



# Transpile
# circuit_time_evolution = transpile(
#     circuit_time_evolution,
#     optimization_level=3,
#     backend=backend,
#     #basis_gates=['cz','id','rz','x'] #cz, id, rx, rz, rzz, sx, x
# )

from qiskit.transpiler import generate_preset_pass_manager

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
isa_circuit = pm.run(circuit_time_evolution)
print(f">>> Circuit ops (ISA): {isa_circuit.count_ops()}")

hamiltonian_terms=[]
energies=[]
errors=[]

labels=[]
for label, coeff in zip(hamiltonian_q.paulis.to_labels(), hamiltonian_q.coeffs):
    print(f"Term: {label}, Coefficient: {coeff.real:.6f}")
    labels.append(label)
    

for hamiltonian_term in hamiltonian_q:

    isa_observable = hamiltonian_term.apply_layout(isa_circuit.layout)



    # Run the estimator: compute ⟨ψ|H|ψ⟩
    job = estimator.run(
        [(isa_circuit, isa_observable)],
    )
    result = job.result()

    energy = result[0].data.evs  # Expectation value of the Hamiltonian
    energy_err = result[0].data.stds  # Standard deviation (if available)

    print(f"\nEstimated energy ⟨H⟩ = {energy:.6f} ± {energy_err:.6f}")
    hamiltonian_terms.append(hamiltonian_term)
    energies.append(energy)
    errors.append(energy_err)
    
np.savez('data/qiskit_circuit_energy_estimation_results_on_simulator_tau_3',energies=energies,errors=errors,hamiltonian_terms=hamiltonian_terms,labels=labels)

