import numpy as np
import json
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from src.qiskit_utils import get_hamiltonian
from qiskit_ionq import IonQProvider, GPI2Gate, MSGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter,QDrift
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile
from qiskit.quantum_info import Statevector
import json
from qiskit.providers.basic_provider import BasicSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError
from qiskit_aer import AerSimulator
from typing import Dict


# hyperparameters
total_time=4
time_steps=40



# function for bluilding the target hamiltonian
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


# class for building the circuit for the QA algorithm, and to calculate the expectation value of a Hamiltonian with respect to the circuit state.
class QACircuit():
    
    def __init__(self, n_qubits, g_onebody, h_driver, total_time, n_steps):
        self.n_qubits = n_qubits
        self.g_onebody = g_onebody
        self.h_driver = h_driver
        self.total_time = total_time
        self.n_steps = n_steps
        self.dt = total_time / n_steps
        self.circuit = QuantumCircuit(n_qubits)



    def initialize_state(self, initial_state):
        """Initialize the quantum circuit to a given initial state.

        Args:
            initial_state (list): List of qubit indices to be set to |1>.
        """
        for qubit in initial_state:
            self.circuit.x(qubit)
    
    def build_circuit(self,save_energy=False,hamiltonian_target=None,hamiltonian_driver=None):
        for i in range(self.n_steps):
            t = i * self.dt
            if save_energy:
                # Add measurement to save energy at each step
                if i==0:
                    self.energy=[]
                self.energy.append(self.expectation_value((1-t/self.total_time)*hamiltonian_driver+t/self.total_time*hamiltonian_target))
            self.circuit.compose(self.unitary_step(t), inplace=True)


    def expectation_value(self, hamiltonian):
        """Calculate the expectation value of a Hamiltonian with respect to the circuit state.

        Args:
            hamiltonian (SparsePauliOp): The Hamiltonian operator.

        Returns:
            float: The expectation value.
        """
        statevector = Statevector(self.circuit)
        exp_val = statevector.expectation_value(hamiltonian)
        return exp_val.real

    def unitary_step(self,t):
        """Create the time dependent unitary step for the QA algorithm. It corresponds to exp(-i*H(t)*dt),
        where H(t) is the time dependent Hamiltonian of the QA algorithm, and dt is the time step.
        We use a 1st order Trotterization to implement the unitary step,
        which means that we first apply the driver Hamiltonian terms, and then we apply the target Hamiltonian terms.

        Args:
            t (float): time parameter.
            
        Returns:
            QuantumCircuit: The unitary step circuit.
        """
        qc=QuantumCircuit(self.n_qubits)
        

        
        # driver Hamiltonian terms in the 1st order Trotterization
        
        # the driver Hamiltonian is diagonal in the Z basis, so we can implement it with Rz gates
        for key in h_driver.keys():
            i=key
            coeff=self.dt*self.h_driver[key]*self.driver_d[int(t/self.dt)]
            qc.rz(-0.5*coeff, i)
        
        # for the target Hamiltonian, we have both diagonal and off-diagonal terms,
        # so we need to implement them with Rz and Rxx/Ryy gates respectively.
        # we start with the diagonal terms of the target Hamiltonian in the 1st order Trotterization
        for i in range(self.n_qubits):
            coeff=self.dt*self.g_onebody[(i,i)]*self.driver_t[int(t/self.dt )]
            qc.rz(-0.5*coeff, i)

        # and then we add the off diagonal terms of the target Hamiltonian in the 1st order Trotterization
        for i in range(self.n_qubits):
            for j in range(i+1,self.n_qubits):
                coeff=self.dt*self.g_onebody[(i,j)]*self.driver_t[int(t/self.dt)]
                qc.rxx(0.5*coeff, i, j)
                qc.ryy(0.5*coeff, i, j)
            
        return qc
    
    def get_driver(self,driver_d=None,driver_t=None):
        
        time=np.linspace(0,self.total_time,self.n_steps)
        
        if driver_d is None:
            self.driver_d=1-time/self.total_time
        else:
            self.driver_d=driver_d
        
        if driver_t is None:
            self.driver_t=time/self.total_time
        else:
            self.driver_t=driver_t
    
    
    def circuit_measure_z(self):
        """Add measurement in the Z basis to the circuit."""
        circuit_z=self.circuit.copy()
        circuit_z.measure_all()

        return circuit_z
    
    def circuit_measure_x(self):
        """Add measurement in the X basis to the circuit."""
        circuit_x=self.circuit.copy()
        for qubit in range(self.n_qubits):
            circuit_x.h(qubit)
        circuit_x.measure_all()
        return circuit_x
    
    def circuit_measure_y(self):
        """Add measurement in the Y basis to the circuit."""
        circuit_y=self.circuit.copy()
        for qubit in range(self.n_qubits):
            circuit_y.sdg(qubit)
            circuit_y.h(qubit)
        circuit_y.measure_all()
        return circuit_y
    
    

# load the coupling terms J_ij of J_ij X_i X_j + Y_i Y_j of the target Hamiltonian, which are stored in a numpy file as a dictionary with keys (i,j) and values J_ij.
# the couplings are g_12=1.04 g_23=3.76 g_13=-3.76 // diagonal g_11=-8.43 g_22=-8.43 g_33=-5.12
keys=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
values=[-8.43,-8.43,-5.12,1.04, -3.76, 3.76]
n_qubits=3

g_onebody={}

for a,key in enumerate(keys):
    i,j=key
    g_onebody[(i,j)]=values[a]
    if i!=j:
        g_onebody[(j,i)]=values[a]

#### build the driver Hamiltonian ####
 
coupling_term=-8.43# this is the diagonal term of the H_T with the quasiparticle bitstring selected as in the paper
Z_tuples=[("Z", [0], -0.5*coupling_term)]
I_tuples=[('I',[0],0.5*coupling_term)]
# we need this dictionary to set the driver Hamiltonian in the unitary step of the QA algorithm
h_driver={0:coupling_term}
        
# We create the Hamiltonian as a SparsePauliOp, via the method
# `from_sparse_list`, and multiply by the interaction term.
hamiltonian_driver = SparsePauliOp.from_sparse_list([*Z_tuples,*I_tuples], num_qubits=n_qubits)

#### build the target Hamiltonian ####
hamiltonian_target=get_hamiltonian(g_onebody,n_qubits)




QAHe6=QACircuit(n_qubits, g_onebody, h_driver, total_time=total_time, n_steps=time_steps)

QAHe6.initialize_state([0])  # Initialize to |100> state

QAHe6.get_driver()

QAHe6.build_circuit(save_energy=True, hamiltonian_target=hamiltonian_target,hamiltonian_driver=hamiltonian_driver)
# only for short circuits


print('energy estimated through Statevector=' + str(QAHe6.expectation_value(hamiltonian_target)))

provider = IonQProvider()
backend_native = provider.get_backend("ionq_simulator", gateset="native")
print(QAHe6.circuit.count_ops)
ionq_circuit_transpiled=transpile(QAHe6.circuit, backend=backend_native,optimization_level=0)

print('number of operations:', ionq_circuit_transpiled.count_ops(),'\n')