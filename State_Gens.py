# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:58:37 2026

@author: Ben Lehany-Fee

This file contains the code of all state generators, (i.e. for Haar-random, Clifford stabilizer and number-conserving states.)
This corresponds to Section 3.3 of the report.
"""

import numpy as np
import stim
import scipy as sp
import qiskit

#Pauli spin matrices and spin operators (used for get_c_sparse)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I_sparse = sp.sparse.identity(2, format='csr')
Z_sparse = sp.sparse.csr_matrix(Z)
sigma_plus = 0.5 * sp.sparse.csr_matrix(X - 1j*Y) #sigma_plus|0>=|1>
sigma_minus = 0.5 * sp.sparse.csr_matrix(X + 1j*Y) #sigma_minus|1>=|0>

#================================================================
#                   HAAR-RANDOM STATE GENERATOR
#================================================================

def Haar_random_generator(no_states, N, seed=None, print_ans=False):
    """
    Generates a collection of Haar-random states by sampling real/complex parts of each value
    from the Gaussian distribution.
    no_states: Integer, number of states generated.
    N: Integer, number of qubits.
    seed: Integer/None, determines the normal distribution to sample real/complex values from. (None by default)
    print_ans: Boolean, prints answers if True (False by default)
    Returns the results as NumPy Arrays in a list
    """
    rng = np.random.default_rng(seed) #Random number generation
    results=[] #Placeholder list for results, every state added to this
    for _ in range(no_states):
        real_vals = rng.normal(loc=0, scale=1, size=2**N) #Random real parts
        complex_vals = rng.normal(loc=0, scale=1, size=2**N) #Random complex parts
        psi = real_vals+(complex_vals*1j)
        psi /= np.linalg.norm(psi) #Normalizing
        results.append(psi)
    if print_ans: #Print all states if True
        print(results)
    return results

#================================================================
#              CLIFFORD STABILIZER STATE GENERATORS
#================================================================

def Clifford_stab_generator_v1(no_states, N, gate_range, seed=None, print_ans=False):
    """
    Generates a random sample of Clifford stabilizer states. This version picks a random number from a range of numbers 
    (representing how many gates to apply), and applies that amount of random H,S and CNOT gates.
    Added rng for number of gates applied, and pre-generated both the number of gates applied and the gates per state for speed.
    NOTE: STIM ALWAYS PICKS THE SAME PHASE UNDER STIM.TABLEAUSIMULATOR AND STIM.TABLEAU.TO_STATE_VECTOR, SO THIS IS NOT ACCURATE
    This version is NOT used in the final report.
    no_states: Integer, number of states generated
    N: Integer, number of qubits
    gate_range: List, should contain TWO INTEGERS, for the min and max numbers respectively of gates to apply to ground state
    Any other values in gate_range will be ignored.
    seed: Integer, used for reproducibility, None by default
    print_ans: Boolean, prints answers if True
    """
    rng = np.random.default_rng(seed) #Random number generation
    results=[] #Empty list for results
    no_gates_list = rng.integers(gate_range[0], gate_range[1], size=no_states) #Pre-generate all amounts of gates to apply.
    for i in range(no_states):
        circuit=stim.Circuit() #Holds instructions for circuit, will be made of Hadamard, phase and CNOT gates, applied to ground state.
        no_gates = no_gates_list[i]
        #Main Loop to Apply Random Gates
        gate_type = rng.choice(["H", "S", "CNOT"], size=no_gates) #Pre-generate all gates to apply for each state
        for j in range(no_gates):
            if gate_type[j] in ["H", "S"]:
                rng_qubit = int(rng.integers(0, N)) #Random chosen target qubit (int to remove 1-element NumPy array)
                circuit.append(gate_type[j], [rng_qubit]) #Add the random gate (rng_qubit as list for circuit.append)
            else: #Adding CNOT Gate
                rng_qubit = rng.integers(0, N, size=2) #2 qubits chosen (target and control)
                while rng_qubit[0]==rng_qubit[1]: #Safety check to prevent repeated indices for CNOT gate.
                    rng_qubit[1]=rng.integers(0, N)
                circuit.append(gate_type[j], list(rng_qubit))
        tableau = stim.TableauSimulator() #Makes tableau of the circuit
        tableau.do(circuit) #Applies everything to ground state
        state = tableau.state_vector() #Resulting Clifford state
        results.append(state)
    if print_ans:
        print(results)
    return results

def Clifford_stab_generator_v2(no_states, N, print_ans=False):
    """
    An alternate function to try to generate Clifford stabilizer states. This uses stim's Tableau.Random function
    to ensure a uniform sample across the Clifford stabilizer states.
    NOTE: STIM ALWAYS PICKS THE SAME PHASE UNDER STIM.TABLEAUSIMULATOR AND STIM.TABLEAU.TO_STATE_VECTOR, SO THIS IS NOT ACCURATE
    This is NOT used in the final report.
    no_states: Integer, number of states generated
    N: Integer, number of qubits
    print_ans: Boolean, prints answers if True
    """
    results = [] #List for states generated
    for _ in range(no_states):
        random_Clifford = stim.Tableau.random(N) #Random Clifford operation
        sim = stim.TableauSimulator() #|0>^N
        sim.do_tableau(random_Clifford, list(range(N))) #Apply the Clifford operation to the vacuum state.
        results.append(sim.state_vector())
    if print_ans:
        print(results)
    return results

def Clifford_stab_generator_v3(no_states, N, print_ans=False, seed=None):
    """
    This is the final (and correct) generator for Clifford stabilizer states. This uses Qiskit's quantum_info.random_Clifford
    function, which considers all phases properly, and samples the Clifford stabilizer states uniformly at random.
    no_states: Integer, number of states to generate.
    N: Integer, number of qubits
    print_ans: Boolean, prints answer if True (False by default)
    seed: Integer, used for reproducibility (None by default)
    """
    rng = np.random.default_rng(seed) #random_clifford actually uses this.
    results = [] #Empty list for states generated.
    psi0 = qiskit.quantum_info.Statevector.from_label('0' * N) #|0>^N, pre-generated for efficiency
    for _ in range(no_states):
        Clifford_operation = qiskit.quantum_info.random_clifford(N, seed=rng) #Random Clifford operation
        circuit = Clifford_operation.to_circuit() #Decomposes this into H,S,CNOT Circuit (more efficient)
        state = psi0.evolve(circuit) # Apply Clifford to |0>^N
        results.append(state.data)  #Adds NumPy array to the list
    if print_ans:
        print(results)
    return results

#================================================================
#  NUMBER-CONSERVING STATE GENERATOR (AND SUPPORTING FUNCTIONS)
#================================================================

def generate_random_unitary(d, seed=None, print_ans=False):
    """
    Mezzadri's method of generating random unitary matrices. Samples independent, identically
    distributed numbers from the complex Gaussian distribution, and applies QR decomposition.
    Returns a random unitary matrix.
    d: Integer, size of matrix (d, d)
    seed: Integer, seed for real/complex gaussian distribution (None by default)
    print_ans: Boolean, prints unitary generated if True (False by default)
    """
    rng= np.random.default_rng(seed) #Random number generation
    real_vals = rng.normal(loc=0, scale=1, size=(d, d)) #Random real parts
    complex_vals = rng.normal(loc=0, scale=1, size=(d, d)) #Random complex parts
    matrix = real_vals + (1j*complex_vals)
    q,r = sp.linalg.qr(matrix) #QR decomposition
    lam = np.diagonal(r) #Diagonalizing the matrix
    ph = lam/np.abs(lam) #Dividing each value by its absolute value
    q = np.multiply(q, ph, q) #This matrix is random w.r.t. Haar-measure
    if print_ans:
        print(q)
    return q

def get_c_sparse(N, i, dagger, print_ans=False):
    """
    Builds and returns a Jordan-Wigner operator as a tensor product of Z operators and sigma +/-. 
    Done through sparse matrices for efficiency.
    N: Integer, number of qubits
    i: Integer, index of Jordan-Wigner operator (This counts from 1 to N, e.g. 1=1st qubit, etc.)
    dagger: Boolean, gives c^+ if True, and c if False
    print_ans: Boolean, prints the operator if True (False by default)
    """
    operators = [] #List to contain all operators to multiply together to get c/c^+
    for k in range(1, N+1): #Subscripts 1 to N
        if k<i:
            operators.append(Z_sparse)
        elif k==i:
            if dagger:
                operators.append(sigma_plus) #Adds fermion
            else:
                operators.append(sigma_minus) #Removes fermion
        else: #k>i
            operators.append(I_sparse)
    result = operators[0] #Initial matrix to multiply all other matrices onto
    for vector in operators[1::]:
        result=sp.sparse.kron(result, vector, format='csr') #Individual tensor product
    if print_ans:
        print(result)
    return result
    
def number_conserve_state_generator(no_states, N, n, seed=None, print_ans=False):
    """
    Generates and returns a list of number-conserving free-fermion states.
    no_states: Integer, number of states generated
    N: Integer, number of qubits
    n: Integer, number of particles (0<n<N)
    seed: Integer, used for reproducibility (None by default)
    print_ans: Boolean, prints answers if True (False by default)
    """
    state_list=[] #Empty list for states generated
    c_dag_ops=[get_c_sparse(N, k, dagger=True, print_ans=False) for k in range(1, N+1)] #Pre-built to save time
    vacuum_state = np.zeros(2**N, dtype=complex)
    vacuum_state[0] = 1 #|0>^N, Pre-generated for efficiency
    for i in range(no_states): #Iterates for a chosen no of states
        state = vacuum_state
        if seed!=None:
            seed += 1 #This ensures different unitaries for each beta operator
        V = generate_random_unitary(N, seed=seed, print_ans=False) #(N, N) matrix
        for site in range(0, n): #Site taken as 0-N, NOT 1-N+1
            beta = sum(V[j, site] * c_dag_ops[j] for j in range(N)) #One beta operator
            state = beta @ state
        state /= np.linalg.norm(state) #Normalization of state afterwards
        state_list.append(state)
    if print_ans:
        print(state_list)
    return state_list

