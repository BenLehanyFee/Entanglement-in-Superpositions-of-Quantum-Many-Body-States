# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 15:49:43 2026

@author: Ben Lehany-Fee

This file contains the code for the reduced density matrix calculator, both entropy calculators, and the
code for each validation test. This code corresponds to Sections 3.1, 3.2 and 3.4 of the report.
"""

import numpy as np
from State_Gens import Haar_random_generator, Clifford_stab_generator_v3, number_conserve_state_generator

bell_test=False #If True, tests the entropies of the 4 Bell states
prod_test= False #If True, tests the entropies of all product states of assigned sizes of qubits.
bonus_prod_test = False #If True, tests the entropies of product states (|00>+|01>)/sqrt(2) & 
#(|00>+|01>+|10>+|11>)/2, two product states not in the Fock basis states.
subsys_test= False #If True, tests the entropies of a given state to show S(rho_A)=S(rho_B)

#================================================================
#           RDM AND ENTANGLEMENT ENTROPY CALCULATORS
#================================================================

def reduced_density_matrix(vec, qubits_kept, N, normal=False, print_ans=False):
    """
    Edited version of qiskit.quantum_info.partial_trace (removed the Qiskit Statevector wrapping,
    and removed the reverse index counting they use in that function)
    Calculates the partial trace for a vector WITHOUT building the density matrix using np.tensordot
    (using tensor reshaping)
    vec: NumPy array, state vector to find rdm of
    qubits_kept: List, integer values of qubits kept (0 = 1st qubit, 1=2nd qubit, etc.)
    N: Integer, total number of qubits
    normal: Boolean, normalizes the vector before tracing if True
    print_ans: Boolean, prints the density matrix if True
    """
    if normal:
        vec /= np.linalg.norm(vec) #Optional normalization check
    N_A = len(qubits_kept)
    qubits_traced = [i for i in range(N) if i not in qubits_kept] #List of qubits traced out
    arr = vec.reshape([2]*N)
    rho = np.tensordot(arr, arr.conj(), axes=(qubits_traced, qubits_traced)) 
    #This calculates the outer product of the rows of arr with indices in kept qubits 
    #(arr*arr.conj()), and sums the rows of arr with indices in qubits_traced
    rho_A = rho.reshape(2**N_A, 2**N_A) #Reshape back into density matrix form
    if print_ans:
        print(rho_A)
    return rho_A      

def von_Neumann_entropy(rho, print_ans=True):
    """
    Returns the von Neumann entropy for a given reduced density matrix by taking its eigenvalues 
    and building S = -sum(lambda_i ln lambda_i), lambda_i being the eigenvalues of rho.
    rho: NumPy array, the reduced density matrix
    print_ans: Boolean, prints answers if True
    """
    eigvals = np.linalg.eigvalsh(rho) #eigvalsh used as every density matrix is Hermitean
    eigvals = eigvals[np.abs(eigvals)>1e-12] #Safety check, eliminates eigenvalues at 0 (causes errors with log(eig))
    S = -sum(eigvals * np.log(eigvals)) #von Neumann entropy
    if print_ans:
        print("Von Neumann Entropy:", S)
    return S

def Rényi_2_entropy(rho, print_ans=True):
    """
    Returns the Rényi-2 entropy for a given reduced density matrix, by calculating 
    S = -ln(Tr(rho_A^2))
    rho: NumPy array, reduced density matrix
    print_ans: Boolean, prints answers if True
    """
    S = -np.real(np.log(np.trace(rho @ rho))) #Rényi-2 entropy
    if print_ans:
        print("Rényi-2 Entropy:", S)
    return S

#================================================================
#                   FUNCTIONS FOR VALIDATION TESTS
#================================================================

def generate_basis_states_per_qubit_no(N, print_ans=True):
    """
    Generates and returns all computational basis states per qubit number as a list.
    N: Integer, dimension
    print_ans: Boolean, prints answers if True
    """
    state_list = []
    for index in range(0, 2**N): #2**N used as range excludes last value
        psi=np.zeros(2**N) #|0>^N
        psi[index]=1 #Generates every matrix with a 1 in a single entry, and 0 in all others (i.e.)
        state_list.append(psi)
    if print_ans:
        print(state_list)
    return state_list

def single_product_test(state, N, qubits_kept, print_all=False, TOL=1e-15):
    """
    Tests the von Neumann/Rényi-2 entropies for a given product state, for a given subsystem.
    Returns False if either entropy is non-zero, within a given tolerance.
    state: NumPy array, the state to be tested.
    N: Integer, number of qubits.
    qubits_kept: List, integer positions of qubits in subsystem (index 0 for the leftmost one)
    print_all: Boolean, prints all entropies calculated if True
    TOL: Float, cutoff point for floating point errors (i.e. if |a|<TOL, we treat a=0)
    """
    if print_all:
        print(state)
    rho = reduced_density_matrix(vec=state, qubits_kept=qubits_kept, N=N, normal=False,
                                     print_ans=False) #Reduced density matrix
    v=von_Neumann_entropy(rho, print_ans=print_all)
    r=Rényi_2_entropy(rho, print_ans=print_all)
    if (abs(v)>TOL) or (abs(r)>TOL): #Only counts an entropy value as non-zero if its absolute value is greater than the tolerance.
        return False #Will give an error in main test if False
    else:
        return True #Ok if True
   
def subsystem_test(state, N, A_qubits, TOL=1e-12, print_all=False):
    """
    Calculates S(rho_A), S(rho_B) for the von Neumann and Rényi-2 entropies
    for a given pure state, and subsystems A and B.
    Returns False if the two are not equal within a certain tolerance, and returns True if they match.
    state: Numpy Array, state being tested.
    N: Integer, number of qubits.
    A_qubits: List, integer positions of qubits for subsystem A (starts at 0)
    TOL: Float, tolerance for error
    print_all: Boolean, prints everything if True
    Note the qubits of B are assumed to be all qubits not in A.
    """
    B_qubits = [num for num in range(N) if num not in A_qubits] #Qubits in subsystem B
    #Calculating S(rho_A)
    rho_A = reduced_density_matrix(vec=state, qubits_kept=A_qubits, N=N, normal=False, 
                                   print_ans=False) #RDM of A
    vn1 = von_Neumann_entropy(rho_A, print_ans=False)
    r1 = Rényi_2_entropy(rho_A, print_ans=False)
    #Calculating S(rho_B)
    rho_B = reduced_density_matrix(vec=state, qubits_kept=B_qubits, N=N, normal=False, 
                                   print_ans=False) #RDM of B
    vn2 = von_Neumann_entropy(rho_B, print_ans=False)
    r2 = Rényi_2_entropy(rho_B, print_ans=False)
    if print_all:
        print(state)
        print("Entropies for Subsystem A:")
        print("Von Neumann Entropy for Subsystem A: ", vn1, "Rényi-2 Entropy for Subsystem A: ", r1)
        print("Entropies for Subsystem B:")
        print("Von Neumann Entropy for Subsystem B: ", vn2, "Rényi-2 Entropy for Subsystem B: ", r2)
        print("\n")
    #Safety check
    if (abs(vn2-vn1)>TOL) or (abs(r2-r1)>TOL):
        return False #If False, the entropies don't match
    else:
        return True #If True, the entropies are ok

#================================================================
#                           VALIDATION TESTS
#================================================================

#Bell State Tests: Tests the von Neumann/Rényi-2 entropies for the 4 Bell states.
bell_states = [np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
               np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]),
               np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0]),
               np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])]
if bell_test:
    for state in bell_states:
        print(state)
        print("Tests on First Qubit Subsystem:")
        rho_A = reduced_density_matrix(vec=state, qubits_kept=[0], N=2, normal=False, 
                                       print_ans=False)
        von_Neumann_entropy(rho_A, print_ans=True)
        Rényi_2_entropy(rho_A, print_ans=True)
        print("Tests on Second Qubit Subsystem:")
        rho_B = reduced_density_matrix(vec=state, qubits_kept=[1], N=2, normal=False, 
                                       print_ans=False)
        von_Neumann_entropy(rho_B, print_ans=True)
        Rényi_2_entropy(rho_B, print_ans=True)
        print("\n")
#Both von Neumann and Rényi-2 entropies give S~=ln(2) for each Bell state.

#Product Test: This tests all computational basis states for a specified number of qubits, and a specified subsystem size, and shows that
#they all have 0 von Neumann and Rényi-2 entropy.  
if prod_test: 
    tested_no_qubits=[1, 2, 3, 4, 5, 6] #This list can be changed to test different system sizes.
    subsystem_A_qubits=[0] #This list can be changed to test different subsystem qubits.
    print_all=False #This can be changed, prints all entropies and states if True.
    all_passed=True #Used to check if every product state has 0 entropies
    for N in tested_no_qubits:
        N_qubit_states = generate_basis_states_per_qubit_no(N=N, print_ans=False)
        print(f"{N}-qubit tests:") if print_all else None #Just for print_all=True
        for state in N_qubit_states:
            #Individual tests for each product state for a fixed N
            result = single_product_test(state, N, qubits_kept=subsystem_A_qubits, print_all=print_all)
            if result==False:
                all_passed=False
                break #Safety check, breaks everything if one of the entropies is non-zero
            print("\n") if print_all else None #Just for print_all=True
    if all_passed==False:
        print("Product Test Failed: Not all States have 0 Entropies")
    else:
        print("Product Test Success: 0 Entropies for Every State Tested")
#Tested this with every system size of <=6 qubits, and every possible combination of qubits preserved
#for the subsystem, it still gives both entropies to be 0 for every state.


#Bonus Product Test: Tests the entropies of product states (|00>+|01>)/sqrt(2) & (|00>+|01>+|10>+|11>)/2
if bonus_prod_test:
    state1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0]) #(|00>+|01>)/sqrt(2)
    state2 = np.array([1/2, 1/2, 1/2, 1/2]) #(|00>+|01>+|10>+|11>)/2
    #Tests on first state for both single-qubit subsystems
    test1a = single_product_test(state1, N=2, qubits_kept=[0], print_all=False)
    test1b = single_product_test(state1, N=2, qubits_kept=[1], print_all=False)
    #Tests for second state on both single-qubit subsystems
    test2a = single_product_test(state2, N=2, qubits_kept=[0], print_all=False)
    test2b = single_product_test(state2, N=2, qubits_kept=[1], print_all=False)
    if not (test1a==test1b==test2a==test2b==True):
        print("Error: Not All Entropies Equal to 0") #Error message if entropies not 0
    else:
        print("Both States have Entropy 0 With Both Measures") #All ok

#Subsystem Test:
if subsys_test:
    #Samples to be tested from the Haar-random, Clifford and number-conserving families.
    #Specific details here can be changed, e.g. number of qubits, or amount of states taken
    print_all = False #If True, will print each state and its entropies of both subsystems for both measures.
    Haar_test = Haar_random_generator(no_states=100, N=4) #Haar-random states
    Cliff_test = Clifford_stab_generator_v3(no_states=100, N=4) #Clifford states
    conserve_test = number_conserve_state_generator(no_states=100, N=4, n=2) #Number-conserving states
    A_qubits = [0,2] #Qubits in subsystem A, can be changed
    N=4 #Number of qubits in the system, can be changed.
    total_test = Haar_test + Cliff_test + conserve_test #Combining all 3 types of state into one test
    for state in total_test:
        result = subsystem_test(np.array(state), N, A_qubits, print_all=print_all)
        if result==False: #If any mismatch for any tested state, breaks the whole thing.
            print("Error: Mismatch in Entropies Between Subsystems")
            break
    else:
        print("Subsystem Test Success: All Tested Entropies Match")
#From bell_test and prod_test, both the von Neumann and Rényi-2 entropies are identical for product
#states and Bell states. Haar-random, Clifford and number-conserving states are 
#accurate up to 10^-12 for 4 qubits, and only show occasional errors for 10^-15 tolerance from testing.