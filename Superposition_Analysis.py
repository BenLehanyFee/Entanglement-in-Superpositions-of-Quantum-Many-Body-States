# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:18:38 2026

@author: Ben Lehany-Fee

This file contains all the code to plot <S_2(1)>, S_2(m) and dS_2(m), including plotting functions to compare each quantity between different state families, and different sizes of N.
This corresponds to Sections 3.6 and 4.2 of the report.
"""
                             
import numpy as np
import matplotlib.pyplot as plt
from RDM_Entropy_Calcs_and_Validation import reduced_density_matrix, Rényi_2_entropy
from State_Gens import Haar_random_generator, number_conserve_state_generator, Clifford_stab_generator_v3

S_vs_m_system_size_study = False #If True, plots <S_2(1)>, S_2(m) and dS_2(m) in separate subplots for each state family as presented in report.
#(i.e Haar-random states have 3 subplots as part of one larger plot, Clifford and number-conserving plots give 3 separate figures each)
Haar_single_state_vs_Clifford_superpos = False #If True, this plots the Haar-random mean single-state entropy, and the Clifford superposition entropy
#(the latter should approximately converge to the former)
number_conserve_vs_log_m = False #If True, plots dS_2(m) against ln(m) (highlights the non-logarithmic nature of dS2 for number-conserving states)
dS_state_comparison = False #If True, plots the mean single-state entropy, the superposition entropy and their difference in 3 separate subplots
#(each subplot contains the given quantity for all 3 state families)

#================================================================
#       SUPPORTING FUNCTIONS TO CALCULATE dS2 FOR PLOTTING
#================================================================

def equal_weight_superposition(state_list, print_ans=False):
    """
    Creates an equal sum of a sample of states, and then normalizes it.
    state_list: Collection of states to be summed (ensure this is a list of Numpy Arrays?)
    print_ans: Boolean, print answer if True
    """
    total_sum = sum(state_list) #Sum all states
    superpos = total_sum / np.linalg.norm(total_sum) #Normalize the sum
    if print_ans: #Print the superposition if True
        print(superpos)
    return superpos

def mean_Rényi_entropy(state_list, N, qubits_kept, print_ans=False):
    """
    Calculates the mean Rényi-2 entropy for a given list of states.
    state_list: List, contains the states to measure mean entropy of
    N: Integer, number of qubits
    qubits_kept: List, contains integers to represent qubits NOT traced out (starts from 0=1st qubit, etc.)
    print_ans: Boolean, prints answer if True
    """
    entropies=[] #List for the entropy of each state
    for state in state_list:
        rho = reduced_density_matrix(vec=state, qubits_kept=qubits_kept, N=N, print_ans=False) #Reduced density matrix
        S2 = Rényi_2_entropy(rho, print_ans=False) #Entropy of 1 state
        entropies.append(S2)
    mean_S2 = np.mean(entropies) #Mean Rényi-2 entropy
    if print_ans: #Print mean entropy if True
        print(mean_S2)
    return mean_S2

def superposition_Rényi_entropy(state_list, N, qubits_kept, print_ans=False):
    """
    Calculates the Rényi-2 entropy for the equal-weight superposition of a given list of states.
    state_list: List, contains the states to measure superposition entropy of
    N: integer, number of qubits
    qubits_kept: List, contains integers to represent qubits NOT traced out (starts from 0=1st qubit, etc.)
    print_ans: Boolean, prints answer if True
    """
    superpos = equal_weight_superposition(state_list=state_list, print_ans=False) #Superposition
    superpos_rdm = reduced_density_matrix(vec=superpos, qubits_kept=qubits_kept, N=N, print_ans=False) #Superposition RDM
    superpos_S2 = Rényi_2_entropy(rho=superpos_rdm, print_ans=False) #Entropy
    if print_ans: #Print entropy if True
        print(superpos_S2)
    return superpos_S2

def Rényi_plot_vals_one_step(N, state_list, qubits_kept, print_ans=False):
    """
    Calculates dS2(m) = S2(m) - <S2(1)>, then S2(m), and finally <S2(1)>. Used to get the values for plots of
    dS2 against m, this gives the values needed FOR A SINGLE VALUE OF M
    N: Integer, total number of qubits
    state_list: List, contains family of states to test
    qubits_kept: List, contains all INTEGERS of qubits to keep (START AT 0)
    print_ans: Boolean, print values if True (False by default)
    """
    S2_superpos = superposition_Rényi_entropy(state_list, N, qubits_kept, print_ans=False) #Rényi-2 entropy of superposition
    S2_mean = mean_Rényi_entropy(state_list=state_list, N=N, qubits_kept=qubits_kept, print_ans=False) #Average entropy of all states individually
    dS2 = S2_superpos - S2_mean #Difference
    if print_ans:
        print("dS2:", dS2, "Superposition Entropy: ", S2_superpos, "Mean Entropy of Individual States: ", S2_mean)
    return dS2, S2_superpos, S2_mean

def smart_state_generator(state_gen, no_states, N, n=None, seed=None):
    """
    This function chooses the correct state family generator based on a string inputted, and generates the states needed.
    This can be amended with any other state generators needed.
    state_gen: String, determines which generator to use from State_Gens.py to build states.
        Current accepted inputs:
            "Haar": Generates Haar-random states
            "Clifford": Generates Clifford stabilizer states
            "Conserve": Generates number-conserving free fermion states.
    no_states: Integer, Number of states to generate
    N: Integer, number of qubits in the system
    n: Number of fermions (only relevant for number-conserving states) (None by default)
    MAKE SURE n HAS A VALUE FOR NUMBER-CONSERVING STATES
    """
    if state_gen=="Haar":
        state_list = Haar_random_generator(no_states=no_states, N=N, print_ans=False, seed=seed) #Haar-random states
    elif state_gen=="Clifford":
        state_list = Clifford_stab_generator_v3(no_states=no_states, N=N, seed=seed)#Clifford states
    elif state_gen=="Conserve":
        state_list = number_conserve_state_generator(no_states=no_states, N=N, n=n, print_ans=False, seed=seed) #Number-conserving states
    else:
        return("Invalid state generator.") #Safety check
    return state_list
    
#================================================================
#             dS2 VALUE CALCULATOR (TO USE FOR PLOTS)
#================================================================

def dS2_vs_m_values(N, qubits_kept, no_trials, state_gen, m_vals, seed=None, std=True):
    """
    This function generates the values of the mean single-state entropy, the superposition entropy, and their
    difference dS2 for all values of m in m_vals. These values can be used to plot each quantity as desired.
    Returns the values of  (in order): The mean single state-entropy, its standard deviation, the superposition entropy, its deviation,
    dS2, its deviation, for all values of m. (i.e. each one of these is a list, containing their values for all values of m tested)
    N: Integer, number of qubits in system
    qubits_kept: List, contains integers for qubits in the tested subsystem (counting from 0)
    no_trials: Integer, number of random state samples tested.
    state_gen: String, determines which generator to use from State_Gens.py to build states.
        Current accepted inputs:
            "Haar": Generates Haar-random states
            "Clifford": Generates Clifford stabilizer states
            "Conserve": Generates number-conserving free fermion states.
    m_vals: List, contains integers of all m values to take.
    seed: Integer, used for reproducibility (None by default)
    std: Boolean, calculates standard deviations too if True (True by default) (returns empty lists otherwise)
    """
    dS2_vals = [] #List for average dS2 results
    indiv_S2_vals = [] #List for mean entropy of individual state
    superpos_S2_vals = [] #List for equal-weight superposition entropy vals
    m_max = max(m_vals) #Maximum m value
    for _ in range(no_trials):
        #Optional check: if seed non-empty, add 1 to seed each instance (allows reproducibility, ensures each trial has distinct states)
        if seed!=None:
            seed+=1
        dS2_trial_vals = [] #List for all dS2 trial values
        m_indiv_S2_trials = [] #List for all mean S2 trial values
        m_superpos_S2_trials = [] #List for all superposition S2 trial values
        state_sample = smart_state_generator(state_gen=state_gen, no_states=m_max, N=N, n=N//2) 
        #State sample = m_max states generated once per trial (more efficient/useful than regenerating whole list for each m value)
        np.random.shuffle(state_sample) #Makes sure no state is oversampled
        for m in m_vals: #Loop repeated for each step between m_min, m_max
            m_val_states = state_sample[:m] #First m terms of state_sample used for test
            dS2_m, superpos_S2, indiv_S2 = Rényi_plot_vals_one_step(N=N, state_list=m_val_states, 
                                                               qubits_kept=qubits_kept, print_ans=False)
            #Adding each step with m to the trial results
            dS2_trial_vals.append(dS2_m)
            m_superpos_S2_trials.append(superpos_S2)
            m_indiv_S2_trials.append(indiv_S2)
        #Adding the trial results to the overall results
        dS2_vals.append(dS2_trial_vals)
        superpos_S2_vals.append(m_superpos_S2_trials)
        indiv_S2_vals.append(m_indiv_S2_trials)
    #Finding the means/standard deviations of all trial results:
    #dS2 Means and deviations
    dS2_array = np.array(dS2_vals) #Shape (no_trials, len(m_vals))
    dS2_mean = np.mean(dS2_array, axis=0) #Mean of each column of dS2 array (i.e. the mean dS2 per value of m)
    dS2_stds = np.std(dS2_array, axis=0) if std else None #dS2 standard deviations per value of m
    #Mean and deviation of individual entropies
    indiv_S2_array = np.array(indiv_S2_vals) #Shape (no_trials, len(m_vals))
    indiv_S2_mean = np.mean(indiv_S2_array, axis=0) 
    indiv_S2_stds = np.std(indiv_S2_array, axis=0) if std else None #individual mean entropies standard deviations per value of m
    #Mean and deviation of superposition entropy
    superpos_S2_array = np.array(superpos_S2_vals)
    superpos_S2_mean = np.mean(superpos_S2_array, axis=0)
    superpos_S2_stds = np.std(superpos_S2_array, axis=0) if std else None #Superposition standard deviations per value of m
    return indiv_S2_mean, indiv_S2_stds, superpos_S2_mean, superpos_S2_stds, dS2_mean, dS2_stds

#================================================================
#                       PLOTTING FUNCTIONS
#================================================================

def dS2_vs_m_plot_v3(N, qubits_kept, state_gen, m_min, m_max, m_step, no_trials, seed=None, std=True):
    """
    THIS IS NOT USED IN THE FINAL REPORT.
    For a given method of state generation, this function plots the mean single-state entropy and the equal-weight
    superposition Rényi-2 entropies against the number of states m on a plot on the left, and the difference between the two against m on a plot on the right.
    N: Integer, number of qubits
    qubits_kept: List of integers, denote to qubits to keep in the subsystem
    state_gen: String, determines which generator to use from State_Gens.py to build states.
        Current accepted inputs:
            "Haar": Generates Haar-random states
            "Clifford": Generates Clifford stabilizer states
            "Conserve": Generates number-conserving free fermion states.
    m_min, m_max: Integers, denote the minimum/maximum no of states taken from sample
    m_step: Integer, number of states added at each interval
    no_trials: Integer, determines how many sets of random states are generated per step of m
    seed: Integer, seed for random sampling from state_list (None by default)
    std: Boolean, adds standard deviation region to plot if True (True by default)
    """
    m_vals = np.arange(m_min, m_max+1, m_step) #m_max+1 since np.arange gives [start, stop)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5)) #Subplots (ax1 plots mean indiv and superposition entropies, ax2 plots dS2)
    indiv_S2_mean, indiv_S2_stds, superpos_S2_mean, superpos_S2_stds, dS2_mean, dS2_stds = dS2_vs_m_values(N=N, 
        qubits_kept=qubits_kept, no_trials=no_trials, state_gen=state_gen, m_vals=m_vals, seed=seed, std=std) #Finding the values for everything
    #Plot Everything
    ax2.plot(m_vals, dS2_mean, label="$\Delta S_2$", color="green") #dS2 values
    ax1.plot(m_vals, superpos_S2_mean, label=r"$S_2(m)$", color="red") #Plot of superposition entropies
    ax1.plot(m_vals, indiv_S2_mean, label=r"$\langle S_2(1) \rangle $", color="blue") #Plot of indiv entropies
    if std:
        ax2.fill_between(m_vals, np.add(dS2_mean, dS2_stds), np.subtract(dS2_mean, dS2_stds), color="green", alpha=0.2)
        #Standard deviation of dS2 above
        ax1.fill_between(m_vals, np.add(indiv_S2_mean, indiv_S2_stds), np.subtract(indiv_S2_mean, indiv_S2_stds), 
                     color="blue", alpha=0.2) #Standard deviation of mean entropy of individual states 
        ax1.fill_between(m_vals, np.add(superpos_S2_mean, superpos_S2_stds), np.subtract(superpos_S2_mean, superpos_S2_stds), 
                     color="red", alpha=0.2) #Standard deviation of equal weight superposition states 
    #Labelling axes:
    ax1.set_xlabel("m (Number of States)")
    ax2.set_xlabel("m (Number of States)")
    ax1.set_ylabel("Entanglement Entropy")
    ax2.set_ylabel("Entanglement Entropy")
    ax1.set_title("Components of $\Delta S_2(m)$")
    ax2.set_title("$\Delta S_2(m)$")
    #Grids for both
    ax1.grid()
    ax2.grid()
    ax1.legend()
    plt.show()
    
def comparison_dS2_for_all_state_classes(N, qubits_kept, m_min, m_max, m_step, no_trials, seed=None, std=True):
    """
    This function plots <S_2(1)>, S_2(m) and dS_2(m) for all 3 state families, with each quantity in a 
    separate subplot (and each subplot containing all 3 state families at once.)
    This can be amended with more classes if desired.
    N: Integer, total number of qubits in system
    qubits_kept: List, contains INTEGERS of qubits not to be traced out for reduced density matrices when calculating
    m_min, m_max: Integers, denote the minimum/maximum no of states taken from sample
    m_step: Integer, number of states added at each interval
    no_trials: Integer, determines how many trials of random states are generated per step of m
    seed: Integer, seed for random sampling from state_list (None by default)
    std: Boolean, include standard deviations for everything if True (True by default)
    """
    state_classes = ["Haar", "Clifford", "Conserve"] #Can be modified with more state classes if needed.
    colours = ["blue", "red", "green"] #Must have at least the same amount of elements as state_classes
    state_names = ["Haar-Random", "Clifford", "Number-Conserving"] #Must have at least the same amount of elements as state_classes
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,12))
    #ax1 plots the mean single-state entropy, ax2 plots the superposition entropy, ax3 plots dS2
    m_vals = np.arange(m_min, m_max+1, m_step) #m_max+1 since np.arange gives [start, stop)
    for state_class, colour, state_name in zip(state_classes, colours, state_names):
        #Get all values for each state class.
        indiv_S2_mean, indiv_S2_stds, superpos_S2_mean, superpos_S2_stds, dS2_mean, dS2_stds = dS2_vs_m_values(N=N, 
            qubits_kept=qubits_kept, no_trials=no_trials, state_gen=state_class, m_vals=m_vals, seed=seed, std=True)
        #Plot the means of everything
        ax1.plot(m_vals, indiv_S2_mean, label = state_name, color=colour) #Mean single-state entropy (1st plot)
        ax2.plot(m_vals, superpos_S2_mean, label = state_name, color=colour) #Superposition entropy (1st plot)
        ax3.plot(m_vals, dS2_mean, label=state_name, color=colour) #dS2 values
       #Plots standard deviations only if std==True:
        if std:
            ax1.fill_between(m_vals, np.add(indiv_S2_mean, indiv_S2_stds), np.subtract(indiv_S2_mean, indiv_S2_stds), color=colour, alpha=0.2)
            ax2.fill_between(m_vals, np.add(superpos_S2_mean, superpos_S2_stds), np.subtract(superpos_S2_mean, superpos_S2_stds), color=colour, alpha=0.2)
            ax3.fill_between(m_vals, np.add(dS2_mean, dS2_stds), np.subtract(dS2_mean, dS2_stds), color=colour, alpha=0.2)
    #Labelling plot and adding grids/legends
    ax1.set_ylabel(r"$\langle S_2(1) \rangle$ (Mean Single-State Entropy)")
    ax2.set_ylabel(r"$S_2(m)$ (Superposition Entropy)")
    ax3.set_xlabel("m (Number of States)")
    ax3.set_ylabel("$\Delta S_2(m)$")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    
def Haar_vs_clifford_plot(N, qubits_kept, no_trials, m_min, m_max, m_step, seed=None, title=None, std=False):
    """
    This function is used to directly compare the Haar-random single-state entropy and the Clifford stabilizer superposition entropy.
    N: Integer, number of qubits
    qubits_kept: List, contains integers of qubits in the subsystem kept
    no_trials: Integer, number of lists of random states tested (m_max states generated each time)
    m_min, m_max: Integers, minimum and maximum number of states taken from sample
    m_step: Integer, determines how size of state samples scales (m_step added each time)
    seed: Integer, used for reproducibility (None by default)
    title: String, gives the plot a title (None by default)
    std: Boolean, includes standard deviations if True (False by default)
    """
    m_vals = np.arange(m_min, m_max+1, m_step)
    #Haar-random single-state plot
    H_indiv_S2_mean, H_indiv_S2_stds, _, _, _, _ = dS2_vs_m_values(N=N, 
        qubits_kept=qubits_kept, no_trials=no_trials, state_gen="Haar", m_vals=m_vals, seed=seed, std=True) #Mean/deviation of Haar-random single-state entropy.
    plt.plot(m_vals, H_indiv_S2_mean, label=r"Haar-random Single-State $S_2$", color="blue") #Plot mean of Haar-random single-state entropy
    #Clifford superposition plot
    _, _, Cliff_superpos_mean, Cliff_superpos_stds, _, _ = dS2_vs_m_values(N=N, 
        qubits_kept=qubits_kept, no_trials=no_trials, state_gen="Clifford", m_vals=m_vals, seed=seed, std=True) #Mean/deviation of Clifford superposition entropy.
    plt.plot(m_vals, Cliff_superpos_mean, label=r"Clifford Superposition $S_2$", color="red") #Plot mean of Clifford superposition entropy.
    if std: #Plot deviations of both if std==True
        plt.fill_between(m_vals, np.add(H_indiv_S2_mean, H_indiv_S2_stds), np.subtract(H_indiv_S2_mean, H_indiv_S2_stds), color="blue", alpha=0.2)
        plt.fill_between(m_vals, np.add(Cliff_superpos_mean, Cliff_superpos_stds), np.subtract(Cliff_superpos_mean, Cliff_superpos_stds), color="red", alpha=0.2)
    plt.xlabel("m (Number of States)")
    plt.ylabel("Entanglement Entropy")
    if title!=None:
        plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    
def dS2_vs_m_N_comparison(N_vals, qubits_kept_list, state_gen, m_min, m_max, m_step, no_trials,
                                   seed=None, std=True, subplots=False):
        """
        For a given state generation, this function plots <S_2(1)>, S_2(m) and dS_2(m) for multiple values of N.
        This function will can either have separate plots for each quantity, or have them as subplots in a singular plot.
        N_vals: Lists, contains integers for each system size studied.
        state_gen: String, determines which generator to use from Part2v2 to build states.
            Current accepted inputs:
                "Haar": Generates Haar-random states
                "Clifford": Generates Clifford stabilizer states
                "Conserve": Generates number-conserving free fermion states.
        qubits_kept_list: List, contains multiple LISTS, each of which contains the integers kept when finding the reduced
            density matrix. Note 0=1st qubit, 1=2nd qubit, etc, and that the order of the lists must match the order of the
            numbers in N_vals (e.g. the first list of qubits_kept corresponds to the first value of N_vals taken, etc.)
        m_min, m_max: Integers, denote the minimum/maximum no of states taken from sample
        m_step: Integer, number of states added at each interval
        no_trials: Integer, determines how many sets of random states are generated per step of m
        seed: Integer, seed for random sampling from state_list
        std: Boolean, plots standard deviation if True (True by default)
        subplots: Boolean, plots each quantity as subplots on a single plot if True (otherwise each plot is returned separately)
        """
        colours=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] #Colours listed here to ensure means/deviations match (this list must have more entries than number of N vals, or it breaks!)
        m_vals = np.arange(m_min, m_max+1, m_step) #m_max+1 since np.arange gives [start, stop)
        if subplots:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.5,15)) #Subplots in the same figure(ax1 plots mean indiv and superposition entropies, ax2 plots dS2)
        else: #Used to get separate plots for report.
            fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 5))
            fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 5))
            fig3, ax3 = plt.subplots(1, 1, figsize=(7.5, 5))
        #Iterating for each value of N given:
        for i, N in enumerate(N_vals): #Testing each value of N, i is the index of N_vals
            indiv_S2_mean, indiv_S2_stds, superpos_S2_mean, superpos_S2_stds, dS2_mean, dS2_stds = dS2_vs_m_values(N=N, 
                qubits_kept=qubits_kept_list[i], no_trials=no_trials, state_gen=state_gen, m_vals=m_vals, seed=seed, std=std)
            #Plot Everything
            ax3.plot(m_vals, dS2_mean, label=f"N={N}", color=colours[i]) #dS2 values
            ax2.plot(m_vals, superpos_S2_mean, label=f"N={N}", color=colours[i]) #Plot of superposition entropies
            ax1.plot(m_vals, indiv_S2_mean, label=f"N={N}", color=colours[i]) #Plot of indiv entropies
            if std:
                ax3.fill_between(m_vals, np.add(dS2_mean, dS2_stds), np.subtract(dS2_mean, dS2_stds), color=colours[i], alpha=0.2)
                #Standard deviation of dS2 above
                ax1.fill_between(m_vals, np.add(indiv_S2_mean, indiv_S2_stds), np.subtract(indiv_S2_mean, indiv_S2_stds), 
                            color=colours[i], alpha=0.2) #Standard deviation of mean entropy of individual states 
                ax2.fill_between(m_vals, np.add(superpos_S2_mean, superpos_S2_stds), np.subtract(superpos_S2_mean, superpos_S2_stds), 
                         color=colours[i], alpha=0.2) #Standard deviation of equal weight superposition states 
        #Labelling axes:
        ax1.set_xlabel("m (Number of States)")
        ax2.set_xlabel("m (Number of States)")
        ax3.set_xlabel("m (Number of States)")
        ax1.set_ylabel(r"$\langle S_2(1) \rangle$ (Mean Single-State Entropy)")
        ax2.set_ylabel(r"$S_2(m)$ (Superposition Entropy)")
        ax3.set_ylabel("$\Delta S_2(m)$")
        #Grids and legends for all
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()
        
def dS2_vs_ln_m(N, qubits_kept, state_gen, m_min, m_max, m_step, no_trials, seed=None, std=True):
        """
        This function plots dS2 ONLY, and plots this against ln(m). Similar to dS_vs_m_v3, but only with one plot.
        Used to show number-conserving states do not have logarithmic growth.
        N: Integer, number of qubits
        qubits_kept: List of integers, denote to qubits to keep in the subsystem
        state_gen: String, determines which generator to use from State_Gens.py to build states.
            Current accepted inputs:
                "Haar": Generates Haar-random states
                "Clifford": Generates Clifford stabilizer states
                "Conserve": Generates number-conserving free fermion states.
        m_min, m_max: Integers, denote the minimum/maximum no of states taken from sample
        m_step: Integer, number of states added at each interval
        no_trials: Integer, determines how many sets of random states are generated per step of m
        seed: Integer, seed for random sampling from state_list (None by default)
        std: Boolean, adds standard deviation region to plot if True (True by default)
        """
        m_vals = np.arange(m_min, m_max+1, m_step) #m_max+1 since np.arange gives [start, stop)
        _, _, _, _, dS2_mean, dS2_stds = dS2_vs_m_values(N=N, qubits_kept=qubits_kept, no_trials=no_trials, state_gen=state_gen, m_vals=m_vals, seed=seed, std=std) #dS2 mean and deviation
        #Plot Everything
        plt.plot(np.log(m_vals), dS2_mean, label="$\Delta S_2$", color="#1f77b4") #dS2 values
        if std:
            plt.fill_between(np.log(m_vals), np.add(dS2_mean, dS2_stds), np.subtract(dS2_mean, dS2_stds), color="#1f77b4", alpha=0.2)
            #Standard deviation of dS2 above
        #Labelling axes:
        plt.xlabel("ln(m)")
        plt.ylabel("$\Delta S_2(m)$")
        plt.grid()
        plt.show()

#================================================================
#                               PLOTS
#================================================================

#S vs m System Size Study: Plots Haar-random, Clifford and number-conserving states separately, each for N=4,6,8,10, with 500 trials and m_max=50
#(number-conserving states take half-filling, i.e. n=N/2)
if S_vs_m_system_size_study: #These plots take a while.
    seed_1 = 42 #seed_1=42 used for report.
    #Haar-random plot
    dS2_vs_m_N_comparison(N_vals=[4,6,8,10], qubits_kept_list=[[0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4]],
        state_gen="Haar", m_min=1, m_max=50, m_step=1, no_trials=500, std=True, seed=seed_1, subplots=True)
    #Clifford plot
    dS2_vs_m_N_comparison(N_vals=[4,6,8,10], qubits_kept_list=[[0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4]],
        state_gen="Clifford", m_min=1, m_max=50, m_step=1, no_trials=500, std=True, seed=seed_1, subplots=False)
    #Number-conserving plot
    dS2_vs_m_N_comparison(N_vals=[4,6,8,10], qubits_kept_list=[[0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4]],
        state_gen="Conserve", m_min=1, m_max=50, m_step=1, no_trials=500, std=True, seed=seed_1, subplots=False)

#Haar Single State vs Clifford Superpos: This directly compares the Haar-random single-state entropy to the Clifford superposition.
#Plots both for N=8, 500 trials, up to 50 states.
if Haar_single_state_vs_Clifford_superpos:
    seed_2=42 #seed_2=42 used for report.
    Haar_vs_clifford_plot(N=8, qubits_kept=[0,1,2,3], no_trials=500, m_min=1, m_max=50, m_step=1, seed=seed_2)

#Number-Conserving Log: Plots dS2 against ln(m) for number-conserving states, with N=8, 500 trials, up to 100 states.
if number_conserve_vs_log_m:
    seed_3=42 #seed_3 = 42 used for report.
    dS2_vs_ln_m(N=8, qubits_kept=[0,1,2,3], state_gen="Conserve", m_min=1, m_max=100, m_step=1, no_trials=500, seed=seed_3, std=True)
    
#dS State Comparison: Plots <S_2(1)>, S_2(m) and dS_2(m)=S_2(m) - <S_2(1)> for all 3 state classes, with each quantity in separate subplots.
#This is for N=8, with 500 trials per family and up to 50 states.
if dS_state_comparison:
    seed_4=42 #seed_4 = 42 used for report.
    comparison_dS2_for_all_state_classes(N=8, qubits_kept=[0,1,2,3], m_min=1, m_max=50, m_step=1, no_trials=500, seed=seed_4, std=True)

