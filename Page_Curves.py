# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:22:26 2026

@author: Ben Lehany-Fee

This function contains all functions to plot the Page curves for each state family. This corresponds to Sections 3.5 and 4.1 of the report.
"""

import numpy as np
import matplotlib.pyplot as plt
from RDM_Entropy_Calcs_and_Validation import reduced_density_matrix, von_Neumann_entropy, Rényi_2_entropy
from State_Gens import Haar_random_generator, Clifford_stab_generator_v3, number_conserve_state_generator

Haar_curve_plots = False #If True, plots the von Neumann and Rényi-2 entropies of 1000 Haar-random states for N=6,8,12,16.
#Also prints the mean and standard deviation at N/2 for each N value.
conserve_curve_plots = False #If True, plots the von Neumann and Rényi-2 entropies of 1000 number-conserving states for N=6,8,12,16.
#Also prints the mean and standard deviation at N/2 for each N value.
Clifford_curve_plots = False #If True, plots the von Neumann and Rényi-2 entropies of 1000 number-conserving states for N=6,8,12,16.
#Also prints the mean and standard deviation at N/2 for each N value.
indiv_Clifford_plots = False #If True, plots the entropies of 6 individual 12-qubit Clifford stabilizer states.
S_max_comparison = False #If True, plots the von Neumann entropies of all 3 state families vs N_A ln(2) for N=8,16
peak_comparison = False #If True, plots the peaks of each Page curve for N=4,6,8,10.

#================================================================
#                  PAGE CURVE VALUE CALCULATOR
#================================================================

def page_curve_vals(N, state_list, Rényi, von_Neumann, std):
    """
    This function gives the values to plot the Page Curve of the average entanglement entropy.
    Variables are included to decide what entropies to find the value of (Rényi-2 and/or von Neumann) and to decide if standard deviations are included.
    We always split the system into two subsystems of the N_A leftmost and N-N_A rightmost qubits, taking the smallest subsystem of the two.
    This returns (in order): List of Rényi-2 mean entropies, list of von Neumann mean entropies,
    list of Rényi-2 standard deviation of the entropy (only if std==True), list of von Neumann 
    standard deviation of the entropy (only if std==True)
    All 4 lists must always be assigned as variables, but will be empty unless their corresponding
    variable is True
    N: number of qubits (subsystem sizes are taken as 1,2,...,N-1)
    state_list: List, inputted states
    Rényi: Boolean, finds the values of the Rényi-2 entropy if True
    von_Neumann: Boolean, finds the values of the von Neumann entropy if True
    std: Boolean, returns the standard deviation values of all entropies considered if True
    """
    N_A_vals = range(1, N) #1 to N-1 qubits in A
    S2_means=[] #Rényi-2 mean entropy list
    S2_stds = [] #Rényi-2 standard deviation list
    S_means = [] #von Neumann entropy list
    S_stds = [] #von Neumann standard deviation list
    for N_A in N_A_vals:
        #For optimisation, keep the qubits in the smaller subsystem (as S(rho_A)=S(rho_B))
        if N_A <= (N//2):
            qubits_kept = [i for i in range(0, N_A)] #First N_A qubits kept.
        else:
            qubits_kept = [i for i in range(N_A, N)]  
        #All relevant density matrices
        rdm_list = [reduced_density_matrix(vec=state, qubits_kept=qubits_kept,
                    N=N, normal=False, print_ans=False) for state in state_list]
        if Rényi:
            #All Rényi-2 entropies for each rdm (only calculated if Rényi==True)
            Rényi_2_list=[Rényi_2_entropy(rho, print_ans=False) for rho in rdm_list]
            S2_means.append(np.mean(Rényi_2_list)) #Take the mean entropy of all states
            S2_stds.append(np.std(Rényi_2_list)) if std else None
        if von_Neumann:
            #All von Neumann entropies for each rdm (only calculated if von_Neumann==True)
            von_Neumann_list=[von_Neumann_entropy(rho, print_ans=False) for rho in rdm_list]
            S_means.append(np.mean(von_Neumann_list))
            S_stds.append(np.std(von_Neumann_list)) if std else None
    return S2_means, S_means, S2_stds, S_stds

#================================================================
#                 PAGE CURVE PLOTTING FUNCTIONS
#================================================================

def plot_page_curve(N, state_list, Rényi=True, von_Neumann=True, errors=False, title=None, 
                    print_peak_vals=True):
    """
    Plots the Page curve of one or both entropies for a single list of states.
    N: Integer, number of qubits
    state_list: List, contains inputted states to test.
    Rényi: Boolean, plots the Rényi-2 page curve if True
    von_Neumann: Boolean, plots the von Neumann page curve if True
    errors: Boolean, plots the error lines (mean +/- standard deviation) for each plotted entropy
    title: String, title for the plot, does nothing if None (None by default)
    print_peak_vals: Boolean, if True, prints the values at the peak of each Page curve.
    """
    plt.figure(figsize=(6.4,4.2)) #Subfigures
    N_A_vals = range(1, N) #x-axis
    S2_vals, S_vals, S2_stds, S_stds = page_curve_vals(N, state_list, Rényi, von_Neumann, std=errors)
    if Rényi:
        plt.plot(N_A_vals, S2_vals, label="Rényi-2", color="blue") #Plotting average Rényi
        if errors:
            S2_error_up = np.add(S2_vals, S2_stds)
            S2_error_down = np.subtract(S2_vals, S2_stds)
            plt.fill_between(N_A_vals, S2_error_down, S2_error_up, color="blue", alpha=0.2)
        #Print S2's mean and (OPTIONALLY) std dev at N_A=N//2
        if print_peak_vals:
            print("S2 Mean: ", S2_vals[N//2-1])
            if errors:
                print("S2 Deviation: ", S2_stds[N//2-1])
    if von_Neumann:
        plt.plot(N_A_vals, S_vals, label="von Neumann", color="red") #Plotting average von Neumann
        if errors:
            S_error_up = np.add(S_vals, S_stds)
            S_error_down = np.subtract(S_vals, S_stds)
            plt.fill_between(N_A_vals, S_error_down, S_error_up, color="red", alpha=0.2)
        #Print S2's mean and (OPTIONALLY) std dev at N_A=N//2
        if print_peak_vals:
            print("S Mean: ", S_vals[N//2-1])
            if errors:
                print("S Deviation: ", S_stds[N//2-1])
    plt.xlabel("$N_A$")
    plt.ylabel("Entanglement Entropy")
    plt.grid()
    plt.legend()
    if title!=None: #Optional Title
        plt.title(title)
    plt.show()
    
def page_curve_state_comparison_plot(N, list_of_families, family_names, Rényi=True, 
                                     von_Neumann=True, S_max=False, title=None, label_entropy=False):
    """
    Plots the Page Curves of multiple separate state families into a single plot for a single system size N.
    This function also has the option to plot N_A ln(2) to compare these entropies to the theoretical maximum.
    N: Integer, number of qubits
    list_of_families: List, contains multiple LISTS, each list is treated as a single state family to test and plot
    family_names: List, contains multiple STRINGS, each string will be the label of the plot made by its corresponding
    family.
    Rényi: Boolean, plots Rényi-2 entropy for ALL families if True
    von Neumann: Boolean, plots von Neumann entropy for ALL families if True
    S_max: Boolean, plots N_A ln(2) if True
    title: String, gives title to the plot (None by default)
    label_entropy: Boolean, adds specified entropy names to labels on the plot if True (False by default)
    """
    plt.figure(figsize=(6.4,4.2)) #Subplots
    N_A_vals = range(1, N)
    for i, state_list in enumerate(list_of_families):
        S2_vals, S_vals, S2_stds, S_stds = page_curve_vals(N, state_list, Rényi, von_Neumann, std=False) #Mean/deviation values for both entropies
        if Rényi:
            plt.plot(N_A_vals, S2_vals, label=family_names[i]+(": Rényi-2" if label_entropy else "")) #Plotting average Rényi
        if von_Neumann:
            plt.plot(N_A_vals, S_vals, label=family_names[i]+(": von Neumann" if label_entropy else "")) #Plotting average von Neumann
    #Plotting N_A ln(2) if S_max==True
    if S_max:
        S_max_vals = [min(N_A, N-N_A)*np.log(2) for N_A in N_A_vals] #N_A ln(2), always takes N_A as smaller subsystem
        plt.plot(N_A_vals, S_max_vals, label="$S_{max}=N_A$ ln(2)", ls="--")
    plt.xlabel("$N_A$")
    plt.ylabel("Entanglement Entropy")
    plt.grid()
    plt.legend(loc='lower center')
    if title!=None:
        plt.title(title)
    plt.show()

def page_curve_indiv_state_entropy_comparison_subplots(N, state_list, title=None):
    """
    This function plots the Page Curves of both the Rényi-2 and von Neumann entropies of individual states 
    splitting the results into two plots based on which entropy is measured.
    This function is only used for the Clifford stabilizer states, as they overlap exactly.
    N: Integer, number of qubits
    state_list: List, contains arrays of all states to measure the entropies of (small lists are
    recommended, or your graph will be very cluttered)
    title: String, gives the plot a title (None by default)
    """
    N_A_vals = range(1,N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5)) #Subplots
    for i, state in enumerate(state_list): #Runs through each state
        S2_vals, S_vals, S2_stds, S_stds = page_curve_vals(N, [state], Rényi=True, 
            von_Neumann=True, std=False) #Finds the 'mean' entropies of one state, i.e. just its entropy
        #Plotting the entropies of each state on separate subplots
        ax1.plot(N_A_vals, S2_vals, alpha=0.7, label=f"State {i+1}")
        ax2.plot(N_A_vals, S_vals,  alpha=0.7, label=f"State {i+1}")
    #Labelling subplots
    ax1.set_xlabel("$N_A$")
    ax1.set_ylabel("Entanglement Entropy")
    ax1.set_title("Rényi-2")
    ax1.grid()
    ax2.set_xlabel("$N_A$")
    ax2.set_ylabel("Entanglement Entropy")
    ax2.set_title("von Neumann")
    ax2.grid()
    ax1.legend()
    ax2.legend()
    if title!=None:
        plt.suptitle(title)
    plt.show()
    
def fin_size_study(N_vals, no_states, seed=None, title=None):
    """
    This function plots the peak entropies (at N/2) for each state family, and compares them to the
    theoretical maximum N_A ln(2).
    N_vals: List, contains INTEGERS for each system size N to measure.
    no_states: Integer, Number of states generated per value of N to average across.
    seed: Integer, used for reproducibility (None by default)
    title: Strings, gives plot a title (None by default)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5)) #Subplots
    #Empty lists to store mean/deviations of each state family
    Haar_S2_max, Haar_S_max = [], []
    Cliff_S2_max, Cliff_S_max = [], []
    conserve_S2_max, conserve_S_max = [], []
    for N in N_vals:
        #Haar states
        Haar_states = Haar_random_generator(no_states=no_states, N=N, seed=seed) #Haar-random state sample
        Haar_S2, Haar_S, _, _ = page_curve_vals(N=N, state_list=Haar_states, Rényi=True, von_Neumann=True, std=False) #Haar-random mean values for N
        #Appending values at N/2 to results
        Haar_S2_max.append(Haar_S2[N//2 - 1]) #-1 due to Python indexing
        Haar_S_max.append(Haar_S[N//2 - 1])
        #Clifford states
        Cliff_states = Clifford_stab_generator_v3(no_states=no_states, N=N, seed=seed) #Clifford state sample
        Cliff_S2, Cliff_S, _, _ = page_curve_vals(N=N, state_list=Cliff_states, Rényi=True, von_Neumann=True, std=False) #Clifford mean values for N
        #Appending values at N/2 to results
        Cliff_S2_max.append(Cliff_S2[N//2 - 1])
        Cliff_S_max.append(Cliff_S[N//2 - 1])
        #Number-conserving states
        conserve_states = number_conserve_state_generator(no_states=no_states, N=N, n=N//2, seed=seed) #Number-conserving state sample
        conserve_S2, conserve_S, _, _ = page_curve_vals(N=N, state_list=conserve_states, Rényi=True, von_Neumann=True, std=False)#Number-conserving mean values for N
        #Appending values at N/2 to results
        conserve_S2_max.append(conserve_S2[N//2 - 1])
        conserve_S_max.append(conserve_S[N//2 - 1])
    #Plotting everything
    S_max_vals = [(N//2)*np.log(2) for N in N_vals]
    #Plotting maximum S=N_A ln(2)
    ax1.plot(N_vals, S_max_vals, label="$S_{max}=N_A \ln(2)$", color="red", ls="--")
    ax2.plot(N_vals, S_max_vals, label="$S_{max}=N_A \ln(2)$", color="red", ls="--")
    #von Neumann plot for all families
    ax2.plot(N_vals, Haar_S_max, label="Haar-Random")
    ax2.plot(N_vals, Cliff_S_max, label="Cliffords")
    ax2.plot(N_vals, conserve_S_max, label="Number-Conserving")
    #Rényi-2 plot for all families
    ax1.plot(N_vals, Haar_S2_max, label="Haar-Random")
    ax1.plot(N_vals, Cliff_S2_max, label="Cliffords")
    ax1.plot(N_vals, conserve_S2_max, label="Number-Conserving")
    #Labelling axes and applying grid/legend
    ax1.set_title("Rényi-2 Entropy:")
    ax2.set_title("von Neumann Entropy:")
    ax1.set_xlabel("N (System Size)")
    ax2.set_xlabel("N (System Size)")
    ax1.set_ylabel("Entanglement Entropy")
    ax2.set_ylabel("Entanglement Entropy")
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    #Optional Title
    if title!=None:
        plt.suptitle(title)
    plt.show()

#================================================================
#                               PLOTS
#================================================================

#Haar-Curve Plots: Plots the Haar page curve for N=6,8,12,16, and gives the mean/standard deviation of both entropies at N/2
if Haar_curve_plots:
    hcp_seed=42 #hcp_seed=42 used for report plots (the meaning to life, the universe and everything)
    print("6-qubit values at N_A=3")
    Haar_states_6 = Haar_random_generator(no_states=1000, N=6, seed=hcp_seed)
    plot_page_curve(N=6, state_list=Haar_states_6, Rényi=True, von_Neumann=True, errors=True, 
                    title="Haar-Random States: N=6", print_peak_vals=True) #6-qubit plot
    print("\n", "8-qubit values at N_A=4")
    Haar_states_8 = Haar_random_generator(no_states=1000, N=8, seed=hcp_seed)
    plot_page_curve(N=8, state_list=Haar_states_8, Rényi=True, von_Neumann=True, errors=True, 
                    title="Haar-Random States: N=8", print_peak_vals=True) #8-qubit plot
    print("\n", "12-qubit values at N_A=6")
    Haar_states_12 = Haar_random_generator(no_states=1000, N=12, seed=hcp_seed)
    plot_page_curve(N=12, state_list=Haar_states_12, Rényi=True, von_Neumann=True, errors=True, 
                    title="Haar-Random States: N=12", print_peak_vals=True) #12-qubit plot
    print("\n", "16-qubit values at N_A=8")
    Haar_states_16 = Haar_random_generator(no_states=1000, N=16, seed=hcp_seed)
    plot_page_curve(N=16, state_list=Haar_states_16, Rényi=True, von_Neumann=True, errors=True, 
                    title="Haar-Random States: N=16", print_peak_vals=True) #No errors since it's not visible for 16 qubits.
    print("\n")
    
#Conserve Curve Plots: Plots the Haar page curve for N=6,8,12,16, and gives the mean/standard deviation of both entropies at N/2
if conserve_curve_plots:
    ccp_seed=42 #ccp_seed=42 used for report plots
    print("6-qubit values at N_A=3")
    conserve_states_6 = number_conserve_state_generator(no_states=1000, N=6, n=3, seed=ccp_seed)
    plot_page_curve(N=6, state_list=conserve_states_6, Rényi=True, von_Neumann=True, errors=True, 
                    title="Number-Conserving States: N=6, n=3") #6-qubit plot
    print("\n", "8-qubit values at N_A=4")
    conserve_states_8 = number_conserve_state_generator(no_states=1000, N=8, n=4, seed=ccp_seed)
    plot_page_curve(N=8, state_list=conserve_states_8, Rényi=True, von_Neumann=True, errors=True, 
                    title="Number-Conserving States: N=8, n=4") #8-qubit plot
    print("\n", "12-qubit values at N_A=6")
    conserve_states_12 = number_conserve_state_generator(no_states=1000, N=12, n=6, seed=ccp_seed)
    plot_page_curve(N=12, state_list=conserve_states_12, Rényi=True, von_Neumann=True, errors=True, 
                    title="Number-Conserving States: N=12, n=6") #12-qubit plot
    print("\n", "16-qubit values at N_A=8")
    conserve_states_16 = number_conserve_state_generator(no_states=1000, N=16, n=8, seed=ccp_seed)
    plot_page_curve(N=16, state_list=conserve_states_16, Rényi=True, von_Neumann=True, errors=True, 
                    title="Number-Conserving States: N=16, n=8") #No errors since it's not visible for 16 qubits.
    print("\n")
    
#Clifford Curve Plots: Plots the Clifford Page curves for N=6,8,12,16, and gives the mean/standard deviation of both entropies at N/2
if Clifford_curve_plots:
    Cliff_seed = 42 #Cliff_seed=42 used for report plots
    print("6-qubit values at N_A=3")
    Clifford_sample_6 = Clifford_stab_generator_v3(no_states=1000, N=6, seed=Cliff_seed) #Used for mean entropy plot, 6 qubits
    plot_page_curve(N=6, state_list=Clifford_sample_6, Rényi=True, von_Neumann=True, errors=True, 
                    title="Clifford Stabilizer States: N=6") #6-qubit plot
    print("\n", "8-qubit values at N_A=4")
    Clifford_sample_8 = Clifford_stab_generator_v3(no_states=1000, N=8, seed=Cliff_seed) #Used for mean entropy plot, 6 qubits
    plot_page_curve(N=8, state_list=Clifford_sample_8, Rényi=True, von_Neumann=True, errors=True, 
        title="Clifford Stabilizer States: N=8") #8-qubit plot
    print("\n", "12-qubit values at N_A=6")
    Clifford_sample_12 = Clifford_stab_generator_v3(no_states=1000, N=12, seed=Cliff_seed) #Used for mean entropy plot, 12 qubits
    plot_page_curve(N=12, state_list=Clifford_sample_12, Rényi=True, von_Neumann=True, errors=True, 
                    title="Clifford Stabilizer States: N=12") #12-qubit plot
    #This plot in particular takes a while, but does work.
    print("\n", "16-qubit values at N_A=8")
    Clifford_sample_16 = Clifford_stab_generator_v3(no_states=1000, N=16, seed=Cliff_seed) #Used for mean entropy plot, 16 qubits
    plot_page_curve(N=16, state_list=Clifford_sample_16, Rényi=True, von_Neumann=True, errors=True, 
                    title="Clifford Stabilizer States: N=16") #16-qubit plot
    print("\n")
    
#Indiv Clifford Plots: Plots the individual entropies of a 6 Clifford stabilizer states, for N=12
if indiv_Clifford_plots:
    seed_0=48 #seed_0=48 used in report
    Clifford_sample_indiv = Clifford_stab_generator_v3(no_states=6, N=12, seed=seed_0) #Used for individual entropies
    page_curve_indiv_state_entropy_comparison_subplots(N=12, state_list=Clifford_sample_indiv) #Indiv plot

#S_max Comparison: This tests plots the entanglement entropies of 200 of each state family and compares
#them to the theoretical maximum N_A ln(2). 
#Here, we specifically only consider the von Neumann entropies, as for each state family the Rényi-2 entropy is 
#less than or equal to the von Neumann entropy.
if S_max_comparison:
    seed_1=42 #seed_1=42 used in report
    #First plot, compares the Page Curves of samples of 100 states from each state family in a 8-qubit system. (including N_A ln(2))
    state_samples_1 = [Haar_random_generator(no_states=500, N=8, seed=seed_1, print_ans=False),
                     Clifford_stab_generator_v3(no_states=500, N=8, print_ans=False, seed=seed_1),
                     number_conserve_state_generator(no_states=500, N=8, n=4, seed=seed_1)]
    page_curve_state_comparison_plot(N=8, list_of_families = state_samples_1, 
                                     family_names=["Haar-Random", "Clifford", "Number-Conserving"],
                                     Rényi=False, von_Neumann=True, S_max=True, title="Page Curves vs Maximum Entropy (von Neumann): N=8",
                                     label_entropy=False)
    #Second plot, does the same for 16 qubits.
    state_samples_2 = [Haar_random_generator(no_states=500, N=16, seed=seed_1, print_ans=False),
                     Clifford_stab_generator_v3(no_states=500, N=16, print_ans=False, seed=seed_1),
                     number_conserve_state_generator(no_states=500, N=16, n=8, seed=seed_1)]
    page_curve_state_comparison_plot(N=16, list_of_families = state_samples_2, 
                                     family_names=["Haar-Random", "Clifford", "Number-Conserving"],
                                     Rényi=False, von_Neumann=True, S_max=True, title="Page Curve vs Maximum Entropy (von Neumann): N=16",
                                     label_entropy=False)
    
#Finite Size Study: This test compares the 'peak' of each entanglement distribution (i.e. taking N/2)
if peak_comparison:
    seed_2=42 #seed_2=42 used for report
    fin_size_study(N_vals=[4,6,8,10], no_states=1000, seed=seed_2)
