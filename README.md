Entanglement in Superpositions of Quantum Many-Body States:

This is my final year undergraduate project carried out in Maynooth University. For this project, I studied the behvaiour of Haar-random states, Clifford stabilizer states and number-conserving states under equal-weight superpositions, plotting Page curves for both the Rényi-2 and von Neumann entropies. 
I also investigated the difference between the Rényi-2 entropies of the equal weight superposition versus the average single-state entropy.

Languages and Libraries:

Python

Numpy

Scipy

Matplotlib

Stim

Qiskit

Install all dependencies with:
pip install numpy scipy matplotlib qiskit stim

Project Structure:
File 1: RDM_Entropy_Cals_and_Validation.py

Contains a reduced density matrix, and calculators for both the Rényi-2 and von Neumann entropy. Also includes validation tests (verifying Bell states have maximal entanglement entropy, product states have zero entropy, and that any bipartition of a closed quantum system has equal entropy across both subsystems).
All validation tests can be replicated by the included Boolean variables at the top of this file.

File 2: State_Gens.py
Contains state generators for Haar-random, Clifford stabilizer and number-conserving states. v3 of the Clifford generator is the final version used.

File 3: Page_Curves.py
Computes and plots the Page curves for all 3 state families, including cross-family comparisons (including comparisons to the theoretical maximum entropy) and finite-size studies on system size.
All plots used within the report can be replicated using the Boolean variables at the top of the code.

File 4: Superposition_Analysis.py
Computes the equal-weight superposition entropy S₂(m), the mean single-state entropy ⟨S₂(1)⟩, and their difference ΔS₂(m) as a function of the number of states m, with plots comparing all three families.
All plots used within the report can be replicated using the Boolean variables at the top of the code.

Author:
Ben Lehany-Fee
Bsc Theoretical Physics and Mathematics, National University of Ireland Maynooth.
