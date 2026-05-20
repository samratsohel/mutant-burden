============================================================
REPRESENTATIVE CODES USED IN THIS WORK
============================================================

This repository contains representative simulation and
analysis codes used in the study of mutant-burden dynamics
under different population-growth models.

Three codes are there:

------------------------------------------------------------
1. DETERMINISTIC WELL-MIXED SYSTEM
------------------------------------------------------------

File:
    deterministic_well_mixed.py

Description:
    This Python code implements the deterministic
    well-mixed analytical/ODE framework used to study
    the dependence of mutant burden on the death-to-
    birth ratio δ.

Main features:
    • Deterministic growth equations
    • Finite-size stopping condition
    • Phase-diagram generation
    • Relative-variation analysis
    • Scaling analysis of mutant burden
    • Publication-quality figure generation

Application:
    Used to generate deterministic results and phase
    diagrams for the well-mixed system.

------------------------------------------------------------
2. STOCHASTIC WELL-MIXED SYSTEM
------------------------------------------------------------

File:
    stochastic_well_mixed.cpp

Description:
    This MPI-parallelized C++ Gillespie simulation code
    implements a stochastic well-mixed birth–death–
    mutation process with extinction restarts.

Main features:
    • Gillespie stochastic simulation
    • Mutation dynamics
    • Extinction-and-restart mechanism
    • Parallel execution using MPI
    • Large-scale sampling (up to 10^9 runs)
    • Distribution and average mutant-burden output

Output:
    • avg_Y.txt
    • Y_delta_*.txt

Application:
    Used to generate stochastic well-mixed mutant-burden
    distributions and averages presented in the main text.

------------------------------------------------------------
3. SPATIAL STOCHASTIC SYSTEM
------------------------------------------------------------

File:
    spatial.cpp

Description:
    This MPI-parallelized spatial stochastic simulation
    code implements a two-dimensional lattice birth–
    death–mutation model with spatial structure.

Main features:
    • Two-dimensional lattice dynamics
    • Moore-neighborhood interactions
    • Spatial mutant expansion
    • Extinction restarts
    • Logarithmic checkpoint recording
    • Parallel trajectory generation using MPI
    • Streaming trajectory output

Output:
    mutant_records_deltaXXXX_rankYYYY.txt

Application:
    Used to study spatial effects on mutant-burden
    dynamics and stochastic jackpot behavior.

------------------------------------------------------------
IMPORTANT NOTE
------------------------------------------------------------

The provided files are representative versions of the
codes used in this work. Minor modifications, parameter
changes, optimization adjustments, and plotting scripts
may have been applied for specific figures and analyses
presented in the manuscript and supplementary information.

============================================================
