# Representative Codes Used in This Work

This repository contains representative simulation and analysis codes used in the study of mutant-burden and rescue dynamics under different population-growth models.

The repository contains five representative codes:

1. Deterministic well-mixed system  
2. Stochastic well-mixed system  
3. Spatial stochastic system  
4. Stochastic well-mixed rescue system  
5. Spatial rescue system  

---

## 1. Deterministic Well-Mixed System

**File:** `deterministic_well_mixed.py`

**Description:**  
This Python code implements the deterministic well-mixed analytical/ODE framework used to study the dependence of mutant burden on the death-to-birth ratio \(\delta\).

**Main features:**

- Deterministic growth equations
- Finite-size stopping condition
- Phase-diagram generation
- Relative-variation analysis
- Scaling analysis of mutant burden
- Publication-quality figure generation

**Application:**  
Used to generate deterministic results and phase diagrams for the well-mixed system.

---

## 2. Stochastic Well-Mixed System

**File:** `stochastic_well_mixed.cpp`

**Description:**  
This MPI-parallelized C++ Gillespie simulation code implements a stochastic well-mixed birth-death-mutation process with extinction restarts.

**Main features:**

- Gillespie stochastic simulation
- Mutation dynamics
- Extinction-and-restart mechanism
- Parallel execution using MPI
- Large-scale sampling
- Distribution and average mutant-burden output

**Output:**

- `avg_Y.txt`
- `Y_delta_*.txt`

**Application:**  
Used to generate stochastic well-mixed mutant-burden distributions and averages presented in the main text.

---

## 3. Spatial Stochastic System

**File:** `spatial.cpp`

**Description:**  
This MPI-parallelized spatial stochastic simulation code implements a two-dimensional lattice birth-death-mutation model with spatial structure.

**Main features:**

- Two-dimensional lattice dynamics
- Moore-neighborhood interactions
- Spatial mutant expansion
- Extinction restarts
- Logarithmic checkpoint recording
- Parallel trajectory generation using MPI
- Streaming trajectory output

**Output:**

- `mutant_records_deltaXXXX_rankYYYY.txt`

**Application:**  
Used to study spatial effects on mutant-burden dynamics and stochastic jackpot behavior.

---

## 4. Stochastic Well-Mixed Rescue System

**File:** `stochastic_well_mixed_rescue.cpp`

**Description:**  
This MPI-parallelized C++ Gillespie simulation code implements a two-phase stochastic well-mixed rescue model.

In Phase I, the population grows with mutation and extinction restarts until the total population reaches a prescribed size \(n\). In Phase II, treatment is applied by instantaneously removing all wild-type cells. Therefore, only mutants remain and evolve through stochastic birth-death dynamics.

**Main features:**

- Two-phase stochastic rescue simulation
- Well-mixed Gillespie dynamics
- Extinction-and-restart mechanism in Phase I
- Instantaneous wild-type removal at the start of Phase II
- Mutant-only rescue dynamics after treatment
- MPI-parallelized independent realizations
- Rescue probability estimation

**Output:**

- `rescue_probability.txt`
- `phase1_final_mutants_delta_*.txt`

**Application:**  
Used to estimate the probability of rescue in the stochastic well-mixed system after treatment removes the wild-type population.

---

## 5. Spatial Rescue System

**File:** `spatial_rescue.cpp`

**Description:**  
This MPI-parallelized C++ simulation code implements a two-phase spatial rescue model on a two-dimensional lattice.

In Phase I, the spatial population grows with mutation and extinction restarts until the total population reaches \(n\). In Phase II, all wild-type cells are removed instantaneously, leaving only mutants on the lattice. The remaining mutant population then evolves through spatial birth-death dynamics until either rescue occurs or the mutant population goes extinct.

**Main features:**

- Two-dimensional spatial rescue dynamics
- Moore-neighborhood lattice interactions
- Mutation during spatial expansion
- Extinction-and-restart mechanism in Phase I
- Instantaneous wild-type removal at the start of Phase II
- Mutant-only spatial birth-death dynamics after treatment
- MPI-parallelized large-scale simulations
- Rescue probability calculation

**Output:**

- `rescue_probability.txt`
- `phase1_final_mutants_delta_*.txt`

**Application:**  
Used to study how spatial structure affects mutant rescue probability after treatment removes the wild-type population.

---

## Important Note

The provided files are representative versions of the codes used in this work. Minor parameter changes, optimization adjustments, and plotting scripts may have been applied for specific figures and analyses presented in the manuscript and supplementary information.
