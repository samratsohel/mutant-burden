#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iomanip>

// =====================================================
// PURPOSE OF THIS CODE
// =====================================================
//
// This MPI-parallelized Gillespie simulation code was
// used to generate the stochastic mutant-burden data
// presented in Fig. 1 of the main text.
//
// The code simulates a stochastic birth–death–mutation
// process with repeated extinction restarts until the
// total population reaches a prescribed size N.
//
// For each realization, the final mutant population
// Y_final is recorded at the hitting time:
//
//                  X + Y = N
//
// The simulation outputs:
//
//   (i)  The distribution of mutant burden Y_final
//        for a given death-to-birth ratio δ = τ
//
//   (ii) The average mutant burden ⟨Y⟩ as a function
//        of the death-to-birth ratio
//
// Large-scale parallel simulations (up to 10^9 runs)
// can be performed efficiently using MPI.
//
// Output files:
//
//   avg_Y.txt
//       → average mutant burden versus τ
//
//   Y_tau_*.txt
//       → full stochastic distribution of Y_final
//         for each τ value
//
// These output files can be directly used to construct
// the panels shown in Fig. 1 of the main manuscript.
// =====================================================


// =====================================================
// RANDOM NUMBER GENERATOR
// =====================================================
// Returns a uniformly distributed random number
// in the interval [0,1).
// =====================================================
double random_number() {
    return static_cast<double>(rand()) / RAND_MAX;
}


// =====================================================
// STRUCTURE TO STORE SIMULATION OUTPUT
// =====================================================
// Y_final : final mutant population when total
//           population reaches N
//
// T_N     : hitting time required to reach N
// =====================================================
struct SimulationResult {
    int Y_final;
    double T_N;
};


// =====================================================
// GILLESPIE SIMULATION WITH RESTARTS
// =====================================================
//
// This function simulates a stochastic birth–death–
// mutation process involving:
//
//   X : wild-type population
//   Y : mutant population
//
// The system evolves until:
//
//   (i)  X + Y = N      (target population reached)
//
// or
//
//   (ii) X + Y = 0      (total extinction)
//
// In the extinction case, the system is restarted
// from:
//
//      X = initial_X
//      Y = 0
//
// The simulation uses the Gillespie algorithm
// (continuous-time stochastic simulation).
// =====================================================

SimulationResult simulate_with_restarts(
    double s1,
    double s2,
    double u,
    double delta,
    int N,
    int initial_X
) {

    // Initial populations
    int X = initial_X;
    int Y = 0;

    // Continuous simulation time
    double T = 0.0;

    // Baseline birth rate
    double r = 0.5;

    // =================================================
    // MAIN GILLESPIE LOOP
    // =================================================
    while (X + Y > 0 && X + Y < N) {

        // =============================================
        // REACTION RATES
        // =============================================
        //
        // 0 : X -> X + X
        // 1 : X -> ∅
        // 2 : X -> X + Y
        // 3 : Y -> Y + Y
        // 4 : Y -> ∅
        // =============================================
        std::vector<double> rates = {

            r * (1.0 - u) * X,          // Wild-type birth

            r * delta * X,                // Wild-type death

            r * u * X,                  // Mutation event

            r * (1.0 - s1) * Y,         // Mutant birth

            r * delta * (1.0 + s2) * Y   // Mutant death
        };

        // Total reaction rate
        double a0 = 0;

        for (double rate : rates)
            a0 += rate;

        // Stop if all rates vanish
        if (a0 == 0)
            break;

        // Random numbers for Gillespie update
        double r1 = random_number();
        double r2 = random_number();

        // =============================================
        // TIME INCREMENT
        // =============================================
        // Gillespie waiting-time distribution
        // =============================================
        double dt = -std::log(r1) / a0;

        T += dt;

        // =============================================
        // REACTION SELECTION
        // =============================================
        double scaled_r2 = r2 * a0;

        size_t reaction_index = 0;

        double cumulative = 0;

        for (; reaction_index < rates.size(); ++reaction_index) {

            cumulative += rates[reaction_index];

            if (scaled_r2 <= cumulative)
                break;
        }

        // =============================================
        // APPLY REACTION
        // =============================================
        if (reaction_index == 0)
            X += 1;        // X birth

        else if (reaction_index == 1)
            X -= 1;        // X death

        else if (reaction_index == 2)
            Y += 1;        // Mutation

        else if (reaction_index == 3)
            Y += 1;        // Y birth

        else if (reaction_index == 4)
            Y -= 1;        // Y death


        // =============================================
        // RESTART AFTER TOTAL EXTINCTION
        // =============================================
        if (X + Y == 0) {

            X = initial_X;

            Y = 0;
        }
    }

    // =============================================
    // STORE OUTPUT
    // =============================================
    SimulationResult result;

    result.Y_final = Y;

    result.T_N = T;

    return result;
}


// =====================================================
// MAIN PROGRAM
// =====================================================
int main(int argc, char** argv) {

    // =============================================
    // INITIALIZE MPI
    // =============================================
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // =============================================
    // RANDOM SEED
    // =============================================
    // Each MPI rank receives a different seed.
    // =============================================
    srand(static_cast<unsigned>(time(0)) + rank);


    // =================================================
    // MODEL PARAMETERS
    // =================================================
    double s1 = 0.3;      // Mutant birth disadvantage

    double s2 = 0.0;      // Mutant death disadvantage

    double u = 1e-4;      // Mutation probability

    int N = 10000;        // Target population size

    int initial_X = 1;    // Initial wild-type count

    // Total number of stochastic realizations
    int A = 1000000000;


    // =================================================
    // WORK DISTRIBUTION ACROSS MPI RANKS
    // =================================================
    int runs_per_process = A / size;


    // =================================================
    // VALUES OF THE DEATH-TO-BIRTH RATIO τ
    // =================================================

    /*
    std::vector<double> delta_values = {
        0.00, 0.05, 0.10, 0.15, 0.20,
        0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90
    };
    */

    std::vector<double> delta_values = {
        0.55
    };


    // =================================================
    // FILE FOR AVERAGE MUTANT BURDEN
    // =================================================
    std::ofstream avg_file;

    if (rank == 0) {

        avg_file.open("avg_Y.txt");

        avg_file << "delta, Avg_Y\n";
    }


    // =================================================
    // LOOP OVER τ VALUES
    // =================================================
    for (int delta_idx = 0;
         delta_idx < (int)delta_values.size();
         ++delta_idx) {

        double delta = delta_values[delta_idx];


        // =============================================
        // LOCAL STORAGE
        // =============================================
        std::vector<int> local_Ys;

        local_Ys.reserve(runs_per_process);

        double local_sum_Y = 0;

        int local_count = 0;


        // =============================================
        // RUN STOCHASTIC SIMULATIONS
        // =============================================
        for (int i = 0; i < runs_per_process; ++i) {

            // Progress monitor from rank 0
            if (rank == 0 && (i % 1000 == 0)) {

                std::cout
                    << "delta " << delta
                    << " progress: "
                    << i
                    << " / "
                    << runs_per_process
                    << " on rank 0"
                    << std::endl;
            }

            // Run one realization
            SimulationResult res =
                simulate_with_restarts(
                    s1,
                    s2,
                    u,
                    delta,
                    N,
                    initial_X
                );

            // Store final mutant count
            local_Ys.push_back(res.Y_final);

            local_sum_Y += res.Y_final;

            local_count++;
        }


        // =============================================
        // GLOBAL REDUCTION
        // =============================================
        // Compute global average mutant burden.
        // =============================================
        double global_sum_Y = 0;

        int global_count = 0;

        MPI_Reduce(
            &local_sum_Y,
            &global_sum_Y,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &local_count,
            &global_count,
            1,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );


        // =============================================
        // WRITE GLOBAL AVERAGE
        // =============================================
        if (rank == 0) {

            double avg_Y =
                global_sum_Y / global_count;

            avg_file
                << delta
                << ", "
                << avg_Y
                << "\n";

            avg_file.flush();

            std::cout
                << "delta = " << delta
                << " Avg_Y = " << avg_Y
                << " based on "
                << global_count
                << " runs."
                << std::endl;
        }


        // =============================================
        // GATHER ALL Y_final VALUES
        // =============================================
        std::vector<int> global_Ys;

        if (rank == 0) {

            global_Ys.resize(
                runs_per_process * size
            );
        }

        MPI_Gather(
            local_Ys.data(),
            runs_per_process,
            MPI_INT,

            global_Ys.data(),
            runs_per_process,
            MPI_INT,

            0,
            MPI_COMM_WORLD
        );


        // =============================================
        // WRITE DISTRIBUTION FILE
        // =============================================
        // One file is written per τ value.
        // =============================================
        if (rank == 0) {

            std::ostringstream fname_stream;

            fname_stream
                << "Y_delta_"
                << std::fixed
                << std::setprecision(3)
                << delta
                << ".txt";

            std::string fname =
                fname_stream.str();

            std::ofstream out_file(
                fname,
                std::ios::app
            );

            for (int y : global_Ys) {

                out_file << y << "\n";
            }

            out_file.close();

            std::cout
                << "All results written to "
                << fname
                << " for delta = "
                << delta
                << std::endl;
        }
    }


    // =================================================
    // FINALIZE
    // =================================================
    if (rank == 0)
        avg_file.close();

    MPI_Finalize();

    return 0;
}