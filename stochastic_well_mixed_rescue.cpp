#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>


// ======================================================
// PURPOSE OF THIS CODE
// ======================================================
//
// This MPI-parallelized Gillespie simulation code
// implements a TWO-PHASE stochastic well-mixed model
// for mutant rescue dynamics.
//
// ------------------------------------------------------
// PHASE I
// ------------------------------------------------------
//
// Population growth with mutation and extinction
// restart dynamics until the total population reaches:
//
//                  X + Y = n
//
// If extinction occurs:
//
//                  X + Y = 0
//
// the system restarts from:
//
//                  X = 1
//                  Y = 0
//
// ------------------------------------------------------
// PHASE II (TREATMENT / RESCUE PHASE)
// ------------------------------------------------------
//
// Once the population reaches n:
//
//   • ALL wild-type cells die instantaneously
//
//   • Therefore:
//
//            X = 0
//
//   • Only mutants survive
//
//   • Mutants evolve through stochastic birth–death
//     dynamics
//
// Rescue occurs if:
//
//                  Y = N
//
// before mutant extinction.
//
// ======================================================


// ======================================================
// RANDOM NUMBER GENERATOR
// ======================================================

// Uniform random number in [0,1)
double random_number() {

    return static_cast<double>(rand()) / RAND_MAX;
}


// ======================================================
// PHASE I RESULT STRUCTURE
// ======================================================

struct SimulationResultPhase1 {

    int X_final;

    int Y_final;

    double T_n;
};


// ======================================================
// PHASE II RESULT STRUCTURE
// ======================================================

struct SimulationResultPhase2 {

    int rescue_stat;

    double T_N;
};


// ======================================================
// PHASE I — GROWTH WITH RESTARTS
// ======================================================
//
// Population evolves until:
//
//              X + Y = n
//
// If extinction occurs:
//
//              X + Y = 0
//
// system automatically restarts.
//
// ======================================================
SimulationResultPhase1
simulate_with_restarts_phase1(

    double s1,

    double s2,

    double u,

    double delta,

    int n,

    int initial_X
) {

    // Initial populations
    int X = initial_X;

    int Y = 0;

    // Continuous time
    double T = 0.0;

    // Baseline birth rate
    double r = 0.25;


    // ==================================================
    // MAIN GILLESPIE LOOP
    // ==================================================
    while (X + Y > 0 && X + Y < n) {

        // ===============================================
        // REACTION RATES
        // ===============================================
        //
        // 0 : WT birth
        // 1 : WT death
        // 2 : WT mutation
        // 3 : mutant birth
        // 4 : mutant death
        //
        // ===============================================
        std::vector<double> rates = {

            r * (1.0 - u) * X,

            r * delta * X,

            r * u * X,

            r * (1.0 - s1) * Y,

            r * delta * (1.0 + s2) * Y
        };

        // Total reaction rate
        double a0 = 0;

        for (double rate : rates)
            a0 += rate;

        if (a0 == 0)
            break;

        // Random numbers
        double r1 = random_number();

        double r2 = random_number();

        // Gillespie time increment
        double dt = -std::log(r1) / a0;

        T += dt;

        // Reaction selection
        double scaled_r2 = r2 * a0;

        double cumulative = 0;

        size_t reaction_index = 0;

        for (; reaction_index < rates.size();
               ++reaction_index) {

            cumulative += rates[reaction_index];

            if (scaled_r2 <= cumulative)
                break;
        }

        // ===============================================
        // APPLY REACTION
        // ===============================================

        // WT birth
        if (reaction_index == 0)
            X += 1;

        // WT death
        else if (reaction_index == 1)
            X -= 1;

        // WT mutation
        else if (reaction_index == 2)
            Y += 1;

        // Mutant birth
        else if (reaction_index == 3)
            Y += 1;

        // Mutant death
        else if (reaction_index == 4)
            Y -= 1;


        // ===============================================
        // RESTART AFTER EXTINCTION
        // ===============================================
        if (X + Y == 0) {

            X = initial_X;

            Y = 0;
        }
    }

    return {X, Y, T};
}


// ======================================================
// PHASE II — RESCUE PHASE
// ======================================================
//
// IMPORTANT:
//
// At the START of Phase II:
//
//      ALL wild-type cells die instantly
//
// Therefore:
//
//      X = 0
//
// Only mutants survive.
//
// ======================================================
SimulationResultPhase2
simulate_treatment_phase2(

    double s1,

    double s2,

    double u,

    double delta,

    int N,

    int X,

    int Y
) {

    // ==================================================
    // INSTANTANEOUS WILD-TYPE DEATH
    // ==================================================

    X = 0;

    double T = 0.0;

    double r = 0.25;


    // ==================================================
    // MUTANT-ONLY DYNAMICS
    // ==================================================
    while (Y > 0 && Y < N) {

        // ===============================================
        // REACTION RATES
        // ===============================================
        //
        // 0 : mutant birth
        // 1 : mutant death
        //
        // ===============================================
        std::vector<double> rates = {

            r * (1.0 - s1) * Y,

            r * delta * (1.0 + s2) * Y
        };

        // Total rate
        double a0 = 0;

        for (double rate : rates)
            a0 += rate;

        if (a0 == 0)
            break;

        // Random numbers
        double r1 = random_number();

        double r2 = random_number();

        // Gillespie time step
        double dt = -std::log(r1) / a0;

        T += dt;

        // Reaction selection
        double scaled_r2 = r2 * a0;

        double cumulative = 0;

        size_t reaction_index = 0;

        for (; reaction_index < rates.size();
               ++reaction_index) {

            cumulative += rates[reaction_index];

            if (scaled_r2 <= cumulative)
                break;
        }

        // ===============================================
        // APPLY REACTION
        // ===============================================

        // Mutant birth
        if (reaction_index == 0)
            Y += 1;

        // Mutant death
        else if (reaction_index == 1)
            Y -= 1;
    }

    SimulationResultPhase2 result;

    // Rescue if mutants reach N
    result.rescue_stat = (Y >= N ? 1 : 0);

    result.T_N = (Y >= N ? T : 0.0);

    return result;
}


// ======================================================
// MAIN PROGRAM
// ======================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);

    int rank,size;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Different random seed per MPI rank
    srand(time(0) + rank);


    // ==================================================
    // MODEL PARAMETERS
    // ==================================================

    double s1 = 0.2;

    double s2 = 0.0;

    double u  = 2e-6;

    int n = 10000;

    int N = n + 1000;

    int initial_X = 1;

    long long A = 500000;


    // ==================================================
    // MPI WORK DISTRIBUTION
    // ==================================================

    long long base_runs = A / size;

    long long remainder = A % size;

    long long runs_per_process =

        (rank < remainder
         ? base_runs + 1
         : base_runs);


    // ==================================================
    // DELTA VALUES
    // ==================================================

    std::vector<double> delta_values =
        {0.0,0.1,0.2,0.3,0.4,0.5,0.6};


    // ==================================================
    // OUTPUT FILES
    // ==================================================

    std::ofstream rescue_file;

    std::map<double,std::ofstream> phase1_files;

    if (rank == 0) {

        rescue_file.open("rescue_probability.txt");

        rescue_file
            << "# delta,rescue_probability\n";

        for (double delta : delta_values) {

            std::ostringstream name;

            name
                << "phase1_final_mutants_delta_"
                << delta
                << ".txt";

            phase1_files[delta].open(name.str());

            phase1_files[delta]
                << "# Y_phase1\n";
        }
    }


    // ==================================================
    // MAIN LOOP OVER DELTA
    // ==================================================

    for (double delta : delta_values) {

        long long local_rescues = 0;

        std::vector<int> local_phase1_Y;


        // ===============================================
        // STOCHASTIC REALIZATIONS
        // ===============================================
        for (long long i=0;
             i<runs_per_process;
             i++) {

            if (rank == 0 && i % 10 == 0) {

                std::cout
                    << "[delta=" << delta << "] "
                    << "progress: "
                    << i
                    << "/"
                    << runs_per_process
                    << std::endl;
            }

            // ===========================================
            // PHASE I
            // ===========================================
            SimulationResultPhase1 res1 =

                simulate_with_restarts_phase1(

                    s1,

                    s2,

                    u,

                    delta,

                    n,

                    initial_X
                );

            // Store nonzero mutant counts
            if (res1.Y_final > 0)
                local_phase1_Y.push_back(
                    res1.Y_final
                );


            // ===========================================
            // PHASE II
            // ===========================================
            SimulationResultPhase2 res2 =

                simulate_treatment_phase2(

                    s1,

                    s2,

                    u,

                    delta,

                    N,

                    res1.X_final,

                    res1.Y_final
                );

            local_rescues +=
                res2.rescue_stat;
        }


        // ==================================================
        // REDUCE RESCUE COUNTS
        // ==================================================

        long long global_rescues = 0;

        MPI_Reduce(

            &local_rescues,

            &global_rescues,

            1,

            MPI_LONG_LONG,

            MPI_SUM,

            0,

            MPI_COMM_WORLD
        );


        // ==================================================
        // GATHER PHASE-I MUTANT COUNTS
        // ==================================================

        int local_n = local_phase1_Y.size();

        std::vector<int> recv_counts(size);

        MPI_Gather(

            &local_n,

            1,

            MPI_INT,

            recv_counts.data(),

            1,

            MPI_INT,

            0,

            MPI_COMM_WORLD
        );

        std::vector<int> displs;

        std::vector<int> global_Y;

        if (rank==0) {

            displs.resize(size);

            int total=0;

            for(int i=0;i<size;i++){

                displs[i]=total;

                total+=recv_counts[i];
            }

            global_Y.resize(total);
        }

        MPI_Gatherv(

            local_phase1_Y.data(),

            local_n,

            MPI_INT,

            global_Y.data(),

            recv_counts.data(),

            displs.data(),

            MPI_INT,

            0,

            MPI_COMM_WORLD
        );


        // ==================================================
        // WRITE OUTPUT
        // ==================================================

        if (rank==0) {

            double rescue_prob =

                double(global_rescues)
                / double(A);

            rescue_file
                << delta
                << ","
                << rescue_prob
                << "\n";

            for (int Yval : global_Y)

                phase1_files[delta]
                    << Yval
                    << "\n";
        }
    }


    // ==================================================
    // CLOSE FILES
    // ==================================================

    if (rank==0) {

        rescue_file.close();

        for (auto &kv : phase1_files)

            kv.second.close();
    }

    MPI_Finalize();

    return 0;
}