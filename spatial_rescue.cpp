#include <mpi.h>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <sstream>


// =======================================================
// PURPOSE OF THIS CODE
// =======================================================
//
// This MPI-parallelized stochastic simulation code
// implements a spatial birth–death–mutation model
// on a two-dimensional lattice.
//
// -------------------------------------------------------
// MODEL OVERVIEW
// -------------------------------------------------------
//
// Each lattice site can be:
//
//   0 : empty
//   1 : wild type
//   2 : mutant
//
// The simulation contains TWO PHASES:
//
// -------------------------------------------------------
// PHASE I
// -------------------------------------------------------
//
// Spatial growth with mutation and extinction restarts
// until the total population reaches:
//
//                 population = n
//
// If total extinction occurs:
//
//                 X + Y = 0
//
// the system automatically restarts from a single
// wild-type cell.
//
// -------------------------------------------------------
// PHASE II (TREATMENT and RESCUE)
// -------------------------------------------------------
//
// Once the population reaches n:
//
//   • ALL wild-type cells are instantaneously removed
//
//   • Only mutants survive
//
//   • Mutants evolve through stochastic birth–death
//     dynamics
//
// Rescue occurs if:
//
//                 Y = N
//
// before mutant extinction.
//
// =======================================================


// =======================================================
// RANDOM NUMBER GENERATOR
// =======================================================

std::mt19937 rng;


// Initialize RNG using rank-dependent seed
void initialize_random(int rank) {

    rng.seed(123456 + rank);
}


// Uniform random number in [0,1)
double random_double() {

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    return dist(rng);
}


// Uniform random integer in [min,max]
int random_int(int min, int max) {

    std::uniform_int_distribution<int> dist(min, max);

    return dist(rng);
}


// =======================================================
// GRID INITIALIZATION
// =======================================================
//
// Grid encoding:
//
//   0 : empty
//   1 : wild type
//   2 : mutant
//
// =======================================================
void initialize_grid(

    int grid_size,

    int &population_size,

    int &population_size_mutant,

    int &population_size_wild_type,

    std::vector<std::vector<int>> &grid,

    std::vector<std::pair<int, int>> &occupied_positions
) {

    // Empty lattice
    grid.assign(
        grid_size,
        std::vector<int>(grid_size, 0)
    );

    occupied_positions.clear();

    // Place single wild-type cell at center
    int center = grid_size / 2;

    grid[center][center] = 1;

    occupied_positions.push_back({center, center});

    population_size = 1;

    population_size_mutant = 0;

    population_size_wild_type = 1;
}


// =======================================================
// CELL DIVISION EVENT
// =======================================================
//
// A parent cell attempts to place an offspring
// into a randomly chosen neighboring site.
//
// Wild-type cells may mutate into mutants during
// reproduction.
//
// =======================================================
void divide_cell(

    int x,
    int y,

    int parent_type,

    double mutation_rate,

    std::vector<std::pair<int, int>> &neighbors,

    int &population_size,

    int &population_size_mutant,

    int &population_size_wild_type,

    std::vector<std::vector<int>> &grid,

    std::vector<std::pair<int, int>> &occupied_positions
) {

    // Select random neighboring direction
    int random_index =
        random_int(0, neighbors.size() - 1);

    int dx = neighbors[random_index].first;

    int dy = neighbors[random_index].second;

    // Periodic boundary conditions
    int nx =
        (x + dx + grid.size()) % grid.size();

    int ny =
        (y + dy + grid.size()) % grid.size();

    // Divide only into empty sites
    if (grid[nx][ny] == 0) {

        // Mutation during wild-type division
        int offspring_type =

            (parent_type == 1 &&
             random_double() < mutation_rate)

            ? 2
            : parent_type;

        // Insert offspring
        grid[nx][ny] = offspring_type;

        occupied_positions.push_back({nx, ny});

        population_size++;

        // Update counters
        if (offspring_type == 1)
            population_size_wild_type++;

        if (offspring_type == 2)
            population_size_mutant++;
    }
}


// =======================================================
// CELL REMOVAL EVENT
// =======================================================
void remove_cell(

    int x,
    int y,

    int idx,

    int cell_type,

    int &population_size,

    int &population_size_mutant,

    int &population_size_wild_type,

    std::vector<std::vector<int>> &grid,

    std::vector<std::pair<int, int>> &occupied_positions
) {

    if (grid[x][y] != 0) {

        // Remove cell from lattice
        grid[x][y] = 0;

        population_size--;

        // Update counters
        if (cell_type == 1)
            population_size_wild_type--;

        if (cell_type == 2)
            population_size_mutant--;

        // Fast vector removal
        int last_idx =
            occupied_positions.size() - 1;

        if (idx != last_idx) {

            occupied_positions[idx] =
                occupied_positions[last_idx];
        }

        occupied_positions.pop_back();
    }
}


// =======================================================
// RESULT STRUCTURES
// =======================================================

struct SpatialResultPhase1 {

    int population_size;

    int population_size_wild_type;

    int population_size_mutant;

    double T_n;

    std::vector<std::vector<int>> grid;

    std::vector<std::pair<int,int>> occupied_positions;
};


struct SpatialResultPhase2 {

    int X_final;

    int Y_final;

    int rescue_stat;

    double T_N;
};


// =======================================================
// PHASE I — SPATIAL GROWTH WITH RESTART
// =======================================================
SpatialResultPhase1
simulate_with_restarts_phase1_spatial(

    int grid_size,

    int n,

    double mutation_rate,

    double delta,

    double s1,

    double s2,

    double r,

    std::vector<std::pair<int,int>> neighbors
) {

    int population_size, X, Y;

    std::vector<std::vector<int>> grid;

    std::vector<std::pair<int,int>> occ;

    // Initialize system
    initialize_grid(
        grid_size,
        population_size,
        Y,
        X,
        grid,
        occ
    );

    double T = 0.0;

    // Main stochastic loop
    while (population_size > 0 &&
           population_size < n) {

        int steps = population_size;

        for (int i = 0;
             i < steps && population_size > 0;
             i++) {

            // Choose random occupied site
            int idx =
                random_int(0, population_size - 1);

            auto [x,y] = occ[idx];

            int type = grid[x][y];

            double z = random_double();

            // ===========================================
            // WILD-TYPE DYNAMICS
            // ===========================================
            if (type == 1) {

                // Wild-type death
                if (z < delta * r)

                    remove_cell(
                        x,y,idx,type,
                        population_size,
                        Y,
                        X,
                        grid,
                        occ
                    );

                // Wild-type division
                else if (z < delta * r + r)

                    divide_cell(
                        x,y,type,
                        mutation_rate,
                        neighbors,
                        population_size,
                        Y,
                        X,
                        grid,
                        occ
                    );
            }

            // ===========================================
            // MUTANT DYNAMICS
            // ===========================================
            else {

                // Mutant death
                if (z < delta * r * (1+s2))

                    remove_cell(
                        x,y,idx,type,
                        population_size,
                        Y,
                        X,
                        grid,
                        occ
                    );

                // Mutant division
                else if (z < delta * r * (1+s2)
                              + r*(1-s1))

                    divide_cell(
                        x,y,type,
                        mutation_rate,
                        neighbors,
                        population_size,
                        Y,
                        X,
                        grid,
                        occ
                    );
            }

            // Stop once population reaches n
            if (population_size == n)
                break;
        }

        T += 1.0;

        // ===============================================
        // RESTART AFTER TOTAL EXTINCTION
        // ===============================================
        if (population_size == 0) {

            initialize_grid(
                grid_size,
                population_size,
                Y,
                X,
                grid,
                occ
            );

            T = 0.0;
        }
    }

    return {
        population_size,
        X,
        Y,
        T,
        grid,
        occ
    };
}


// =======================================================
// PHASE II — RESCUE / TREATMENT PHASE
// =======================================================
//
// IMPORTANT:
//
// At the START of Phase II:
//
//     ALL wild-type cells die instantaneously
//
// Therefore:
//
//     X = 0
//
// and only mutants survive.
//
// =======================================================
SpatialResultPhase2
simulate_treatment_phase2_spatial(

    int N,

    double mutation_rate,

    double delta,

    double s1,

    double s2,

    double r,

    std::vector<std::pair<int,int>> neighbors,

    // ---- STATE FROM PHASE I ----
    int &population_size,

    int &X,

    int &Y,

    std::vector<std::vector<int>> &grid,

    std::vector<std::pair<int,int>> &occ
) {

    // ===================================================
    // REMOVE ALL WILD-TYPE CELLS INSTANTANEOUSLY
    // ===================================================

    std::vector<std::pair<int,int>> mutant_occ;

    for (auto [x,y] : occ) {

        // Keep mutants only
        if (grid[x][y] == 2) {

            mutant_occ.push_back({x,y});
        }

        // Remove wild types immediately
        else if (grid[x][y] == 1) {

            grid[x][y] = 0;
        }
    }

    // Replace occupied list
    occ = mutant_occ;

    // After treatment:
    X = 0;

    population_size = Y;


    // ===================================================
    // MUTANT-ONLY DYNAMICS
    // ===================================================

    double T = 0.0;

    while (population_size > 0 && Y < N) {

        int steps = population_size;

        for (int i = 0;
             i < steps && population_size > 0;
             i++) {

            int idx =
                random_int(0, population_size - 1);

            auto [x,y] = occ[idx];

            int type = grid[x][y];

            double z = random_double();

            // Only mutants exist now
            if (type == 2) {

                // Mutant death
                if (z < delta * r * (1+s2))

                    remove_cell(
                        x,y,idx,type,
                        population_size,
                        Y,
                        X,
                        grid,
                        occ
                    );

                // Mutant division
                else if (z < delta * r * (1+s2)
                              + r*(1-s1))

                    divide_cell(
                        x,y,type,
                        mutation_rate,
                        neighbors,
                        population_size,
                        Y,
                        X,
                        grid,
                        occ
                    );
            }
        }

        T += 1.0;
    }

    SpatialResultPhase2 res;

    res.X_final = X;

    res.Y_final = Y;

    // Rescue if mutants reach N
    res.rescue_stat = (Y >= N ? 1 : 0);

    res.T_N = (Y >= N ? T : 0.0);

    return res;
}


// =======================================================
// MAIN PROGRAM
// =======================================================
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    initialize_random(rank);


    // ===================================================
    // MODEL PARAMETERS
    // ===================================================

    double s1 = 0.2;

    double s2 = 0.0;

    double u  = 2e-6;

    double r  = 0.25;

    int grid_size = 500;

    int n = 10000;

    int N = n + 1000;

    long long A = 500000;


    // ===================================================
    // MPI WORK DISTRIBUTION
    // ===================================================

    long long base_runs = A / size;

    long long remainder = A % size;

    long long runs_per_process;

    if (rank < remainder)
        runs_per_process = base_runs + 1;

    else
        runs_per_process = base_runs;


    // ===================================================
    // DELTA VALUES
    // ===================================================

    std::vector<double> delta_values = {
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6
    };


    // Moore neighborhood
    const std::vector<std::pair<int,int>> NEIGHBORS = {

        {1,1}, {1,0}, {1,-1},

        {0,1},        {0,-1},

        {-1,1}, {-1,0}, {-1,-1}
    };


    // ===================================================
    // OUTPUT FILES
    // ===================================================

    std::ofstream rescue_file;

    std::map<double, std::ofstream> phase1_files;

    if (rank == 0) {

        rescue_file.open("rescue_probability.txt");

        rescue_file
            << "# delta,rescue_probability\n";

        for (double delta : delta_values) {

            std::ostringstream fname;

            fname
                << "phase1_final_mutants_delta_"
                << delta
                << ".txt";

            phase1_files[delta].open(fname.str());

            phase1_files[delta]
                << "# Y_final_phase1\n";
        }
    }


    // ===================================================
    // MAIN LOOP OVER DELTA
    // ===================================================

    for (double delta : delta_values) {

        long long local_rescues = 0;

        std::vector<int> local_phase1_Y;

        // ===============================================
        // STOCHASTIC REALIZATIONS
        // ===============================================
        for (long long i = 0;
             i < runs_per_process;
             ++i) {

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
            SpatialResultPhase1 res1 =

                simulate_with_restarts_phase1_spatial(

                    grid_size,

                    n,

                    u,

                    delta,

                    s1,

                    s2,

                    r,

                    NEIGHBORS
                );

            int Y_phase1 =
                res1.population_size_mutant;

            if (Y_phase1 > 0)
                local_phase1_Y.push_back(Y_phase1);


            // ===========================================
            // STATE TRANSFER TO PHASE II
            // ===========================================
            int population_size =
                res1.population_size;

            int X =
                res1.population_size_wild_type;

            int Y =
                res1.population_size_mutant;


            // ===========================================
            // PHASE II
            // ===========================================
            SpatialResultPhase2 res2 =

                simulate_treatment_phase2_spatial(

                    N,

                    u,

                    delta,

                    s1,

                    s2,

                    r,

                    NEIGHBORS,

                    population_size,

                    X,

                    Y,

                    res1.grid,

                    res1.occupied_positions
                );

            local_rescues += res2.rescue_stat;
        }


        // ===================================================
        // REDUCE RESCUE COUNTS
        // ===================================================

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


        // ===================================================
        // GATHER PHASE-I MUTANT COUNTS
        // ===================================================

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

        if (rank == 0) {

            displs.resize(size);

            int total = 0;

            for (int i = 0; i < size; i++) {

                displs[i] = total;

                total += recv_counts[i];
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


        // ===================================================
        // WRITE OUTPUT
        // ===================================================

        if (rank == 0) {

            double rescue_prob =
                double(global_rescues) / double(A);

            rescue_file
                << delta
                << ","
                << rescue_prob
                << "\n";

            for (int Yval : global_Y) {

                phase1_files[delta]
                    << Yval
                    << "\n";
            }
        }
    }


    // ===================================================
    // CLOSE FILES
    // ===================================================

    if (rank == 0) {

        rescue_file.close();

        for (auto &kv : phase1_files)
            kv.second.close();
    }

    MPI_Finalize();

    return 0;
}