// ============================================================
// FULL CODE (TEXT output, ONE FILE PER RANK PER delta)
// ============================================================
//
// PURPOSE OF THE CODE
// ------------------------------------------------------------
// This MPI-parallelized stochastic simulation code generates
// mutant-burden trajectory data for a spatial birth–death–
// mutation model on a two-dimensional lattice.
//
// The code is designed to:
//
//   • Simulate many independent stochastic trajectories
//   • Record mutant counts at logarithmically spaced
//     population-size checkpoints
//   • Handle extinction via automatic restarts
//   • Distribute trajectories over MPI ranks
//   • Write compact trajectory files for later analysis
//
// DATA GENERATED
// ------------------------------------------------------------
// For each trajectory:
//
//   • The total population evolves stochastically
//   • At the FIRST time the total population reaches
//     each checkpoint N_i, the mutant count is stored
//
// OUTPUT FORMAT
// ------------------------------------------------------------
// Each MPI rank writes its own file:
//
//     mutant_records_deltaXXXX_rankYYYY.txt
//
// containing:
//
//   • Header information
//   • Checkpoint list
//   • One row per stochastic trajectory
//
// Each row contains:
//
//   mutant counts at first-hit checkpoints
//
// IMPORTANT FEATURES
// ------------------------------------------------------------
// • 100000 independent trajectories distributed over MPI ranks
//
// • 300 logarithmically spaced checkpoints between
//   population sizes:
//
//          1  →  10^6
//
// • Extinction restarts:
//   If the population reaches zero, the system restarts
//   from a single wild-type cell within the SAME trajectory
//
// • Previously recorded checkpoints are preserved
//   after extinction and restart
//
// • Streaming output:
//   Data are written trajectory-by-trajectory to avoid
//   large memory usage
//
// APPLICATION
// ------------------------------------------------------------
// The generated files can be used to:
//
//   • Compute average mutant burden
//   • Construct mutant-burden distributions
//   • Analyze stochastic jackpot events
//   • Study scaling with population size
//
// ============================================================

#include <mpi.h>
#include <ctime>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>


// ------------------------------------------------------------
// CHECKPOINT GENERATION
// ------------------------------------------------------------
// Generates logarithmically spaced integer checkpoints
// between minN and maxN.
//
// Example:
//
//     1, 2, 3, 4, ..., 10, ..., 10^6
//
// These checkpoints determine the population sizes
// at which mutant counts are recorded.
// ------------------------------------------------------------
std::vector<int> make_log_checkpoints(
    int n_points,
    int minN,
    int maxN
) {

    std::vector<int> cps;

    cps.reserve(n_points);

    double log_min = std::log10((double)minN);

    double log_max = std::log10((double)maxN);

    int last = -1;

    for (int i = 0; i < n_points; ++i) {

        double t = (n_points == 1)
            ? 0.0
            : (double)i / (double)(n_points - 1);

        double x = log_min + t * (log_max - log_min);

        int val =
            (int)std::llround(std::pow(10.0, x));

        if (val < minN)
            val = minN;

        if (val > maxN)
            val = maxN;

        if (val <= last)
            val = last + 1;

        if (val > maxN)
            val = maxN;

        if (!cps.empty() && val == cps.back())
            continue;

        cps.push_back(val);

        last = val;

        if (val == maxN)
            break;
    }

    while ((int)cps.size() < n_points)
        cps.push_back(maxN);

    return cps;
}


// ------------------------------------------------------------
// RANDOM NUMBER GENERATOR (RNG)
// ------------------------------------------------------------

std::mt19937 rng;


// Initialize RNG using hardware entropy
void initialize_random() {

    std::random_device rd;

    rng = std::mt19937(rd());
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


// ------------------------------------------------------------
// GRID INITIALIZATION
// ------------------------------------------------------------
// Initializes a square lattice containing:
//
//   • one wild-type cell at the center
//   • zero mutant cells
//
// Grid encoding:
//
//   0 : empty
//   1 : wild type
//   2 : mutant
// ------------------------------------------------------------
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

    // Place initial wild-type cell at center
    int center = grid_size / 2;

    grid[center][center] = 1;

    occupied_positions.push_back({center, center});

    // Initial populations
    population_size = 1;

    population_size_mutant = 0;

    population_size_wild_type = 1;
}


// ------------------------------------------------------------
// CELL DIVISION EVENT
// ------------------------------------------------------------
// A parent cell attempts to place an offspring
// into a randomly chosen neighboring site.
//
// Wild-type cells may mutate into mutants during
// reproduction with probability mutation_rate.
// ------------------------------------------------------------
void divide_cell(

    int x,
    int y,

    int parent_type,

    double mutation_rate,

    const std::vector<std::pair<int, int>> &neighbors,

    int &population_size,

    int &population_size_mutant,

    int &population_size_wild_type,

    std::vector<std::vector<int>> &grid,

    std::vector<std::pair<int, int>> &occupied_positions
) {

    // Choose random neighboring direction
    int random_index =
        random_int(0, (int)neighbors.size() - 1);

    int dx = neighbors[random_index].first;

    int dy = neighbors[random_index].second;

    // Periodic boundary conditions
    int n = (int)grid.size();

    int nx = (x + dx + n) % n;

    int ny = (y + dy + n) % n;

    // Only divide into empty sites
    if (grid[nx][ny] == 0) {

        // Mutation during division
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


// ------------------------------------------------------------
// CELL REMOVAL EVENT
// ------------------------------------------------------------
// Removes a cell from the lattice and updates
// bookkeeping structures.
// ------------------------------------------------------------
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
            (int)occupied_positions.size() - 1;

        if (idx != last_idx) {

            occupied_positions[idx] =
                occupied_positions[last_idx];
        }

        occupied_positions.pop_back();
    }
}


// ------------------------------------------------------------
// SINGLE STOCHASTIC TRAJECTORY
// ------------------------------------------------------------
//
// Simulates one full realization of the spatial
// birth–death–mutation process.
//
// The mutant count is recorded whenever the total
// population FIRST reaches a checkpoint.
//
// Extinction triggers a restart from one wild-type
// cell while preserving previously recorded values.
//
// Output:
//
//   records_out[i]
//
// = mutant count when checkpoint i is first reached
//
// ------------------------------------------------------------
void simulate_population(

    int grid_size,

    int max_population,

    double mutation_rate,

    double death_prob_mutant,

    double death_prob_wild_type,

    double division_prob_mutant,

    double division_prob_wild_type,

    const std::vector<std::pair<int, int>> &neighbors,

    const std::vector<int> &checkpoints,

    std::vector<int> &records_out,

    int rank
) {

    // Occupied sites
    std::vector<std::pair<int, int>>
        occupied_positions;

    // Spatial lattice
    std::vector<std::vector<int>>
        grid(grid_size,
             std::vector<int>(grid_size, 0));

    // Population counters
    int population_size = 0;

    int population_size_mutant = 0;

    int population_size_wild_type = 0;

    int extinction_count = 0;

    // Initialize output vector
    records_out.assign(checkpoints.size(), -1);

    // Index of next checkpoint
    int next_cp = 0;


    // --------------------------------------------------------
    // CHECKPOINT RECORDING FUNCTION
    // --------------------------------------------------------
    auto try_record = [&]() {

        while (
            next_cp < (int)checkpoints.size()
            &&
            population_size >= checkpoints[next_cp]
        ) {

            records_out[next_cp] =
                population_size_mutant;

            next_cp++;
        }
    };


    // Initialize system
    initialize_grid(
        grid_size,
        population_size,
        population_size_mutant,
        population_size_wild_type,
        grid,
        occupied_positions
    );

    try_record();


    // --------------------------------------------------------
    // MAIN STOCHASTIC DYNAMICS LOOP
    // --------------------------------------------------------
    while (
        population_size > 0
        &&
        population_size < max_population
    ) {

        int iterations = population_size;

        for (int i = 0; i < iterations; i++) {

            if (population_size == 0)
                break;

            // Choose random occupied site
            int idx =
                random_int(0, population_size - 1);

            auto [x, y] = occupied_positions[idx];

            int cell_type = grid[x][y];

            double z = random_double();


            // =================================================
            // WILD-TYPE CELL DYNAMICS
            // =================================================
            if (cell_type == 1) {

                // Death
                if (z < death_prob_wild_type) {

                    remove_cell(
                        x, y, idx, cell_type,
                        population_size,
                        population_size_mutant,
                        population_size_wild_type,
                        grid,
                        occupied_positions
                    );

                    try_record();
                }

                // Division
                else if (
                    z <
                    division_prob_wild_type
                    + death_prob_wild_type
                ) {

                    divide_cell(
                        x, y, cell_type,
                        mutation_rate,
                        neighbors,
                        population_size,
                        population_size_mutant,
                        population_size_wild_type,
                        grid,
                        occupied_positions
                    );

                    try_record();
                }
            }


            // =================================================
            // MUTANT CELL DYNAMICS
            // =================================================
            else if (cell_type == 2) {

                // Death
                if (z < death_prob_mutant) {

                    remove_cell(
                        x, y, idx, cell_type,
                        population_size,
                        population_size_mutant,
                        population_size_wild_type,
                        grid,
                        occupied_positions
                    );

                    try_record();
                }

                // Division
                else if (
                    z <
                    division_prob_mutant
                    + death_prob_mutant
                ) {

                    divide_cell(
                        x, y, cell_type,
                        mutation_rate,
                        neighbors,
                        population_size,
                        population_size_mutant,
                        population_size_wild_type,
                        grid,
                        occupied_positions
                    );

                    try_record();
                }
            }
        }


        // =====================================================
        // EXTINCTION RESTART
        // =====================================================
        if (population_size == 0) {

            extinction_count++;

            initialize_grid(
                grid_size,
                population_size,
                population_size_mutant,
                population_size_wild_type,
                grid,
                occupied_positions
            );

            try_record();
        }


        // =====================================================
        // OPTIONAL PROGRESS OUTPUT
        // =====================================================
        if (rank == 0) {

            static int last_print = 0;

            if (population_size >= last_print + 500000) {

                std::cout
                    << "[rank0] pop="
                    << population_size
                    << " mutants="
                    << population_size_mutant
                    << " wt="
                    << population_size_wild_type
                    << std::endl
                    << std::flush;

                last_print = population_size;
            }
        }
    }
}


// ------------------------------------------------------------
// MAIN PROGRAM
// ------------------------------------------------------------
int main(int argc, char **argv) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank = 0;

    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        std::cout
            << "MPI size="
            << size
            << std::endl;

    // Initialize random number generator
    initialize_random();


    // =========================================================
    // MODEL PARAMETERS
    // =========================================================

    const int GRID_SIZE = 2000;

    const int NUM_SIMULATIONS = 100000;

    const double DIVISION_PROB_WILD_TYPE = 0.5;

    // Death-to-birth ratios
    std::vector<double> delta_values = {
        0.0,
        0.05,
        0.1,
        0.25,
        0.35
    };

    // Moore neighborhood
    const std::vector<std::pair<int, int>> NEIGHBORS = {

        { 1, 1}, { 1, 0}, { 1, -1},

        { 0, 1},          { 0, -1},

        {-1, 1}, {-1, 0}, {-1, -1}
    };

    const int MAX_POPULATION =
        (int)std::pow(10, 6);

    const double MUTATION_RATE =
        std::pow(10, -4);

    const double s1 = 0.0;

    const double s2 = 0.05;


    // =========================================================
    // CHECKPOINTS
    // =========================================================
    const int N_CHECKPOINTS = 300;

    std::vector<int> checkpoints =
        make_log_checkpoints(
            N_CHECKPOINTS,
            1,
            MAX_POPULATION
        );


    // =========================================================
    // DISTRIBUTE TRAJECTORIES ACROSS MPI RANKS
    // =========================================================

    int base = NUM_SIMULATIONS / size;

    int rem  = NUM_SIMULATIONS % size;

    // First `rem` ranks get one extra trajectory
    int local_num_simulations =
        base + (rank < rem ? 1 : 0);

    // Global starting row index
    int start_row =
        rank * base + std::min(rank, rem);


    // =========================================================
    // MAIN LOOP OVER τ
    // =========================================================
    for (double delta : delta_values) {

        // Derived rates
        double death_prob_wild_type =
            delta * DIVISION_PROB_WILD_TYPE;

        double division_prob_mutant =
            DIVISION_PROB_WILD_TYPE * (1.0 - s1);

        double death_prob_mutant =
            death_prob_wild_type * (1.0 + s2);


        // =====================================================
        // OUTPUT FILE NAME
        // =====================================================
        std::ostringstream fname_stream;

        fname_stream
            << "mutant_records_delta"
            << std::fixed
            << std::setprecision(4)
            << delta
            << "_rank"
            << std::setw(4)
            << std::setfill('0')
            << rank
            << ".txt";

        std::string fname =
            fname_stream.str();


        // Open output file
        std::ofstream out(fname);

        if (!out) {

            std::cerr
                << "Rank "
                << rank
                << " cannot open output file: "
                << fname
                << std::endl;

            MPI_Abort(MPI_COMM_WORLD, 1);
        }


        // =====================================================
        // FILE HEADER
        // =====================================================
        out << "# delta "
            << std::fixed
            << std::setprecision(6)
            << delta
            << "\n";

        out << "# rank "
            << rank
            << " / "
            << size
            << "\n";

        out << "# global_rows "
            << NUM_SIMULATIONS
            << "\n";

        out << "# local_rows "
            << local_num_simulations
            << "\n";

        out << "# global_start_row "
            << start_row
            << "\n";

        out << "# ncheck "
            << N_CHECKPOINTS
            << "\n";

        out << "# checkpoints ";

        for (int i = 0; i < N_CHECKPOINTS; ++i) {

            out << checkpoints[i];

            if (i + 1 < N_CHECKPOINTS)
                out << " ";
        }

        out << "\n";

        out << "# data: each row = one trajectory, "
            << N_CHECKPOINTS
            << " integers = mutant counts at first-hit checkpoints\n";

        out.flush();


        // =====================================================
        // START MESSAGE
        // =====================================================
        if (rank == 0) {

            auto now =
                std::chrono::system_clock::now();

            std::time_t now_time =
                std::chrono::system_clock::to_time_t(now);

            std::cout
                << "["
                << std::put_time(
                    std::localtime(&now_time),
                    "%Y-%m-%d %H:%M:%S"
                )
                << "] "
                << "delta="
                << std::fixed
                << std::setprecision(4)
                << delta
                << " writing per-rank files; rank0 local_rows="
                << local_num_simulations
                << std::endl
                << std::flush;
        }


        // =====================================================
        // RUN LOCAL TRAJECTORIES
        // =====================================================
        for (int sim = 0;
             sim < local_num_simulations;
             ++sim) {

            if (rank == 0) {

                std::cout
                    << "  [rank0] sim="
                    << sim
                    << "/"
                    << local_num_simulations
                    << " (delta="
                    << std::fixed
                    << std::setprecision(4)
                    << delta
                    << ")"
                    << std::endl
                    << std::flush;
            }

            std::vector<int> rec;

            rec.reserve(N_CHECKPOINTS);


            // Run one stochastic trajectory
            simulate_population(

                GRID_SIZE,

                MAX_POPULATION,

                MUTATION_RATE,

                death_prob_mutant,

                death_prob_wild_type,

                division_prob_mutant,

                DIVISION_PROB_WILD_TYPE,

                NEIGHBORS,

                checkpoints,

                rec,

                rank
            );


            // =================================================
            // WRITE ONE TRAJECTORY ROW
            // =================================================
            for (int j = 0;
                 j < N_CHECKPOINTS;
                 ++j) {

                out << rec[j];

                if (j + 1 < N_CHECKPOINTS)
                    out << ' ';
            }

            out << '\n';

            out.flush();
        }


        // Close file
        out.close();


        // Synchronize MPI ranks
        MPI_Barrier(MPI_COMM_WORLD);


        // =====================================================
        // FINISH MESSAGE
        // =====================================================
        if (rank == 0) {

            auto now =
                std::chrono::system_clock::now();

            std::time_t now_time =
                std::chrono::system_clock::to_time_t(now);

            std::cout
                << "["
                << std::put_time(
                    std::localtime(&now_time),
                    "%Y-%m-%d %H:%M:%S"
                )
                << "] "
                << "Finished delta="
                << std::fixed
                << std::setprecision(4)
                << delta
                << " (files: mutant_records_deltaXXXX_rankYYYY.txt)"
                << std::endl
                << std::flush;
        }
    }


    // Finalize MPI
    MPI_Finalize();

    return 0;
}