#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <string>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <thread>
#include <mutex>
#include <chrono>

using namespace Eigen;
using namespace std;

// Structure for solution
struct Solution
{
    vector<double> variables;
    double weight;
    double max_displacement;
    vector<double> stresses;
    vector<double> constraint_violations;
    bool feasible;

    // For multiple load cases
    vector<double> max_displacements_per_case;
    vector<vector<double>> stresses_per_case;
    vector<vector<double>> constraint_violations_per_case;
    vector<bool> feasible_per_case;

    // For APM fitness
    double fitness_weight;
    double fitness_displacement;

    // For multi-objective comparison
    vector<double> objectives;
    double crowding_distance;
    int rank;

    // For SaDE - track which parameters were used
    double used_CR;
    double used_F;

    Solution() : weight(0.0), max_displacement(0.0), feasible(false),
                 fitness_weight(0.0), fitness_displacement(0.0),
                 crowding_distance(0.0), rank(0), used_CR(0.5), used_F(0.5) {}
};

struct TrussProblem
{
    int n_bars;
    int n_nodes;
    int n_dof;
    int n_vars;
    double s_adm;
    vector<double> areas;
    MatrixXd nodes;
    MatrixXi bars;
    VectorXd f_vector;
    vector<int> fixed_dof;
    vector<int> member_groups;

    double rho;
    double E;

    int n_pop;
    int gen;
    double CR; // Initial value, will be adapted by SaDE
    double F;  // Initial value, will be adapted by SaDE

    // SaDE parameters
    int H; // Memory size for CR and F
    double CR_min, CR_max;
    double F_min, F_max;
    int LP; // Learning period

    TrussProblem() : rho(0.1), E(10000.0), H(5), CR_min(0.05), CR_max(0.95),
                     F_min(0.1), F_max(1.0), LP(50) {}
};

class GDE3_SaDE_APM
{
private:
    TrussProblem problem_;
    vector<Solution> population_;
    vector<Solution> external_archive_;
    mt19937 rng_;
    long seed_;

    // SaDE memory structures
    vector<double> M_CR; // Memory for CR values
    vector<double> M_F;  // Memory for F values
    int memory_index_;

    // Success tracking for SaDE
    vector<double> successful_CR_;
    vector<double> successful_F_;
    vector<double> unsuccessful_CR_;
    vector<double> unsuccessful_F_;

public:
    GDE3_SaDE_APM(const TrussProblem &problem, long seed = 0) : problem_(problem), seed_(seed), memory_index_(0)
    {
        if (seed == 0)
        {
            // Use random seed if not specified
            random_device rd;
            seed_ = rd();
        }
        rng_.seed(seed_);

        // Initialize SaDE memory
        initializeSaDEMemory();
    }

    // Initialize SaDE memory with initial values
    void initializeSaDEMemory()
    {
        M_CR.resize(problem_.H, problem_.CR);
        M_F.resize(problem_.H, problem_.F);

        successful_CR_.clear();
        successful_F_.clear();
        unsuccessful_CR_.clear();
        unsuccessful_F_.clear();

        memory_index_ = 0;
    }

    // Generate CR value using SaDE method
    double generateCR()
    {
        uniform_int_distribution<int> mem_dist(0, problem_.H - 1);
        int k = mem_dist(rng_);

        normal_distribution<double> normal_dist(M_CR[k], 0.1);
        double cr = normal_dist(rng_);

        // Truncate to [CR_min, CR_max]
        cr = max(problem_.CR_min, min(cr, problem_.CR_max));
        return cr;
    }

    // Generate F value using SaDE method
    double generateF()
    {
        uniform_int_distribution<int> mem_dist(0, problem_.H - 1);
        int k = mem_dist(rng_);

        // Use Cauchy distribution for F
        cauchy_distribution<double> cauchy_dist(M_F[k], 0.1);
        double f = cauchy_dist(rng_);

        // Truncate to [F_min, F_max] and regenerate if too large
        while (f <= problem_.F_min || f > problem_.F_max)
        {
            f = cauchy_dist(rng_);
        }
        return f;
    }

    // Update SaDE memory based on successful and unsuccessful parameters
    void updateSaDEMemory()
    {
        if (successful_CR_.empty() && successful_F_.empty())
            return;

        // Update CR memory
        if (!successful_CR_.empty())
        {
            double mean_successful_CR = accumulate(successful_CR_.begin(), successful_CR_.end(), 0.0) / successful_CR_.size();
            M_CR[memory_index_] = mean_successful_CR;
        }

        // Update F memory using Lehmer mean
        if (!successful_F_.empty())
        {
            double sum_F = 0.0, sum_F_sq = 0.0;
            for (double f : successful_F_)
            {
                sum_F += f;
                sum_F_sq += f * f;
            }
            double lehmer_mean = sum_F_sq / sum_F;
            M_F[memory_index_] = lehmer_mean;
        }

        // Update memory index
        memory_index_ = (memory_index_ + 1) % problem_.H;

        // Clear success records for next learning period
        successful_CR_.clear();
        successful_F_.clear();
        unsuccessful_CR_.clear();
        unsuccessful_F_.clear();
    }

    // Record successful parameters
    void recordSuccessfulParameters(double CR, double F)
    {
        successful_CR_.push_back(CR);
        successful_F_.push_back(F);
    }

    // Record unsuccessful parameters
    void recordUnsuccessfulParameters(double CR, double F)
    {
        unsuccessful_CR_.push_back(CR);
        unsuccessful_F_.push_back(F);
    }

    // Truss analysis function
    pair<double, VectorXd> solveTruss(const vector<double> &areas)
    {
        for (int i = 0; i < problem_.n_bars; i++)
        {
            int node1 = problem_.bars(i, 0) - 1;
            int node2 = problem_.bars(i, 1) - 1;

            if (node1 < 0 || node1 >= problem_.n_nodes || node2 < 0 || node2 >= problem_.n_nodes)
            {
                cerr << "Error: Bar " << i << " has invalid node indices: " << node1 + 1 << ", " << node2 + 1 << endl;
                cerr << "Valid node range: 1 to " << problem_.n_nodes << endl;
                return {1e10, VectorXd::Zero(problem_.n_bars)};
            }
        }

        int n_dof = problem_.n_dof;
        int dim = (problem_.nodes.cols() == 2) ? 2 : 3;

        MatrixXd K = MatrixXd::Zero(n_dof, n_dof);

        // Build stiffness matrix
        for (int i = 0; i < problem_.n_bars; i++)
        {
            int node1 = problem_.bars(i, 0) - 1;
            int node2 = problem_.bars(i, 1) - 1;
            double A = areas[i];

            VectorXd coord1 = problem_.nodes.row(node1);
            VectorXd coord2 = problem_.nodes.row(node2);
            VectorXd delta = coord2 - coord1;
            double L = delta.norm();

            if (L < 1e-10)
                continue;

            VectorXd direction = delta / L;
            double k_val = (problem_.E * A) / L;

            if (dim == 2)
            {
                double cx = direction[0], cy = direction[1];
                MatrixXd ke(4, 4);
                ke << cx * cx, cx * cy, -cx * cx, -cx * cy,
                    cx * cy, cy * cy, -cx * cy, -cy * cy,
                    -cx * cx, -cx * cy, cx * cx, cx * cy,
                    -cx * cy, -cy * cy, cx * cy, cy * cy;
                ke *= k_val;

                vector<int> dofs = {node1 * 2, node1 * 2 + 1, node2 * 2, node2 * 2 + 1};
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        K(dofs[i], dofs[j]) += ke(i, j);
            }
            else
            {
                // 3D truss element
                double cx = direction[0], cy = direction[1], cz = direction[2];
                MatrixXd ke(6, 6);
                ke << cx * cx, cx * cy, cx * cz, -cx * cx, -cx * cy, -cx * cz,
                    cx * cy, cy * cy, cy * cz, -cx * cy, -cy * cy, -cy * cz,
                    cx * cz, cy * cz, cz * cz, -cx * cz, -cy * cz, -cz * cz,
                    -cx * cx, -cx * cy, -cx * cz, cx * cx, cx * cy, cx * cz,
                    -cx * cy, -cy * cy, -cy * cz, cx * cy, cy * cy, cy * cz,
                    -cx * cz, -cy * cz, -cz * cz, cx * cz, cy * cz, cz * cz;
                ke *= k_val;

                vector<int> dofs = {node1 * 3, node1 * 3 + 1, node1 * 3 + 2,
                                    node2 * 3, node2 * 3 + 1, node2 * 3 + 2};
                for (int i = 0; i < 6; i++)
                    for (int j = 0; j < 6; j++)
                        K(dofs[i], dofs[j]) += ke(i, j);
            }
        }

        // Apply boundary conditions
        vector<int> free_dofs;
        for (int i = 0; i < n_dof; i++)
        {
            if (find(problem_.fixed_dof.begin(), problem_.fixed_dof.end(), i) == problem_.fixed_dof.end())
            {
                free_dofs.push_back(i);
            }
        }

        int n_free = free_dofs.size();
        if (n_free == 0)
            return {0.0, VectorXd::Zero(problem_.n_bars)};

        MatrixXd K_free(n_free, n_free);
        VectorXd F_free(n_free);

        for (int i = 0; i < n_free; i++)
        {
            F_free(i) = problem_.f_vector(free_dofs[i]);
            for (int j = 0; j < n_free; j++)
            {
                K_free(i, j) = K(free_dofs[i], free_dofs[j]);
            }
        }

        // Add small regularization for stability
        K_free += MatrixXd::Identity(n_free, n_free) * 1e-12;

        // Solve
        VectorXd U_free = K_free.ldlt().solve(F_free);

        // Reconstruct full displacement vector
        VectorXd U = VectorXd::Zero(n_dof);
        for (int i = 0; i < n_free; i++)
        {
            U(free_dofs[i]) = U_free(i);
        }

        // Calculate stresses
        VectorXd stresses(problem_.n_bars);
        for (int i = 0; i < problem_.n_bars; i++)
        {
            int node1 = problem_.bars(i, 0) - 1;
            int node2 = problem_.bars(i, 1) - 1;

            VectorXd coord1 = problem_.nodes.row(node1);
            VectorXd coord2 = problem_.nodes.row(node2);
            VectorXd delta = coord2 - coord1;
            double L = delta.norm();
            if (L < 1e-10)
            {
                stresses(i) = 0.0;
                continue;
            }

            VectorXd direction = delta / L;

            VectorXd u1, u2;
            if (dim == 2)
            {
                u1 = U.segment(node1 * 2, 2);
                u2 = U.segment(node2 * 2, 2);
            }
            else
            {
                u1 = U.segment(node1 * 3, 3);
                u2 = U.segment(node2 * 3, 3);
            }

            VectorXd du = u2 - u1;
            double elongation = du.dot(direction);
            double strain = elongation / L;
            stresses(i) = problem_.E * strain;
        }

        double max_disp = U.array().abs().maxCoeff();
        return {max_disp, stresses};
    }

    double calculateWeight(const vector<double> &areas)
    {
        double weight = 0.0;
        for (int i = 0; i < problem_.n_bars; i++)
        {
            int node1 = problem_.bars(i, 0) - 1;
            int node2 = problem_.bars(i, 1) - 1;
            VectorXd coord1 = problem_.nodes.row(node1);
            VectorXd coord2 = problem_.nodes.row(node2);
            double L = (coord2 - coord1).norm();
            weight += problem_.rho * areas[i] * L;
        }
        return weight;
    }

    // Convert design variables to actual areas
    vector<double> getAreasFromVariables(const vector<double> &variables)
    {
        vector<double> areas(problem_.n_bars);
        for (int i = 0; i < problem_.n_bars; i++)
        {
            int var_index = problem_.member_groups[i];
            if (!problem_.areas.empty())
            {
                int area_index = round(variables[var_index]);
                area_index = max(0, min(area_index, (int)problem_.areas.size() - 1));
                areas[i] = problem_.areas[area_index];
            }
            else
            {
                areas[i] = variables[var_index];
            }
        }
        return areas;
    }

    // Improved function to evaluate 72-bar truss with multiple load cases - using average values
    void evaluate72BarWithMultipleLoadCases(Solution &sol, const vector<double> &areas)
    {
        // Store original force vector
        VectorXd original_f_vector = problem_.f_vector;

        // Define load cases for 72-bar truss
        vector<VectorXd> load_cases;

        // Load case 1: Top nodes with various forces
        VectorXd load_case1 = VectorXd::Zero(60);
        // Node 17: Fx = 5, Fy = 5, Fz = -5
        load_case1(48) = 5.0;  // Node 17, Fx
        load_case1(49) = 5.0;  // Node 17, Fy
        load_case1(50) = -5.0; // Node 17, Fz

        // Load case 2: All top nodes with vertical loads
        VectorXd load_case2 = VectorXd::Zero(60);
        // All top nodes (17-20) with Fz = -5
        load_case2(50) = -5.0; // Node 17, Fz
        load_case2(53) = -5.0; // Node 18, Fz
        load_case2(56) = -5.0; // Node 19, Fz
        load_case2(59) = -5.0; // Node 20, Fz

        load_cases.push_back(load_case1);
        load_cases.push_back(load_case2);

        // Evaluate for each load case
        sol.max_displacements_per_case.clear();
        sol.stresses_per_case.clear();
        sol.constraint_violations_per_case.clear();
        sol.feasible_per_case.clear();

        // Use average instead of worst-case for better continuity
        double avg_max_disp = 0.0;
        vector<double> avg_stresses(problem_.n_bars, 0.0);
        vector<double> max_constraint_violations(problem_.n_bars, 0.0);
        bool all_feasible = true;

        for (size_t lc = 0; lc < load_cases.size(); lc++)
        {
            // Apply current load case
            problem_.f_vector = load_cases[lc];

            auto [max_disp, stresses] = solveTruss(areas);

            // Check if the truss analysis failed
            if (max_disp >= 1e10)
            {
                // Truss analysis failed - set high penalty values
                sol.max_displacements_per_case.push_back(1e10);
                sol.stresses_per_case.push_back(vector<double>(problem_.n_bars, 1e10));

                vector<double> case_violations(problem_.n_bars, 1e10);
                sol.constraint_violations_per_case.push_back(case_violations);
                sol.feasible_per_case.push_back(false);

                avg_max_disp = 1e10;
                fill(avg_stresses.begin(), avg_stresses.end(), 1e10);
                fill(max_constraint_violations.begin(), max_constraint_violations.end(), 1e10);
                all_feasible = false;
                continue;
            }

            sol.max_displacements_per_case.push_back(max_disp);
            sol.stresses_per_case.push_back(vector<double>(stresses.data(), stresses.data() + stresses.size()));

            // Calculate constraint violations for this load case
            vector<double> case_violations(problem_.n_bars);
            bool case_feasible = true;

            avg_max_disp += max_disp;

            for (int i = 0; i < problem_.n_bars; i++)
            {
                avg_stresses[i] += abs(stresses[i]);
                double stress_violation = max(0.0, abs(stresses[i]) - problem_.s_adm);
                case_violations[i] = stress_violation;
                max_constraint_violations[i] = max(max_constraint_violations[i], stress_violation);

                if (stress_violation > 1e-6)
                {
                    case_feasible = false;
                    all_feasible = false;
                }
            }

            sol.constraint_violations_per_case.push_back(case_violations);
            sol.feasible_per_case.push_back(case_feasible);
        }

        // Restore original force vector
        problem_.f_vector = original_f_vector;

        // Use averages for objectives, worst-case for constraints
        if (all_feasible && !load_cases.empty())
        {
            avg_max_disp /= load_cases.size();
            for (int i = 0; i < problem_.n_bars; i++)
            {
                avg_stresses[i] /= load_cases.size();
            }
        }

        sol.max_displacement = avg_max_disp;
        sol.stresses = avg_stresses;
        sol.constraint_violations = max_constraint_violations;
        sol.feasible = all_feasible;
    }

    // Evaluate solution
    void evaluateSolution(Solution &sol)
    {
        vector<double> areas = getAreasFromVariables(sol.variables);
        sol.weight = calculateWeight(areas);

        // For 72-bar truss, we need to handle multiple load cases
        if (problem_.n_bars == 72)
        {
            evaluate72BarWithMultipleLoadCases(sol, areas);
        }
        else
        {
            // Original single load case evaluation for other problems
            auto [max_disp, stresses] = solveTruss(areas);
            sol.max_displacement = max_disp;
            sol.stresses = vector<double>(stresses.data(), stresses.data() + stresses.size());

            // Calculate constraint violations
            sol.constraint_violations.resize(problem_.n_bars);
            sol.feasible = true;
            for (int i = 0; i < problem_.n_bars; i++)
            {
                double stress_violation = max(0.0, abs(sol.stresses[i]) - problem_.s_adm);
                sol.constraint_violations[i] = stress_violation;
                if (stress_violation > 1e-6)
                {
                    sol.feasible = false;
                }
            }
        }

        sol.objectives = {sol.weight, sol.max_displacement};
    }

    // More relaxed APM fitness for 72-bar truss
    void calculateAPMFitnessRelaxed(vector<Solution> &population)
    {
        int m = problem_.n_bars;

        // Calculate averages with focus on feasible solutions
        double avg_weight = 0.0, avg_disp = 0.0;
        vector<double> avg_violations(m, 0.0);
        int total_count = population.size();

        for (const auto &sol : population)
        {
            avg_weight += sol.weight;
            avg_disp += sol.max_displacement;

            for (int j = 0; j < m; j++)
            {
                avg_violations[j] += sol.constraint_violations[j];
            }
        }

        if (total_count > 0)
        {
            avg_weight /= total_count;
            avg_disp /= total_count;
            for (int j = 0; j < m; j++)
            {
                avg_violations[j] /= total_count;
            }
        }

        // Calculate violation norm with regularization to avoid division by zero
        double violation_norm = 1e-10;
        for (int j = 0; j < m; j++)
        {
            violation_norm += avg_violations[j] * avg_violations[j];
        }
        violation_norm = sqrt(violation_norm);

        // Calculate penalty coefficients with VERY reduced magnitude for 72-bar
        vector<double> k_j(m, 0.0);
        if (violation_norm > 1e-10)
        {
            for (int j = 0; j < m; j++)
            {
                // Much reduced penalty coefficient for 72-bar truss
                k_j[j] = 0.01 * (abs(avg_weight) / violation_norm) * (avg_violations[j] / violation_norm);
            }
        }

        // Calculate fitness for each solution with very relaxed penalties
        for (auto &sol : population)
        {
            if (sol.feasible)
            {
                sol.fitness_weight = sol.weight;
                sol.fitness_displacement = sol.max_displacement;
            }
            else
            {
                // Calculate total violation with minimal impact
                double total_violation = 0.0;
                for (int j = 0; j < m; j++)
                {
                    total_violation += k_j[j] * sol.constraint_violations[j];
                }

                // Use very soft penalty for infeasible solutions
                double base_weight = max(sol.weight, avg_weight * 0.5);
                sol.fitness_weight = base_weight + total_violation * 0.1;

                double base_disp = max(sol.max_displacement, avg_disp * 0.5);
                sol.fitness_displacement = base_disp + total_violation * 0.1;
            }
        }
    }

    // APM fitness calculation according to the article
    void calculateAPMFitness(vector<Solution> &population)
    {
        // For 72-bar truss, use a more relaxed constraint handling
        if (problem_.n_bars == 72)
        {
            calculateAPMFitnessRelaxed(population);
            return;
        }

        int m = problem_.n_bars; // number of constraints

        // Calculate averages for current population
        double avg_weight = 0.0, avg_disp = 0.0;
        vector<double> avg_violations(m, 0.0);
        int count = 0;

        for (const auto &sol : population)
        {
            avg_weight += sol.weight;
            avg_disp += sol.max_displacement;
            for (int j = 0; j < m; j++)
            {
                avg_violations[j] += sol.constraint_violations[j];
            }
            count++;
        }

        if (count > 0)
        {
            avg_weight /= count;
            avg_disp /= count;
            for (int j = 0; j < m; j++)
            {
                avg_violations[j] /= count;
            }
        }

        // Calculate violation norm
        double violation_norm = 0.0;
        for (int j = 0; j < m; j++)
        {
            violation_norm += avg_violations[j] * avg_violations[j];
        }
        violation_norm = sqrt(violation_norm);

        // Calculate penalty coefficients
        vector<double> k_j(m, 0.0);
        if (violation_norm > 1e-10)
        {
            for (int j = 0; j < m; j++)
            {
                k_j[j] = (abs(avg_weight) / violation_norm) * (avg_violations[j] / violation_norm);
            }
        }

        // Calculate fitness for each solution
        for (auto &sol : population)
        {
            if (sol.feasible)
            {
                sol.fitness_weight = sol.weight;
                sol.fitness_displacement = sol.max_displacement;
            }
            else
            {
                // Calculate total violation
                double total_violation = 0.0;
                for (int j = 0; j < m; j++)
                {
                    total_violation += k_j[j] * sol.constraint_violations[j];
                }

                // Weight fitness according to APM formula
                double base_weight = (sol.weight > avg_weight) ? sol.weight : avg_weight;
                sol.fitness_weight = base_weight + total_violation;

                // Displacement fitness according to APM formula
                double base_disp = (sol.max_displacement > avg_disp) ? sol.max_displacement : avg_disp;
                sol.fitness_displacement = base_disp + total_violation;
            }
        }
    }

    // DE mutation and crossover with SaDE parameters
    vector<double> createTrialVector(int target_idx, const vector<Solution> &population, double CR_i, double F_i)
    {
        int n = problem_.n_vars;
        vector<double> trial = population[target_idx].variables;

        // Select three distinct random individuals
        uniform_int_distribution<int> dist(0, population.size() - 1);
        int r1, r2, r3;
        do
        {
            r1 = dist(rng_);
        } while (r1 == target_idx);
        do
        {
            r2 = dist(rng_);
        } while (r2 == r1 || r2 == target_idx);
        do
        {
            r3 = dist(rng_);
        } while (r3 == r1 || r3 == r2 || r3 == target_idx);

        // Ensure at least one dimension is crossed over
        uniform_int_distribution<int> j_rand_dist(0, n - 1);
        int J = j_rand_dist(rng_);

        uniform_real_distribution<double> rand_dist(0.0, 1.0);

        for (int j = 0; j < n; j++)
        {
            if (rand_dist(rng_) < CR_i || j == J)
            {
                // Mutation: u = x_r1 + F * (x_r2 - x_r3)
                double continuous_val = population[r1].variables[j] +
                                        F_i * (population[r2].variables[j] - population[r3].variables[j]);

                // Handle discrete variables
                if (!problem_.areas.empty())
                {
                    int index = round(continuous_val);
                    index = max(0, min(index, (int)problem_.areas.size() - 1));
                    trial[j] = index;
                }
                else
                {
                    // For continuous variables, apply bounds
                    double min_val = 0.1;
                    double max_val = problem_.areas.empty() ? 30.0 : problem_.areas.size() - 1;
                    trial[j] = max(min_val, min(continuous_val, max_val));
                }
            }
        }

        return trial;
    }

    // Check domination using APM fitness values
    bool dominatesByFitness(const Solution &a, const Solution &b)
    {
        bool better_or_equal = (a.fitness_weight <= b.fitness_weight) &&
                               (a.fitness_displacement <= b.fitness_displacement);
        bool strictly_better = (a.fitness_weight < b.fitness_weight) ||
                               (a.fitness_displacement < b.fitness_displacement);
        return better_or_equal && strictly_better;
    }

    // Domination check based on original objectives
    bool dominatesByObjectives(const Solution &a, const Solution &b)
    {
        // For feasible solutions, use standard Pareto dominance
        if (a.feasible && b.feasible)
        {
            bool better_or_equal = (a.weight <= b.weight) &&
                                   (a.max_displacement <= b.max_displacement);
            bool strictly_better = (a.weight < b.weight) ||
                                   (a.max_displacement < b.max_displacement);
            return better_or_equal && strictly_better;
        }
        // Feasible dominates infeasible
        else if (a.feasible && !b.feasible)
        {
            return true;
        }
        // Infeasible doesn't dominate feasible
        else if (!a.feasible && b.feasible)
        {
            return false;
        }
        // Both infeasible - use constraint violation
        else
        {
            double total_viol_a = 0.0, total_viol_b = 0.0;
            for (double v : a.constraint_violations)
                total_viol_a += v;
            for (double v : b.constraint_violations)
                total_viol_b += v;
            return total_viol_a < total_viol_b;
        }
    }

    // Non-dominated sorting based on APM fitness
    void nonDominatedSort(vector<Solution> &solutions)
    {
        int n = solutions.size();
        if (n == 0)
            return;

        vector<vector<int>> domination_list(n);
        vector<int> domination_count(n, 0);

        // Calculate domination relationships
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (dominatesByFitness(solutions[i], solutions[j]))
                {
                    domination_list[i].push_back(j);
                    domination_count[j]++;
                }
                else if (dominatesByFitness(solutions[j], solutions[i]))
                {
                    domination_list[j].push_back(i);
                    domination_count[i]++;
                }
            }
        }

        // Assign ranks
        vector<int> current_front;
        for (int i = 0; i < n; i++)
        {
            solutions[i].rank = 0;
            if (domination_count[i] == 0)
            {
                current_front.push_back(i);
                solutions[i].rank = 1;
            }
        }

        int rank = 1;
        while (!current_front.empty())
        {
            vector<int> next_front;
            for (int i : current_front)
            {
                for (int j : domination_list[i])
                {
                    domination_count[j]--;
                    if (domination_count[j] == 0)
                    {
                        next_front.push_back(j);
                        solutions[j].rank = rank + 1;
                    }
                }
            }
            current_front = next_front;
            rank++;
        }
    }

    // Non-dominated sorting based on original objectives (weight and displacement)
    void nonDominatedSortByObjectives(vector<Solution> &solutions)
    {
        int n = solutions.size();
        if (n == 0)
            return;

        vector<vector<int>> domination_list(n);
        vector<int> domination_count(n, 0);

        // Calculate domination relationships based on original objectives
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (dominatesByObjectives(solutions[i], solutions[j]))
                {
                    domination_list[i].push_back(j);
                    domination_count[j]++;
                }
                else if (dominatesByObjectives(solutions[j], solutions[i]))
                {
                    domination_list[j].push_back(i);
                    domination_count[i]++;
                }
            }
        }

        // Assign ranks
        vector<int> current_front;
        for (int i = 0; i < n; i++)
        {
            solutions[i].rank = 0;
            if (domination_count[i] == 0)
            {
                current_front.push_back(i);
                solutions[i].rank = 1;
            }
        }

        int rank = 1;
        while (!current_front.empty())
        {
            vector<int> next_front;
            for (int i : current_front)
            {
                for (int j : domination_list[i])
                {
                    domination_count[j]--;
                    if (domination_count[j] == 0)
                    {
                        next_front.push_back(j);
                        solutions[j].rank = rank + 1;
                    }
                }
            }
            current_front = next_front;
            rank++;
        }
    }

    // Crowding distance calculation
    void calculateCrowdingDistance(vector<Solution> &solutions)
    {
        int n = solutions.size();
        if (n == 0)
            return;

        for (auto &sol : solutions)
        {
            sol.crowding_distance = 0.0;
        }

        // Sort by weight fitness
        sort(solutions.begin(), solutions.end(),
             [](const Solution &a, const Solution &b)
             {
                 return a.fitness_weight < b.fitness_weight;
             });

        solutions[0].crowding_distance = numeric_limits<double>::infinity();
        solutions[n - 1].crowding_distance = numeric_limits<double>::infinity();

        double min_w = solutions[0].fitness_weight;
        double max_w = solutions[n - 1].fitness_weight;
        double range_w = max_w - min_w;

        if (range_w > 1e-10)
        {
            for (int i = 1; i < n - 1; i++)
            {
                solutions[i].crowding_distance +=
                    (solutions[i + 1].fitness_weight - solutions[i - 1].fitness_weight) / range_w;
            }
        }

        // Sort by displacement fitness
        sort(solutions.begin(), solutions.end(),
             [](const Solution &a, const Solution &b)
             {
                 return a.fitness_displacement < b.fitness_displacement;
             });

        solutions[0].crowding_distance = numeric_limits<double>::infinity();
        solutions[n - 1].crowding_distance = numeric_limits<double>::infinity();

        double min_d = solutions[0].fitness_displacement;
        double max_d = solutions[n - 1].fitness_displacement;
        double range_d = max_d - min_d;

        if (range_d > 1e-10)
        {
            for (int i = 1; i < n - 1; i++)
            {
                solutions[i].crowding_distance +=
                    (solutions[i + 1].fitness_displacement - solutions[i - 1].fitness_displacement) / range_d;
            }
        }
    }

    // Crowding distance based on original objectives
    void calculateCrowdingDistanceByObjectives(vector<Solution> &solutions)
    {
        int n = solutions.size();
        if (n == 0)
            return;

        for (auto &sol : solutions)
        {
            sol.crowding_distance = 0.0;
        }

        // Sort by weight
        sort(solutions.begin(), solutions.end(),
             [](const Solution &a, const Solution &b)
             {
                 return a.weight < b.weight;
             });

        solutions[0].crowding_distance = numeric_limits<double>::infinity();
        solutions[n - 1].crowding_distance = numeric_limits<double>::infinity();

        double min_w = solutions[0].weight;
        double max_w = solutions[n - 1].weight;
        double range_w = max_w - min_w;

        if (range_w > 1e-10)
        {
            for (int i = 1; i < n - 1; i++)
            {
                solutions[i].crowding_distance +=
                    (solutions[i + 1].weight - solutions[i - 1].weight) / range_w;
            }
        }

        // Sort by displacement
        sort(solutions.begin(), solutions.end(),
             [](const Solution &a, const Solution &b)
             {
                 return a.max_displacement < b.max_displacement;
             });

        solutions[0].crowding_distance = numeric_limits<double>::infinity();
        solutions[n - 1].crowding_distance = numeric_limits<double>::infinity();

        double min_d = solutions[0].max_displacement;
        double max_d = solutions[n - 1].max_displacement;
        double range_d = max_d - min_d;

        if (range_d > 1e-10)
        {
            for (int i = 1; i < n - 1; i++)
            {
                solutions[i].crowding_distance +=
                    (solutions[i + 1].max_displacement - solutions[i - 1].max_displacement) / range_d;
            }
        }
    }

    // Selection using crowding distance - UNIFIED FOR ALL PROBLEMS
    void selectWithCrowdingDistance(vector<Solution> &solutions, int target_size)
    {
        if (solutions.size() <= target_size)
            return;

        // Use unified selection for all problems
        nonDominatedSort(solutions);
        calculateCrowdingDistance(solutions);

        sort(solutions.begin(), solutions.end(),
             [](const Solution &a, const Solution &b)
             {
                 if (a.rank != b.rank)
                     return a.rank < b.rank;
                 return a.crowding_distance > b.crowding_distance;
             });

        solutions.resize(target_size);
    }

    // Efficient domination check for archive update
    bool isDominatedByArchive(const Solution &sol, const vector<Solution> &archive)
    {
        for (const auto &archive_sol : archive)
        {
            if (dominatesByFitness(archive_sol, sol))
            {
                return true;
            }
        }
        return false;
    }

    // Remove solutions from archive that are dominated by new solution
    void removeDominatedFromArchive(const Solution &sol, vector<Solution> &archive)
    {
        vector<Solution> new_archive;
        for (const auto &archive_sol : archive)
        {
            if (!dominatesByFitness(sol, archive_sol))
            {
                new_archive.push_back(archive_sol);
            }
        }
        archive = new_archive;
    }

    void removeDuplicatesFromVector(vector<Solution> &solutions)
    {
        vector<Solution> unique_solutions;
        vector<vector<double>> seen_variables;

        for (const auto &sol : solutions)
        {
            bool is_duplicate = false;
            for (const auto &seen : seen_variables)
            {
                if (vectorsEqual(sol.variables, seen, 1e-6))
                {
                    is_duplicate = true;
                    break;
                }
            }

            if (!is_duplicate)
            {
                unique_solutions.push_back(sol);
                seen_variables.push_back(sol.variables);
            }
        }

        solutions = unique_solutions;
    }

    bool vectorsEqual(const vector<double> &a, const vector<double> &b, double tolerance)
    {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); i++)
        {
            if (abs(a[i] - b[i]) > tolerance)
                return false;
        }
        return true;
    }

    // Remove duplicate solutions from archive
    void removeDuplicatesFromArchive()
    {
        removeDuplicatesFromVector(external_archive_);
    }

    // Check if a solution is already in the archive using variables comparison
    bool isSolutionInArchive(const Solution &sol, const vector<Solution> &archive)
    {
        for (const auto &arch_sol : archive)
        {
            if (vectorsEqual(sol.variables, arch_sol.variables, 1e-6))
            {
                return true;
            }
        }
        return false;
    }

    // Improved archive update without hard limits
    void updateExternalArchive()
    {
        vector<Solution> combined = external_archive_;
        combined.insert(combined.end(), population_.begin(), population_.end());

        removeDuplicatesFromVector(combined);

        // Use non-dominated sorting based on original objectives
        nonDominatedSortByObjectives(combined);
        calculateCrowdingDistanceByObjectives(combined);

        // Keep only non-dominated solutions (rank 1)
        external_archive_.clear();
        for (const auto &sol : combined)
        {
            if (sol.rank == 1)
            {
                external_archive_.push_back(sol);
            }
        }

        // Remove duplicates again to be safe
        removeDuplicatesFromVector(external_archive_);
    }

    // Main optimization loop with SaDE
    void optimize()
    {
        cout << "Initializing population with SaDE..." << endl;

        // Initialize population
        population_.resize(problem_.n_pop);
        double max_index = problem_.areas.empty() ? 30.0 : problem_.areas.size() - 1;

        // For 72-bar truss, use much wider initial distribution
        if (problem_.n_bars == 72 && !problem_.areas.empty())
        {
            max_index = problem_.areas.size() - 1;
            uniform_real_distribution<double> init_dist(0.0, max_index);

            for (auto &sol : population_)
            {
                sol.variables.resize(problem_.n_vars);
                for (auto &var : sol.variables)
                {
                    var = init_dist(rng_);
                    var = max(0.0, min(var, max_index));
                }
                evaluateSolution(sol);
            }
        }
        else
        {
            uniform_real_distribution<double> init_dist(1.0, max_index);
            for (auto &sol : population_)
            {
                sol.variables.resize(problem_.n_vars);
                for (auto &var : sol.variables)
                {
                    var = init_dist(rng_);
                }
                evaluateSolution(sol);
            }
        }

        // Calculate initial APM fitness
        calculateAPMFitness(population_);

        // Initialize archive with entire population
        external_archive_ = population_;
        removeDuplicatesFromArchive();

        cout << "Starting SaDE optimization..." << endl;

        // Main optimization loop
        for (int gen = 0; gen < problem_.gen; gen++)
        {
            vector<Solution> new_population;

            // For each individual in current population - UNIFIED SELECTION
            for (int i = 0; i < problem_.n_pop; i++)
            {
                // Generate SaDE parameters for this individual
                double CR_i = generateCR();
                double F_i = generateF();

                // Create trial vector with SaDE parameters
                vector<double> trial_vars = createTrialVector(i, population_, CR_i, F_i);

                // Evaluate trial solution
                Solution trial_sol;
                trial_sol.variables = trial_vars;
                trial_sol.used_CR = CR_i;
                trial_sol.used_F = F_i;
                evaluateSolution(trial_sol);

                // UNIFIED APM-BASED SELECTION FOR ALL PROBLEMS
                vector<Solution> temp_pop = population_;
                temp_pop.push_back(trial_sol);
                calculateAPMFitness(temp_pop);
                trial_sol.fitness_weight = temp_pop.back().fitness_weight;
                trial_sol.fitness_displacement = temp_pop.back().fitness_displacement;

                bool trial_dominates = (trial_sol.fitness_weight <= population_[i].fitness_weight) &&
                                       (trial_sol.fitness_displacement <= population_[i].fitness_displacement);
                bool current_dominates = (population_[i].fitness_weight <= trial_sol.fitness_weight) &&
                                         (population_[i].fitness_displacement <= trial_sol.fitness_displacement);

                if (trial_dominates && !current_dominates)
                {
                    new_population.push_back(trial_sol);
                    recordSuccessfulParameters(CR_i, F_i);
                }
                else if (current_dominates && !trial_dominates)
                {
                    new_population.push_back(population_[i]);
                    recordUnsuccessfulParameters(CR_i, F_i);
                }
                else
                {
                    // Keep both and let crowding distance handle it later
                    new_population.push_back(population_[i]);
                    new_population.push_back(trial_sol);
                    // Consider this as neutral - don't record for adaptation
                }
            }

            // Reduce population to N using proper selection
            calculateAPMFitness(new_population);
            selectWithCrowdingDistance(new_population, problem_.n_pop);
            population_ = new_population;

            // Update SaDE memory every learning period
            if (gen % problem_.LP == 0 && gen > 0)
            {
                updateSaDEMemory();
                cout << "Generation " << gen << ": Updated SaDE memory. Current CR memory: [";
                for (size_t i = 0; i < M_CR.size(); i++)
                {
                    cout << M_CR[i];
                    if (i < M_CR.size() - 1)
                        cout << ", ";
                }
                cout << "], F memory: [";
                for (size_t i = 0; i < M_F.size(); i++)
                {
                    cout << M_F[i];
                    if (i < M_F.size() - 1)
                        cout << ", ";
                }
                cout << "]" << endl;
            }

            // Update external archive every generation
            updateExternalArchive();

            // Progress reporting
            if (gen % 100 == 0)
            {
                int feasible_count = 0;
                double avg_weight = 0.0, avg_disp = 0.0;
                for (const auto &sol : population_)
                {
                    if (sol.feasible)
                        feasible_count++;
                    avg_weight += sol.weight;
                    avg_disp += sol.max_displacement;
                }
                avg_weight /= population_.size();
                avg_disp /= population_.size();

                // Count solutions by rank in archive
                vector<int> rank_count(10, 0);
                for (const auto &sol : external_archive_)
                {
                    if (sol.rank < 10)
                    {
                        rank_count[sol.rank]++;
                    }
                }

                cout << "Generation " << gen << ": " << feasible_count << "/" << problem_.n_pop
                     << " feasible, Avg weight: " << avg_weight << ", Avg disp: " << avg_disp
                     << ", Archive: " << external_archive_.size() << " solutions";

                // Show rank distribution
                cout << " [Ranks: ";
                for (int i = 1; i <= 3; i++)
                {
                    if (rank_count[i] > 0)
                    {
                        cout << "R" << i << ":" << rank_count[i] << " ";
                    }
                }
                cout << "]" << endl;
            }
        }

        // Final archive update
        updateExternalArchive();

        int feasible_final = 0;
        for (const auto &sol : external_archive_)
        {
            if (sol.feasible)
                feasible_final++;
        }

        cout << "SaDE optimization completed!" << endl;
        cout << "Final archive: " << external_archive_.size() << " solutions ("
             << feasible_final << " feasible)" << endl;

        // Print final SaDE memory state
        cout << "Final SaDE CR memory: [";
        for (size_t i = 0; i < M_CR.size(); i++)
        {
            cout << M_CR[i];
            if (i < M_CR.size() - 1)
                cout << ", ";
        }
        cout << "]" << endl;
        cout << "Final SaDE F memory: [";
        for (size_t i = 0; i < M_F.size(); i++)
        {
            cout << M_F[i];
            if (i < M_F.size() - 1)
                cout << ", ";
        }
        cout << "]" << endl;
    }

    // Export results to CSV - optimized for large archives
    void exportResults(const string &filename)
    {
        ofstream file(filename);
        file << "weight,max_displacement,feasible,rank,crowding_distance,used_CR,used_F,variables" << endl;

        // Sort by weight for better visualization
        vector<Solution> sorted_archive = external_archive_;
        sort(sorted_archive.begin(), sorted_archive.end(),
             [](const Solution &a, const Solution &b)
             {
                 return a.weight < b.weight;
             });

        for (const auto &sol : sorted_archive)
        {
            file << sol.weight << "," << sol.max_displacement << ","
                 << (sol.feasible ? "true" : "false") << ","
                 << sol.rank << "," << sol.crowding_distance << ","
                 << sol.used_CR << "," << sol.used_F << ",";
            for (size_t i = 0; i < sol.variables.size(); i++)
            {
                file << sol.variables[i];
                if (i < sol.variables.size() - 1)
                    file << ";";
            }
            file << endl;
        }
        file.close();
        cout << "Results exported to " << filename << " (" << external_archive_.size() << " solutions)" << endl;
    }

    const vector<Solution> &getParetoFront() const { return external_archive_; }
};

// Create 10-bar truss problem
TrussProblem create10BarTruss()
{
    TrussProblem problem;

    problem.n_bars = 10;
    problem.n_nodes = 6;
    problem.n_dof = 12;
    problem.n_vars = 10;
    problem.s_adm = 25.0;

    // Available areas
    vector<double> areas = {
        1.62, 1.80, 1.99, 2.13, 2.38, 2.62, 2.63, 2.88, 2.93, 3.09,
        3.13, 3.38, 3.47, 3.55, 3.63, 3.84, 3.87, 3.88, 4.18, 4.22,
        4.49, 4.59, 4.80, 4.97, 5.12, 5.74, 7.22, 7.97, 11.50, 13.50,
        13.90, 14.20, 15.50, 16.00, 16.90, 18.80, 19.90, 22.00, 22.90,
        26.50, 30.00, 33.50};
    problem.areas = areas;

    // Node coordinates
    problem.nodes = MatrixXd(6, 2);
    problem.nodes << 360, 360, // Node 1
        360, 0,                // Node 2
        0, 360,                // Node 3
        0, 0,                  // Node 4
        720, 360,              // Node 5
        720, 0;                // Node 6

    // Bar connectivity
    problem.bars = MatrixXi(10, 2);
    problem.bars << 1, 2, // Bar 1
        1, 4,             // Bar 2
        2, 3,             // Bar 3
        2, 4,             // Bar 4
        2, 5,             // Bar 5
        3, 4,             // Bar 6
        3, 6,             // Bar 7
        4, 5,             // Bar 8
        4, 6,             // Bar 9
        5, 6;             // Bar 10

    // Force vector
    problem.f_vector = VectorXd(12);
    problem.f_vector << 0, 0, // Node 1
        0, -100,              // Node 2
        0, 0,                 // Node 3
        0, -100,              // Node 4
        0, 0,                 // Node 5
        0, 0;                 // Node 6

    // Fixed DOFs
    problem.fixed_dof = {4, 5, 10, 11}; // Node 3 (DOF 4,5) and Node 6 (DOF 10,11)

    // Member groups (each bar has its own variable)
    problem.member_groups = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Optimization parameters
    problem.n_pop = 50;
    problem.gen = 1000;
    problem.CR = 0.5; // Initial value for SaDE
    problem.F = 0.5;  // Initial value for SaDE

    return problem;
}

// Create 25-bar truss problem
TrussProblem create25BarTruss()
{
    TrussProblem problem;

    problem.n_bars = 25;
    problem.n_nodes = 10;
    problem.n_dof = 30;
    problem.n_vars = 8;
    problem.s_adm = 40.0;

    // Available areas
    vector<double> areas = {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
        2.5, 2.6, 2.8, 3.0, 3.2, 3.4};
    problem.areas = areas;

    // Node coordinates (3D)
    problem.nodes = MatrixXd(10, 3);
    problem.nodes << 0, 0, 0, // Node 1
        0, 75, 0,             // Node 2
        0, 150, 0,            // Node 3
        75, 75, 75,           // Node 4
        75, 75, -75,          // Node 5
        -75, 75, 75,          // Node 6
        -75, 75, -75,         // Node 7
        0, 150, 150,          // Node 8
        0, 150, -150,         // Node 9
        0, 225, 0;            // Node 10

    // Bar connectivity
    problem.bars = MatrixXi(25, 2);
    problem.bars << 1, 2, // Bar 1
        1, 4,             // Bar 2
        1, 6,             // Bar 3
        2, 3,             // Bar 4
        2, 4,             // Bar 5
        2, 5,             // Bar 6
        2, 6,             // Bar 7
        2, 7,             // Bar 8
        3, 4,             // Bar 9
        3, 5,             // Bar 10
        3, 8,             // Bar 11
        3, 9,             // Bar 12
        4, 5,             // Bar 13
        4, 6,             // Bar 14
        4, 8,             // Bar 15
        4, 10,            // Bar 16
        5, 7,             // Bar 17
        5, 9,             // Bar 18
        5, 10,            // Bar 19
        6, 7,             // Bar 20
        6, 8,             // Bar 21
        6, 10,            // Bar 22
        7, 9,             // Bar 23
        7, 10,            // Bar 24
        8, 9;             // Bar 25

    // Force vector (3D - 30 DOFs)
    problem.f_vector = VectorXd(30);
    problem.f_vector << 0, 0, 0, // Node 1
        0, 0, -20,               // Node 2
        0, 0, -20,               // Node 3
        0, 0, -20,               // Node 4
        0, 0, -20,               // Node 5
        0, 0, -20,               // Node 6
        0, 0, -20,               // Node 7
        0, 0, 0,                 // Node 8
        0, 0, 0,                 // Node 9
        0, 0, -20;               // Node 10

    // Fixed DOFs (Nodes 8, 9, 10 - last 9 DOFs)
    problem.fixed_dof = {21, 22, 23, 24, 25, 26, 27, 28, 29};

    // Member groups (8 symmetry groups)
    problem.member_groups = {
        0, // Bar 1: Group 0
        1, // Bar 2: Group 1
        1, // Bar 3: Group 1
        2, // Bar 4: Group 2
        3, // Bar 5: Group 3
        3, // Bar 6: Group 3
        4, // Bar 7: Group 4
        4, // Bar 8: Group 4
        5, // Bar 9: Group 5
        5, // Bar 10: Group 5
        6, // Bar 11: Group 6
        6, // Bar 12: Group 6
        7, // Bar 13: Group 7
        7, // Bar 14: Group 7
        7, // Bar 15: Group 7
        7, // Bar 16: Group 7
        7, // Bar 17: Group 7
        7, // Bar 18: Group 7
        7, // Bar 19: Group 7
        7, // Bar 20: Group 7
        7, // Bar 21: Group 7
        7, // Bar 22: Group 7
        7, // Bar 23: Group 7
        7, // Bar 24: Group 7
        7  // Bar 25: Group 7
    };

    // Optimization parameters
    problem.n_pop = 50;
    problem.gen = 1000;
    problem.CR = 0.5; // Initial value for SaDE
    problem.F = 0.5;  // Initial value for SaDE

    return problem;
}

// Create 60-bar truss problem (ring structure)
TrussProblem create60BarTruss()
{
    TrussProblem problem;

    problem.n_bars = 60;
    problem.n_nodes = 24; // Ring structure with 24 nodes
    problem.n_dof = 48;   // 2 DOF per node
    problem.n_vars = 25;  // 25 design variable groups
    problem.s_adm = 60.0; // Allowable stress 60 ksi

    // Available areas: 0.5 to 4.9 in steps of 0.1 (45 options)
    vector<double> areas;
    for (double a = 0.5; a <= 4.9; a += 0.1)
    {
        areas.push_back(a);
    }
    problem.areas = areas;

    // Node coordinates for ring structure
    // Outer radius = 100 in, inner radius = 90 in
    // We'll create two concentric rings with 12 nodes each
    problem.nodes = MatrixXd(24, 2);
    double outer_radius = 100.0;
    double inner_radius = 90.0;

    // Outer ring nodes (1-12)
    for (int i = 0; i < 12; i++)
    {
        double angle = 2.0 * M_PI * i / 12.0;
        problem.nodes(i, 0) = outer_radius * cos(angle);
        problem.nodes(i, 1) = outer_radius * sin(angle);
    }

    // Inner ring nodes (13-24)
    for (int i = 0; i < 12; i++)
    {
        double angle = 2.0 * M_PI * i / 12.0;
        problem.nodes(i + 12, 0) = inner_radius * cos(angle);
        problem.nodes(i + 12, 1) = inner_radius * sin(angle);
    }

    // Bar connectivity - FIXED: Ensure all indices are within bounds
    problem.bars = MatrixXi(60, 2);

    // Outer ring circumferential bars (bars 1-12)
    for (int i = 0; i < 12; i++)
    {
        problem.bars(i, 0) = i + 1;
        problem.bars(i, 1) = ((i + 1) % 12) + 1;
    }

    // Inner ring circumferential bars (bars 13-24)
    for (int i = 0; i < 12; i++)
    {
        problem.bars(i + 12, 0) = i + 13;
        problem.bars(i + 12, 1) = ((i + 1) % 12) + 13;
    }

    // Radial bars connecting outer and inner rings (bars 25-36)
    for (int i = 0; i < 12; i++)
    {
        problem.bars(i + 24, 0) = i + 1;
        problem.bars(i + 24, 1) = i + 13;
    }

    // Diagonal bars in the ring structure (bars 37-60) - FIXED: Only 24 bars, not 48
    int bar_idx = 36;
    for (int i = 0; i < 12; i++)
    {
        // Only 2 bars per iteration, not 4
        problem.bars(bar_idx, 0) = i + 1;
        problem.bars(bar_idx, 1) = ((i + 1) % 12) + 13;
        bar_idx++;

        problem.bars(bar_idx, 0) = i + 13;
        problem.bars(bar_idx, 1) = ((i + 1) % 12) + 1;
        bar_idx++;
    }

    // Force vector - based on the loading data from Table 4
    problem.f_vector = VectorXd::Zero(48); // 24 nodes * 2 DOF

    // Load case 1: Node 1: Fx = -10.0, Fy = 0
    problem.f_vector(0) = -10.0; // Node 1, Fx (index 0)
    problem.f_vector(1) = 0.0;   // Node 1, Fy (index 1)

    // Load case 1: Node 7: Fx = 9.0, Fy = 0
    problem.f_vector(12) = 9.0; // Node 7, Fx (node 7 = index 6, 6*2=12)
    problem.f_vector(13) = 0.0; // Node 7, Fy

    // Load case 2: Node 15: Fx = -8.0, Fy = 3.0
    problem.f_vector(28) = -8.0; // Node 15, Fx (node 15 = index 14, 14*2=28)
    problem.f_vector(29) = 3.0;  // Node 15, Fy

    // Load case 2: Node 18: Fx = -8.0, Fy = 3.0
    problem.f_vector(34) = -8.0; // Node 18, Fx (node 18 = index 17, 17*2=34)
    problem.f_vector(35) = 3.0;  // Node 18, Fy

    // Load case 3: Node 22: Fx = -20.0, Fy = 10.0
    problem.f_vector(42) = -20.0; // Node 22, Fx (node 22 = index 21, 21*2=42)
    problem.f_vector(43) = 10.0;  // Node 22, Fy

    // Fixed DOFs - assume the ring is fixed at the bottom (nodes 6, 7, 18, 19)
    // This is a typical support configuration for ring structures
    vector<int> fixed_nodes = {5, 6, 17, 18}; // 0-based node indices
    problem.fixed_dof.clear();
    for (int node : fixed_nodes)
    {
        problem.fixed_dof.push_back(node * 2);     // x-dof
        problem.fixed_dof.push_back(node * 2 + 1); // y-dof
    }

    // Member groups - based on Table 5 - FIXED: Corrected indexing
    problem.member_groups.resize(60);

    // Group A1: bars 49 to 60 (indices 48-59 in 0-based)
    for (int i = 48; i < 60; i++)
    {
        problem.member_groups[i] = 0; // Group 0
    }

    // Groups A2 to A13: bars 1-24 in pairs (indices 0-23 in 0-based)
    for (int i = 0; i < 12; i++)
    {
        // A2: bars 1 and 13 -> groups 1
        problem.member_groups[i] = i + 1;      // Bars 1-12: groups 1-12
        problem.member_groups[i + 12] = i + 1; // Bars 13-24: same groups 1-12
    }

    // Groups A14 to A25: bars 25-48 in pairs (indices 24-47 in 0-based)
    for (int i = 0; i < 12; i++)
    {
        // A14: bars 25 and 37 -> groups 13
        problem.member_groups[i + 24] = i + 13; // Bars 25-36: groups 13-24
        problem.member_groups[i + 36] = i + 13; // Bars 37-48: same groups 13-24
    }

    // Optimization parameters
    problem.n_pop = 50;
    problem.gen = 1000;
    problem.CR = 0.5; // Initial value for SaDE
    problem.F = 0.5;  // Initial value for SaDE

    return problem;
}

// Create 72-bar truss problem with corrected loading and indexing
TrussProblem create72BarTruss()
{
    TrussProblem problem;

    problem.n_bars = 72;
    problem.n_nodes = 20;
    problem.n_dof = 60;
    problem.n_vars = 16;
    problem.s_adm = 25.0;

    // Available areas: 0.1 to 2.5 in steps of 0.1 (25 options)
    vector<double> areas;
    for (double a = 0.1; a <= 2.5; a += 0.1)
    {
        areas.push_back(a);
    }
    problem.areas = areas;

    // Node coordinates for 72-bar spatial truss
    problem.nodes = MatrixXd(20, 3);

    // First level (nodes 1-4) - bottom square
    double bottom_size = 120.0;
    problem.nodes << -bottom_size / 2, -bottom_size / 2, 0, // Node 1
        bottom_size / 2, -bottom_size / 2, 0,               // Node 2
        bottom_size / 2, bottom_size / 2, 0,                // Node 3
        -bottom_size / 2, bottom_size / 2, 0,               // Node 4

        // Second level (nodes 5-8)
        -bottom_size / 3, -bottom_size / 3, 120, // Node 5
        bottom_size / 3, -bottom_size / 3, 120,  // Node 6
        bottom_size / 3, bottom_size / 3, 120,   // Node 7
        -bottom_size / 3, bottom_size / 3, 120,  // Node 8

        // Third level (nodes 9-12)
        -bottom_size / 4, -bottom_size / 4, 240, // Node 9
        bottom_size / 4, -bottom_size / 4, 240,  // Node 10
        bottom_size / 4, bottom_size / 4, 240,   // Node 11
        -bottom_size / 4, bottom_size / 4, 240,  // Node 12

        // Fourth level (nodes 13-16)
        -bottom_size / 6, -bottom_size / 6, 360, // Node 13
        bottom_size / 6, -bottom_size / 6, 360,  // Node 14
        bottom_size / 6, bottom_size / 6, 360,   // Node 15
        -bottom_size / 6, bottom_size / 6, 360,  // Node 16

        // Top level (nodes 17-20)
        -bottom_size / 8, -bottom_size / 8, 480, // Node 17
        bottom_size / 8, -bottom_size / 8, 480,  // Node 18
        bottom_size / 8, bottom_size / 8, 480,   // Node 19
        -bottom_size / 8, bottom_size / 8, 480;  // Node 20

    // Bar connectivity - 72 bars in total
    problem.bars = MatrixXi(72, 2);

    // Group A1: bars 1-4 (vertical bars at bottom level)
    problem.bars << 1, 5, // Bar 1
        2, 6,             // Bar 2
        3, 7,             // Bar 3
        4, 8,             // Bar 4

        // Group A2: bars 5-12 (diagonals and horizontals between levels 1-2)
        1, 6, // Bar 5
        1, 2, // Bar 6
        2, 7, // Bar 7
        2, 3, // Bar 8
        3, 8, // Bar 9
        3, 4, // Bar 10
        4, 5, // Bar 11
        4, 1, // Bar 12

        // Group A3: bars 13-16 (vertical bars level 2-3)
        5, 9,  // Bar 13
        6, 10, // Bar 14
        7, 11, // Bar 15
        8, 12, // Bar 16

        // Group A4: bars 17-18 (diagonals level 2-3)
        5, 10, // Bar 17
        6, 11, // Bar 18

        // Group A5: bars 19-22 (vertical bars level 3-4)
        9, 13,  // Bar 19
        10, 14, // Bar 20
        11, 15, // Bar 21
        12, 16, // Bar 22

        // Group A6: bars 23-30 (diagonals and horizontals level 3-4)
        9, 14,  // Bar 23
        9, 10,  // Bar 24
        10, 15, // Bar 25
        10, 11, // Bar 26
        11, 16, // Bar 27
        11, 12, // Bar 28
        12, 13, // Bar 29
        12, 9,  // Bar 30

        // Group A7: bars 31-34 (vertical bars level 4-5)
        13, 17, // Bar 31
        14, 18, // Bar 32
        15, 19, // Bar 33
        16, 20, // Bar 34

        // Group A8: bars 35-36 (diagonals level 4-5)
        13, 18, // Bar 35
        14, 19, // Bar 36

        // Group A9: bars 37-40 (additional verticals)
        5, 13, // Bar 37
        6, 14, // Bar 38
        7, 15, // Bar 39
        8, 16, // Bar 40

        // Group A10: bars 41-48 (additional diagonals)
        1, 7,  // Bar 41
        2, 8,  // Bar 42
        3, 5,  // Bar 43
        4, 6,  // Bar 44
        5, 11, // Bar 45
        6, 12, // Bar 46
        7, 9,  // Bar 47
        8, 10, // Bar 48

        // Group A11: bars 49-52 (cross bracing)
        9, 15,  // Bar 49
        10, 16, // Bar 50
        11, 13, // Bar 51
        12, 14, // Bar 52

        // Group A12: bars 53-54 (top diagonals)
        17, 19, // Bar 53
        18, 20, // Bar 54

        // Group A13: bars 55-58 (final verticals)
        13, 19, // Bar 55
        14, 20, // Bar 56
        15, 17, // Bar 57
        16, 18, // Bar 58

        // Group A14: bars 59-66 (top level bracing)
        17, 18, // Bar 59
        18, 19, // Bar 60
        19, 20, // Bar 61
        20, 17, // Bar 62
        13, 20, // Bar 63
        14, 17, // Bar 64
        15, 18, // Bar 65
        16, 19, // Bar 66

        // Group A15: bars 67-70 (final cross members)
        9, 16,  // Bar 67
        10, 13, // Bar 68
        11, 14, // Bar 69
        12, 15, // Bar 70

        // Group A16: bars 71-72 (final diagonals)
        1, 8, // Bar 71
        2, 5; // Bar 72

    // Force vector - CORRECTED: Apply forces to top nodes (17-20) with proper indexing
    problem.f_vector = VectorXd::Zero(60);

    // Load case 1: Apply forces to top nodes (17-20)
    // Node 17: Fx = 5, Fy = 5, Fz = -5
    // Node 17 index: 16 (0-based), DOFs: 48, 49, 50
    problem.f_vector(48) = 5.0;  // Node 17, Fx
    problem.f_vector(49) = 5.0;  // Node 17, Fy
    problem.f_vector(50) = -5.0; // Node 17, Fz

    // Node 18: Fx = 0, Fy = 0, Fz = -5
    // Node 18 index: 17 (0-based), DOFs: 51, 52, 53
    problem.f_vector(53) = -5.0; // Node 18, Fz

    // Node 19: Fx = 0, Fy = 0, Fz = -5
    // Node 19 index: 18 (0-based), DOFs: 54, 55, 56
    problem.f_vector(56) = -5.0; // Node 19, Fz

    // Node 20: Fx = 0, Fy = 0, Fz = -5
    // Node 20 index: 19 (0-based), DOFs: 57, 58, 59
    problem.f_vector(59) = -5.0; // Node 20, Fz

    // Fixed DOFs - base nodes (1-4) are fixed
    problem.fixed_dof.clear();
    vector<int> fixed_nodes = {0, 1, 2, 3}; // Nodes 1-4 (0-based)
    for (int node : fixed_nodes)
    {
        problem.fixed_dof.push_back(node * 3);     // x-dof
        problem.fixed_dof.push_back(node * 3 + 1); // y-dof
        problem.fixed_dof.push_back(node * 3 + 2); // z-dof
    }

    // Member groups - 16 groups as specified
    problem.member_groups.resize(72);

    // Initialize all to -1 to catch any unassigned bars
    fill(problem.member_groups.begin(), problem.member_groups.end(), -1);

    // Group A1: bars 1-4 (0-based: 0-3)
    for (int i = 0; i < 4; i++)
        problem.member_groups[i] = 0;

    // Group A2: bars 5-12 (0-based: 4-11)
    for (int i = 4; i < 12; i++)
        problem.member_groups[i] = 1;

    // Group A3: bars 13-16 (0-based: 12-15)
    for (int i = 12; i < 16; i++)
        problem.member_groups[i] = 2;

    // Group A4: bars 17-18 (0-based: 16-17)
    for (int i = 16; i < 18; i++)
        problem.member_groups[i] = 3;

    // Group A5: bars 19-22 (0-based: 18-21)
    for (int i = 18; i < 22; i++)
        problem.member_groups[i] = 4;

    // Group A6: bars 23-30 (0-based: 22-29)
    for (int i = 22; i < 30; i++)
        problem.member_groups[i] = 5;

    // Group A7: bars 31-34 (0-based: 30-33)
    for (int i = 30; i < 34; i++)
        problem.member_groups[i] = 6;

    // Group A8: bars 35-36 (0-based: 34-35)
    for (int i = 34; i < 36; i++)
        problem.member_groups[i] = 7;

    // Group A9: bars 37-40 (0-based: 36-39)
    for (int i = 36; i < 40; i++)
        problem.member_groups[i] = 8;

    // Group A10: bars 41-48 (0-based: 40-47)
    for (int i = 40; i < 48; i++)
        problem.member_groups[i] = 9;

    // Group A11: bars 49-52 (0-based: 48-51)
    for (int i = 48; i < 52; i++)
        problem.member_groups[i] = 10;

    // Group A12: bars 53-54 (0-based: 52-53)
    for (int i = 52; i < 54; i++)
        problem.member_groups[i] = 11;

    // Group A13: bars 55-58 (0-based: 54-57)
    for (int i = 54; i < 58; i++)
        problem.member_groups[i] = 12;

    // Group A14: bars 59-66 (0-based: 58-65)
    for (int i = 58; i < 66; i++)
        problem.member_groups[i] = 13;

    // Group A15: bars 67-70 (0-based: 66-69)
    for (int i = 66; i < 70; i++)
        problem.member_groups[i] = 14;

    // Group A16: bars 71-72 (0-based: 70-71)
    for (int i = 70; i < 72; i++)
        problem.member_groups[i] = 15;

    // Verify all bars are assigned
    for (int i = 0; i < 72; i++)
    {
        if (problem.member_groups[i] == -1)
        {
            cerr << "Error: Bar " << i + 1 << " not assigned to any group!" << endl;
        }
    }

    // Optimization parameters - KEEPING ORIGINAL VALUES AS REQUESTED
    problem.n_pop = 50;
    problem.gen = 1000;
    problem.CR = 0.5; // Initial value for SaDE
    problem.F = 0.5;  // Initial value for SaDE

    return problem;
}

// Parallel execution function for each truss problem
void runTrussOptimization(const string &problem_name,
                          function<TrussProblem()> create_problem_func,
                          long seed,
                          mutex &cout_mutex)
{

    auto start_time = chrono::high_resolution_clock::now();

    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "Starting " << problem_name << " optimization with SaDE on thread "
             << this_thread::get_id() << endl;
    }

    try
    {
        // Create problem and optimizer
        TrussProblem problem = create_problem_func();
        GDE3_SaDE_APM optimizer(problem, seed);

        // Run optimization
        optimizer.optimize();

        // Export results
        string filename = "pareto_front_" + problem_name + "_SaDE.csv";
        optimizer.exportResults(filename);

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);

        lock_guard<mutex> lock(cout_mutex);
        cout << problem_name << " SaDE optimization completed in "
             << duration.count() << " seconds" << endl;
        cout << "Results saved to: " << filename << endl;
        cout << "Final archive size: " << optimizer.getParetoFront().size() << " solutions" << endl;
        cout << "----------------------------------------" << endl;
    }
    catch (const exception &e)
    {
        lock_guard<mutex> lock(cout_mutex);
        cerr << "Error in " << problem_name << " optimization: " << e.what() << endl;
    }
}

int main()
{
    cout << "=== GDE3+APM+SaDE Truss Optimization - Parallel Execution ===" << endl;

    auto total_start_time = chrono::high_resolution_clock::now();

    // Define seeds for reproducibility
    long seed = 20;

    // Mutex for thread-safe console output
    mutex cout_mutex;

    // Create threads for each truss problem
    vector<thread> threads;

    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "Launching parallel optimization threads with SaDE..." << endl;
    }

    // Start 10-bar truss optimization thread
    threads.emplace_back(runTrussOptimization, "10bar", create10BarTruss, seed, ref(cout_mutex));

    // Start 25-bar truss optimization thread
    threads.emplace_back(runTrussOptimization, "25bar", create25BarTruss, seed, ref(cout_mutex));

    // Start 60-bar truss optimization thread
    threads.emplace_back(runTrussOptimization, "60bar", create60BarTruss, seed, ref(cout_mutex));

    // Start 72-bar truss optimization thread
    threads.emplace_back(runTrussOptimization, "72bar", create72BarTruss, seed, ref(cout_mutex));

    // Wait for all threads to complete
    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "Waiting for all SaDE optimizations to complete..." << endl;
    }

    for (auto &t : threads)
    {
        t.join();
    }

    auto total_end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::seconds>(total_end_time - total_start_time);

    cout << "\n=== All SaDE Optimizations Complete ===" << endl;
    cout << "Total execution time: " << total_duration.count() << " seconds" << endl;
    cout << "Results saved to:" << endl;
    cout << " - pareto_front_10bar_SaDE.csv" << endl;
    cout << " - pareto_front_25bar_SaDE.csv" << endl;
    cout << " - pareto_front_60bar_SaDE.csv" << endl;
    cout << " - pareto_front_72bar_SaDE.csv" << endl;

    return 0;
}
