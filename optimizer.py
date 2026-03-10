from typing import Tuple, List
import numpy as np
import casadi as ca
import yaml
from collections import OrderedDict

from src.constraints import ConstraintType
from .hybrid_systems.hybrid_systems import HybridSystem
from .robot import Robot
from .trajectory_manager import TrajectoryManager


# Custom YAML representer to format arrays as single lines
class SingleLineArrayDumper(yaml.Dumper):
    def represent_list(self, data):
        # For bounds arrays, format as single line
        if len(data) > 0 and isinstance(data[0], (int, float)):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    
    def represent_dict(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)
    
    def represent_ordereddict(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)


SingleLineArrayDumper.add_representer(list, SingleLineArrayDumper.represent_list)
SingleLineArrayDumper.add_representer(dict, SingleLineArrayDumper.represent_dict)
SingleLineArrayDumper.add_representer(OrderedDict, SingleLineArrayDumper.represent_ordereddict)


class TrajectoryOptimizer:
    """Class for solving trajectory optimization problems."""

    def __init__(self, robot: Robot):
        """Create the optimizer object.
        T (float): Total time (seconds).
        robot (Robot): The robot object to optimize.
        N (int): Number of shooting nodes.
        """
        self.robot = robot

        # Create optimizer
        self.opti = ca.Opti()

        self.variables: dict[str, Tuple] = {}

    def setup_problem(self, hybrid_system):
        """Setup the optimization problem."""
        nstates = self.robot.nq + self.robot.nv

        domain_node_ranges = hybrid_system.get_domain_node_ranges()

        cost = 0

        # Iterate through the domains
        for name, domain in hybrid_system.domains.items():
            print(f"Setting up optimization for {name}...")
            start, stop = domain_node_ranges[name]
            N = stop - start
            force_dof = domain.force_dof

            # Decision variables
            self.variables[name] = (
                self.opti.variable(nstates, N),
                self.opti.variable(self.robot.nj_act + force_dof*len(domain.contact_frames), N),
                self.opti.variable(domain.aux_index)
            )

            X = self.variables[name][0]
            U = self.variables[name][1]
            S = self.variables[name][2]
            
            # Debug output
            print(f"Domain '{name}' variables:")
            print(f"  X shape: {X.shape} (states x nodes)")
            print(f"  U shape: {U.shape} (inputs x nodes)")
            print(f"  S shape: {S.shape if hasattr(S, 'shape') else 'scalar'} (aux vars)")
            print(f"  aux_index: {domain.aux_index}")
            print(f"  contact_frames: {domain.contact_frames}")
            print(f"  force_dof: {force_dof}")
            print(f"  Expected U size: nj_act({self.robot.nj_act}) + force_dof({force_dof})*contacts({len(domain.contact_frames)}) = {self.robot.nj_act + force_dof*len(domain.contact_frames)}")

            # Set the initial guess for the states
            print(f"  Setting initial guess for {name}:")
            if domain.x_guess is not None:
                print(f"    x_guess shape: {domain.x_guess.shape}")
                if domain.x_guess.ndim > 1:
                    print(f"    Using 2D x_guess directly")
                    self.opti.set_initial(X, domain.x_guess)
                elif domain.x_guess.ndim == 1:
                    print(f"    Broadcasting 1D x_guess to all {N} nodes")
                    for i in range(N):
                        self.opti.set_initial(X[:, i], domain.x_guess)
                else:
                    raise ValueError("Invalid x_guess shape!")
            else:
                print(f"    No x_guess provided for {name}")

            # Set initial values for auxiliary variables
            print(f"    aux_index: {domain.aux_index}")
            if domain.aux_index > 0:
                S_init = np.zeros(domain.aux_index)
                
                # Set initial value for time variable T
                if "T" in domain.index_key:
                    T_idx = domain.index_key["T"]
                    S_init[T_idx[0]:T_idx[1]] = 0.4  # Default time of 0.4 seconds
                
                # Set initial values for Bezier coefficients
                if "bezier_coeffs" in domain.index_key:
                    bezier_idx = domain.index_key["bezier_coeffs"]
                    n_bezier = bezier_idx[1] - bezier_idx[0]
                    
                    # Get initial joint positions from x_guess for Bezier coefficients
                    if domain.x_guess is not None:
                        # Extract joint positions from initial guess
                        base_size = self.robot.FLOATING_POS if not self.robot.planar else self.robot.PLANAR_FLOATING_POS
                        q_init = domain.x_guess[base_size:base_size + self.robot.nj_act]
                        
                        # Set all Bezier coefficients to the initial joint positions
                        # This creates a constant trajectory as initial guess
                        bezier_degree = 5  # Default degree
                        n_constraints = n_bezier // (bezier_degree + 1)
                        
                        for i in range(n_constraints):
                            if i < len(q_init):
                                joint_val = q_init[i]
                            else:
                                joint_val = 0.0
                            
                            # Set all coefficients for this joint to the same value
                            start_coeff = i * (bezier_degree + 1)
                            end_coeff = (i + 1) * (bezier_degree + 1)
                            S_init[bezier_idx[0] + start_coeff:bezier_idx[0] + end_coeff] = joint_val
                
                self.opti.set_initial(S, S_init)

            # Print constraint/cost summary for this domain
            total_constraints = len(domain.constraints)
            total_costs = len(domain.costs)
            print(f"  Domain summary: {total_constraints} constraints, {total_costs} costs")
            
            # Iterate through the nodes for this domain
            for i in range(0, N): #range(*domain_node_ranges[name] - start):
                # --------------------------------- #
                # ---------- Constraints ---------- #
                # --------------------------------- #
                # Get the constraints at this node:
                constraints = domain.get_constraints_for_node(i)
   
                # Iterate through the constraints for this node
                for constraint in constraints:
                    if constraint.get_constraint_type() == ConstraintType.DYNAMICS:
                        if i < N - 1:  # Use local domain size, not global ranges
                            self.opti.subject_to(
                                constraint.evaluate(X[:, i], U[:, i], X[:, i + 1], S) == np.zeros(nstates)
                            )
                    elif constraint.get_constraint_type() == ConstraintType.ACCELERATIONS:
                        if i < N - 1:  # Use local domain size, not global ranges
                            self.opti.subject_to(
                                constraint.evaluate(X[:, i], U[:, i], X[:, i + 1], S) == np.zeros(self.robot.nj_act)
                            )
                    # TODO: Start by specifying the global position bc i know the original position
                    # TODO: Add the ability to specify cross-node constraints so I can constrain
                    #   the end of the swing phase position relative to the other foot position
                    #   in the first node
                    elif constraint.get_constraint_type() == ConstraintType.STATE:
                        if constraint.is_equality:
                            lb, ub = constraint.get_bounds()
                            self.opti.subject_to(self.opti.bounded(lb, constraint.evaluate(X[:, i], S), ub))
                        else:
                            lb, ub = constraint.get_bounds()
                            self.opti.subject_to(
                                self.opti.bounded(lb, constraint.evaluate(X[:, i], S), ub)
                            )
                    elif constraint.get_constraint_type() == ConstraintType.INPUT:
                        if constraint.is_equality:
                            self.opti.subject_to(
                                constraint.evaluate(U[:, i]) == constraint.get_bounds()[0]
                            )
                        else:
                            lb, ub = constraint.get_bounds()
                            self.opti.subject_to(
                                self.opti.bounded(lb, constraint.evaluate(U[:, i]), ub)
                            )
                    elif constraint.get_constraint_type() == ConstraintType.AUXILIARY:
                        if hasattr(constraint, 'requires_prev_frame') and constraint.requires_prev_frame():
                            if constraint.is_equality:
                                lb, ub = constraint.get_bounds()
                                self.opti.subject_to(
                                    self.opti.bounded(lb, constraint.evaluate(X[:, i], U[:, i], S, i, X[:, 0]), ub))
                            else:
                                lb, ub = constraint.get_bounds()
                                self.opti.subject_to(
                                    self.opti.bounded(lb, constraint.evaluate(X[:, i], U[:, i], S, i, X[:, 0]), ub)
                                )
                        else:
                            #check
                            # if i == domain_node_ranges[name][0]:
                                # Only set the auxiliary constraints once in a domain
                            if constraint.is_equality:
                                lb, ub = constraint.get_bounds()
                                self.opti.subject_to(self.opti.bounded(lb, constraint.evaluate(X[:,i],U[:, i],S,i), ub))
                            else:
                                lb, ub = constraint.get_bounds()
                                self.opti.subject_to(
                                    self.opti.bounded(lb, constraint.evaluate(X[:,i],U[:, i],S,i), ub)
                                )
                    else:
                        raise NotImplementedError

                # --------------------------------- #
                # ------------- Cost -------------- #
                # --------------------------------- #
                # Add costs for each node in this domain
                costs = domain.get_costs_for_node(i)
                for cost_term in costs:
                    cost += cost_term.evaluate(X[:, i], U[:, i], S)

        # Iterate through the edges
        for name, edge in hybrid_system.edges.items():
            print(f"Setting up edge {name}...")
            # --------------------------------- #
            # ---------- Constraints ---------- #
            # --------------------------------- #

            prev_domain_name = edge.from_domain.name
            next_domain_name = edge.to_domain.name

            Xk = self.variables[prev_domain_name][0][:, -1]
            Uk = self.variables[prev_domain_name][1][:, -1]
            Sk = self.variables[prev_domain_name][2]  # S is a vector, not a matrix
            Xkp1 = self.variables[next_domain_name][0][:, 0]
            
            # Debug output for edge variables
            print(f"Edge '{name}' ({prev_domain_name} -> {next_domain_name}):")
            print(f"  Xk shape: {Xk.shape} (from {prev_domain_name} last node)")
            print(f"  Uk shape: {Uk.shape} (from {prev_domain_name} last node)")  
            print(f"  Sk shape: {Sk.shape if hasattr(Sk, 'shape') else 'scalar'} (from {prev_domain_name})")
            print(f"  Xkp1 shape: {Xkp1.shape} (to {next_domain_name} first node)")
            print(f"  Edge constraints: {len(edge.constraints)}")

            if len(edge.constraints) == 0:
                raise ValueError("No edge constraint found!")

            # Iterate through the constraints for this node
            for constraint in edge.constraints:
                if constraint.get_constraint_type() == ConstraintType.DYNAMICS:
                    print(f"Setting up edge constraint {constraint.name}...")
                    lb, ub = constraint.get_bounds()
                    print(f"  Constraint bounds shape: {len(lb)} (lb), {len(ub)} (ub)")
                    constraint_value = constraint.evaluate(Xk, Uk, Xkp1, Sk)
                    print(f"  Constraint output dimensions: {constraint_value.shape if hasattr(constraint_value, 'shape') else 'scalar'}")
                    self.opti.subject_to(self.opti.bounded(lb, constraint_value, ub))
                else:
                    raise ValueError("Edge nodes only support dynamics constraints!")

        self.opti.minimize(cost)

    def solve(self, hybrid_system):
        """Solve the optimization problem."""
        self.setup_problem(hybrid_system)
            
        # Set solver options for better numerical stability and debugging
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 3100,
            "constr_viol_tol": 1e-2,
            # Add options to help with the restoration phase issue
            "acceptable_tol": 1e-6,
            "acceptable_constr_viol_tol": 1e-6,
            # "mu_strategy": "adaptive",
            # "nlp_scaling_method": "gradient-based",
            # Enable more verbose output for debugging
            # "print_level": 5,
            # "print_frequency_iter": 1,
            # "print_timing_statistics": "yes"
        }
        self.opti.solver('ipopt', p_opts, s_opts)

        try:
            # Solve the problem
            sol = self.opti.solve()
            return self.parse_solution(hybrid_system, sol), sol.value(self.opti.f)
        except Exception as e:
            print(f"\nOptimization failed with error: {e}")
            
            # Analyze constraint violations to help debug
            print("\n" + "="*60)
            print("CONSTRAINT VIOLATION ANALYSIS")
            print("="*60)
            self._analyze_constraint_violations(hybrid_system)
            
            # Try to return the most recent intermediate solution using opti.debug
            print("\nAttempting to extract intermediate solution using opti.debug...")
            try:
                # Create a mock solution object with current values
                class MockSolution:
                    def __init__(self, opti_instance):
                        self.opti = opti_instance
                    
                    def value(self, var):
                        return self.opti.debug.value(var)
                
                mock_sol = MockSolution(self.opti)
                intermediate_parsed = self.parse_solution(hybrid_system, mock_sol)
                
                # Try to get objective value, use inf if not available
                try:
                    cost = self.opti.debug.value(self.opti.f)
                except:
                    cost = float('inf')
                
                print(f"Successfully extracted intermediate solution with cost: {cost}")
                return intermediate_parsed, cost
                
            except Exception as debug_e:
                print(f"Could not extract intermediate solution: {debug_e}")
                raise e  # Re-raise the original exception

    def solve_with_warm_start_file(self, hybrid_system, config, solution_file):
        """Solve the optimization problem with warm start file."""
        self.setup_problem(hybrid_system)

        # Set the warm start from the file
        self._set_warm_start_from_file(hybrid_system, config, solution_file)

        # Set solver options for better numerical stability
        p_opts = {"expand": True}  # TODO: What does this do?
        s_opts = {"max_iter": 3100,
                  "acceptable_tol": 1e-3,
                  "acceptable_constr_viol_tol": 1e-2,
                  "constr_viol_tol": 1e-2}
        self.opti.solver('ipopt', p_opts, s_opts)

        # Solve the problem
        sol = self.opti.solve()

        return self.parse_solution(hybrid_system, sol), sol.value(self.opti.f)

    def solve_with_warm_start_file_fine(self, hybrid_system_list, config, solution_file):
        """Solve the optimization problem with warm start file."""
        coarse_parsed = self.solve_with_warm_start_file(hybrid_system_list[0], config, solution_file)

        self._apply_simple_warm_start(hybrid_system_list[1], coarse_parsed, hybrid_system_list[0])
        print("Warm start applied from coarse solution")

        self.opti = ca.Opti()       # Reset the opti stack

        # Set normal iterations for fine problem
        p_opts = {"expand": True}
        s_opts = {"max_iter": 3000, "constr_viol_tol": 1e-2}
        self.opti.solver('ipopt', p_opts, s_opts)

        self.setup_problem(hybrid_system_list[1])

        fine_sol = self.opti.solve()
        fine_parsed = self.parse_solution(hybrid_system_list[1], fine_sol)
        fine_cost = fine_sol.value(self.opti.f)

        print(f"Fine problem solved! Cost: {fine_cost:.6f}")

        return fine_parsed, fine_cost




    def solve_with_warm_start(self, hybrid_system_list: list[HybridSystem]):
        """Solve the optimization problem using warm-starting with coarse-to-fine refinement.
        
        Args:
            hybrid_system_list: The list of hybrid systems to optimize
            
        Returns:
            tuple: (parsed_solution, final_cost)
        """
        print(f"Starting warm-start optimization...")
        if len(hybrid_system_list) != 2:
            raise ValueError("Hybrid systems list must be of length 2 for warm-starting!")
        # Step 1: Solve coarse problem
        print(f"\n=== Step 1: Solving coarse problem ===")

        # Set fewer iterations for coarse problem
        p_opts = {"expand": True}
        s_opts = {"max_iter": 3100, "constr_viol_tol": 1e-2}
        self.opti.solver('ipopt', p_opts, s_opts)

        self.setup_problem(hybrid_system_list[0])
        coarse_sol = self.opti.solve()
        coarse_parsed = self.parse_solution(hybrid_system_list[0], coarse_sol)
        coarse_cost = coarse_sol.value(self.opti.f)

        print(f"Coarse problem solved! Cost: {coarse_cost:.6f}")
        
        # Step 2: Solve fine problem with warm start
        print(f"\n=== Step 2: Solving fine problem  ===")
        self.opti = ca.Opti()       # Reset the opti stack

        # Set normal iterations for fine problem
        p_opts = {"expand": True}
        s_opts = {"max_iter": 3000, "constr_viol_tol": 1e-2}
        self.opti.solver('ipopt', p_opts, s_opts)

        self.setup_problem(hybrid_system_list[1])

        # Apply warm start if we have a coarse solution
        if coarse_parsed is not None:
            self._apply_simple_warm_start(hybrid_system_list[1], coarse_parsed, hybrid_system_list[0])
            print("Warm start applied from coarse solution")

        fine_sol = self.opti.solve()
        fine_parsed = self.parse_solution(hybrid_system_list[1], fine_sol)
        fine_cost = fine_sol.value(self.opti.f)

        print(f"Fine problem solved! Cost: {fine_cost:.6f}")

        return fine_parsed, fine_cost

    def _create_simple_coarse_system(self, hybrid_system, coarse_nodes):
        """Create a simple coarse version by just changing the node count."""
        # Create a new hybrid system
        from src.hybrid_systems.hybrid_systems import HybridSystem
        
        coarse_system = HybridSystem()
        
        # Copy domains with updated node counts
        for domain_name, domain in hybrid_system.domains.items():
            # Create a simple copy with just the node count changed
            coarse_domain = self._create_simple_domain_copy(domain, coarse_nodes)
            coarse_system.add_domain(coarse_domain)
            print(f"Domain {domain_name}: {coarse_nodes} nodes")
        
        # Copy edges
        for edge_name, edge in hybrid_system.edges.items():
            coarse_system.add_edge(edge)
        
        # Copy domain sequence
        if hasattr(hybrid_system, 'domain_sequence'):
            coarse_system.set_domain_sequence(hybrid_system.domain_sequence)
        
        return coarse_system

    def _create_simple_fine_system(self, hybrid_system, fine_nodes):
        """Create a simple fine version by just changing the node count."""
        # Create a new hybrid system
        from src.hybrid_systems.hybrid_systems import HybridSystem
        
        fine_system = HybridSystem()
        
        # Copy domains with updated node counts
        for domain_name, domain in hybrid_system.domains.items():
            # Create a simple copy with just the node count changed
            fine_domain = self._create_simple_domain_copy(domain, fine_nodes)
            fine_system.add_domain(fine_domain)
            print(f"Domain {domain_name}: {fine_nodes} nodes")
        
        # Copy edges
        for edge_name, edge in hybrid_system.edges.items():
            fine_system.add_edge(edge)
        
        # Copy domain sequence
        if hasattr(hybrid_system, 'domain_sequence'):
            fine_system.set_domain_sequence(hybrid_system.domain_sequence)
        
        return fine_system

    def _create_simple_domain_copy(self, original_domain, new_num_nodes):
        """Create a simple copy of a domain with just the node count changed."""
        from src.hybrid_systems.hybrid_systems import Domain
        
        # Create new domain with same configuration but different node count
        new_domain = Domain(
            name=original_domain.name,
            num_nodes=new_num_nodes,
            robot=original_domain.robot,
            contact_frames=original_domain.contact_frames,
            x_guess=original_domain.x_guess,
            force_dof=original_domain.force_dof
        )
        
        # Copy index key and auxiliary variables
        new_domain.index_key = original_domain.index_key.copy()
        new_domain.aux_index = original_domain.aux_index
        
        # Copy configuration
        new_domain.config_reference = original_domain.config_reference.copy()
        
        # Copy constraints with simple node mapping
        for constraint in original_domain.constraints:
            # Find which nodes this constraint was active on in the original domain
            original_active_nodes = []
            for node_idx, constraints_list in original_domain.active_node_constraints.items():
                if constraint in constraints_list:
                    original_active_nodes.append(node_idx)
            
            # Simple mapping: scale nodes proportionally
            adjusted_active_nodes = self._simple_node_mapping(
                original_active_nodes, original_domain.num_nodes, new_num_nodes
            )
            new_domain.add_constraint(constraint, adjusted_active_nodes)
        
        # Copy costs with simple node mapping
        for cost in original_domain.costs:
            # Find which nodes this cost was active on in the original domain
            original_active_nodes = []
            for node_idx, costs_list in original_domain.active_node_costs.items():
                if cost in costs_list:
                    original_active_nodes.append(node_idx)
            
            # Simple mapping: scale nodes proportionally
            adjusted_active_nodes = self._simple_node_mapping(
                original_active_nodes, original_domain.num_nodes, new_num_nodes
            )
            new_domain.add_cost(cost, adjusted_active_nodes)
        
        return new_domain

    def _simple_node_mapping(self, original_active_nodes, original_count, new_count):
        """Simple proportional mapping of node indices."""
        if not original_active_nodes:
            return []
        
        # Simple proportional scaling
        adjusted_nodes = []
        for node_idx in original_active_nodes:
            # Scale the index proportionally
            scaled_idx = int(node_idx * (new_count - 1) / (original_count - 1))
            # Ensure it's within bounds
            scaled_idx = max(0, min(scaled_idx, new_count - 1))
            adjusted_nodes.append(scaled_idx)
        
        return adjusted_nodes

    def _apply_simple_warm_start(self, fine_system, coarse_solution, coarse_system):
        """Apply simple warm start by duplicating coarse solution states."""
        for domain_name in fine_system.domains.keys():
            if domain_name in coarse_solution:
                # Get coarse and fine data
                coarse_x = coarse_solution[domain_name]['x']
                coarse_u = coarse_solution[domain_name]['u']
                coarse_S = coarse_solution[domain_name]['S']
                
                fine_domain = fine_system.domains[domain_name]
                coarse_domain = coarse_system.domains[domain_name]
                
                # Create fine trajectory by duplicating coarse states
                fine_x = np.zeros((coarse_x.shape[0], fine_domain.num_nodes))
                fine_u = np.zeros((coarse_u.shape[0], fine_domain.num_nodes))
                
                # Duplicate states: each coarse node gets duplicated to fill fine nodes
                for i in range(fine_domain.num_nodes):
                    # Map fine node to coarse node
                    coarse_idx = int(i * (coarse_domain.num_nodes - 1) / (fine_domain.num_nodes - 1))
                    coarse_idx = max(0, min(coarse_idx, coarse_domain.num_nodes - 1))
                    
                    # Use the coarse state/control for this fine node
                    fine_x[:, i] = coarse_x[:, coarse_idx]
                    fine_u[:, i] = coarse_u[:, coarse_idx]

                # Set initial values
                self.opti.set_initial(self.variables[domain_name][0], fine_x)
                self.opti.set_initial(self.variables[domain_name][1], fine_u)
                
                # For auxiliary variables (like time), use the same values
                if isinstance(coarse_S, float):
                    self.opti.set_initial(self.variables[domain_name][2], coarse_S)
                elif len(coarse_S) > 0:
                    self.opti.set_initial(self.variables[domain_name][2], coarse_S)

                print(f"Applied simple warm start to domain {domain_name}")
                print(f"  Coarse: {coarse_domain.num_nodes} nodes")
                print(f"  Fine: {fine_domain.num_nodes} nodes")
                print(f"  Duplicated states for initial guess")

    def _set_warm_start_from_file(self, hybrid_system, config, filename):
        """Set the warm start from the file."""
        traj_manager = TrajectoryManager(self.robot, config)
        warm_start = traj_manager.load_trajectory(filename)

        for domain_name in hybrid_system.domains.keys():
            domain = hybrid_system.domains[domain_name]
            domain_ws = warm_start[domain_name]

            # Create fine trajectory by duplicating coarse states
            x_ws = np.zeros((domain_ws['x'].shape[0], domain.num_nodes))
            u_ws = np.zeros((domain_ws['u'].shape[0], domain.num_nodes))

            # Duplicate states: each coarse node gets duplicated to fill fine nodes
            for i in range(domain.num_nodes):
                # Use the coarse state/control for this fine node
                x_ws[:, i] = domain_ws['x'][:, i]
                u_ws[:, i] = domain_ws['u'][:, i]

            # Set initial values
            self.opti.set_initial(self.variables[domain_name][0], x_ws)
            self.opti.set_initial(self.variables[domain_name][1], u_ws)

            # For auxiliary variables (like time), use the same values
            self.opti.set_initial(self.variables[domain_name][2], domain_ws['S'])



    def solve_adaptive(self, hybrid_system, min_nodes=5, max_nodes=50, 
                      target_cost_improvement=0.1, max_iterations=3):
        """Solve with adaptive node refinement based on cost improvement.
        
        Args:
            hybrid_system: The hybrid system to optimize
            min_nodes: Minimum number of nodes to start with
            max_nodes: Maximum number of nodes to use
            target_cost_improvement: Minimum cost improvement to continue refinement
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            tuple: (parsed_solution, final_cost)
        """
        print(f"Starting adaptive optimization...")
        print(f"Node range: {min_nodes} to {max_nodes}")
        
        current_nodes = min_nodes
        previous_cost = float('inf')
        
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1}: {current_nodes} nodes ===")
            
            # Create system with current number of nodes
            current_system = self._create_coarse_system(hybrid_system, current_nodes)
            
            # Solve with warm start if not first iteration
            if iteration == 0:
                sol, cost = self.solve(current_system)
            else:
                sol, cost = self.solve_with_warm_start(current_system, 
                                                     coarse_nodes=None, 
                                                     fine_nodes=current_nodes)
            
            print(f"Cost: {cost:.6f}")
            
            # Check if we should continue refining
            if iteration > 0:
                cost_improvement = (previous_cost - cost) / previous_cost
                print(f"Cost improvement: {cost_improvement:.2%}")
                
                if cost_improvement < target_cost_improvement:
                    print("Cost improvement below threshold, stopping refinement")
                    break
            
            # Prepare for next iteration
            previous_cost = cost
            current_nodes = min(current_nodes * 2, max_nodes)
            
            if current_nodes >= max_nodes:
                print("Reached maximum nodes, stopping refinement")
                break
        
        return sol, cost

    def parse_solution(self, hybrid_system, sol):
        parsed_sol : dict[dict[str, np.ndarray]] = {}
        # Iterate through the domains
        # print(f"debug: {self.opti.debug.value()}")
        for name, domain in hybrid_system.domains.items():
            # parsed_sol[name] = {'x': self.opti.debug.value(self.variables[name][0]),
            #                     'u': self.opti.debug.value(self.variables[name][1]),
            #                     'S': self.opti.debug.value(self.variables[name][2])}
            parsed_sol[name] = {'x': sol.value(self.variables[name][0]),
                                'u': sol.value(self.variables[name][1]),
                                'S': sol.value(self.variables[name][2])}

        return parsed_sol

    def export_nlp(self, hybrid_system, filename: str):
        """Export the NLP constraints and bounds to a YAML file.
        
        Args:
            hybrid_system: The hybrid system containing domains and edges
            filename: Output YAML filename
        """
        nlp_data = {
            'problem_info': {
                'robot_name': (self.robot.model.name if hasattr(self.robot.model, 'name') 
                              else 'unknown'),
                'nq': self.robot.nq,
                'nv': self.robot.nv,
                'nj_act': self.robot.nj_act,
                'planar': self.robot.planar
            },
            'domains': {},
            'edges': {},
            'constraints_summary': {
                'total_constraints': 0,
                'equality_constraints': 0,
                'inequality_constraints': 0
            }
        }
        
        total_constraints = 0
        equality_constraints = 0
        inequality_constraints = 0
        
        # Export domain information
        for domain_name, domain in hybrid_system.domains.items():
            domain_data = {
                'name': domain_name,
                'num_nodes': domain.num_nodes,
                'contact_frames': domain.contact_frames,
                'force_dof': domain.force_dof,
                'constraints': [],
                'costs': []
            }
            
            # Collect all constraints for this domain
            for constraint in domain.constraints:
                lb, ub = constraint.get_bounds()
                lb_list = lb.tolist() if hasattr(lb, 'tolist') else lb
                ub_list = ub.tolist() if hasattr(ub, 'tolist') else ub
                
                constraint_info = OrderedDict()
                constraint_info['name'] = constraint.name
                constraint_info['type'] = constraint.get_constraint_type().value
                constraint_info['is_equality'] = constraint.is_equality
                constraint_info['active_nodes'] = self._get_constraint_active_nodes(domain, constraint)
                constraint_info['bounds'] = OrderedDict([
                    ('lower', lb_list),
                    ('upper', ub_list)
                ])
                domain_data['constraints'].append(constraint_info)
                
                # Count constraints
                if constraint.is_equality:
                    equality_constraints += 1
                else:
                    inequality_constraints += 1
                total_constraints += 1
            
            # Collect costs
            for cost in domain.costs:
                cost_info = OrderedDict()
                cost_info['name'] = cost.name
                cost_info['weight'] = cost.weight
                cost_info['active_nodes'] = self._get_cost_active_nodes(domain, cost)
                domain_data['costs'].append(cost_info)
            
            nlp_data['domains'][domain_name] = domain_data
        
        # Export edge information
        for edge_name, edge in hybrid_system.edges.items():
            edge_data = {
                'name': edge_name,
                'from_domain': edge.from_domain.name,
                'to_domain': edge.to_domain.name,
                'constraints': []
            }
            
            for constraint in edge.constraints:
                lb, ub = constraint.get_bounds()
                lb_list = lb.tolist() if hasattr(lb, 'tolist') else lb
                ub_list = ub.tolist() if hasattr(ub, 'tolist') else ub
                
                constraint_info = OrderedDict()
                constraint_info['name'] = constraint.name
                constraint_info['type'] = constraint.get_constraint_type().value
                constraint_info['is_equality'] = constraint.is_equality
                constraint_info['bounds'] = OrderedDict([
                    ('lower', lb_list),
                    ('upper', ub_list)
                ])
                edge_data['constraints'].append(constraint_info)
                
                # Count constraints
                if constraint.is_equality:
                    equality_constraints += 1
                else:
                    inequality_constraints += 1
                total_constraints += 1
            
            nlp_data['edges'][edge_name] = edge_data
        
        # Update summary
        nlp_data['constraints_summary'] = {
            'total_constraints': total_constraints,
            'equality_constraints': equality_constraints,
            'inequality_constraints': inequality_constraints
        }
        
        # Write to YAML file with flow style for arrays
        with open(filename, 'w') as f:
            yaml.dump(nlp_data, f, Dumper=SingleLineArrayDumper, indent=2, sort_keys=False)
        
        print(f"NLP exported to {filename}")
        print(f"Total constraints: {total_constraints}")
        print(f"Equality constraints: {equality_constraints}")
        print(f"Inequality constraints: {inequality_constraints}")
    
    def _analyze_constraint_violations(self, hybrid_system):
        """Analyze constraint violations to help debug optimization failures."""
        violations = []
        
        # Get current variable values from debug
        for domain_idx, domain_name in enumerate(hybrid_system.domains):
            domain = hybrid_system.domains[domain_name]

            if domain_name not in self.variables:
                raise RuntimeError(f"Domain {domain_name} not in optimizer variables!")
                
            X, U, S = self.variables[domain_name]
            X_val = self.opti.debug.value(X)
            U_val = self.opti.debug.value(U)
            S_val = self.opti.debug.value(S)

            if isinstance(S_val, float):
                S_val = [S_val]
            
            # Analyze domain constraints
            for node_idx in range(X_val.shape[1]):
                constraints = domain.get_constraints_for_node(node_idx)
                for constraint in constraints:
                    # Evaluate constraint at current values
                    if constraint.get_constraint_type() == ConstraintType.DYNAMICS and node_idx < X_val.shape[1] - 1:
                        # Create CasADi symbolic variables for function evaluation
                        x_sym = ca.MX.sym('x', X_val.shape[0])
                        u_sym = ca.MX.sym('u', U_val.shape[0]) 
                        xnext_sym = ca.MX.sym('xnext', X_val.shape[0])
                        s_sym = ca.MX.sym('s', len(S_val))
                        
                        # Create constraint function
                        constraint_expr = constraint.evaluate(x_sym, u_sym, xnext_sym, s_sym)
                        constraint_func = ca.Function('constraint', [x_sym, u_sym, xnext_sym, s_sym], [constraint_expr])
                        
                        # Evaluate with numpy values
                        constraint_val = constraint_func(X_val[:, node_idx], U_val[:, node_idx], 
                                                       X_val[:, node_idx + 1], S_val).full().flatten()
                        expected_val = np.zeros(constraint_val.shape)
                        violation = np.linalg.norm(constraint_val - expected_val)
                        
                    elif constraint.get_constraint_type() == ConstraintType.STATE:
                        # Create CasADi symbolic variables for function evaluation
                        x_sym = ca.MX.sym('x', X_val.shape[0])
                        s_sym = ca.MX.sym('s', len(S_val))
                        
                        # Create constraint function
                        constraint_expr = constraint.evaluate(x_sym, s_sym)
                        constraint_func = ca.Function('constraint', [x_sym, s_sym], [constraint_expr])
                        
                        # Evaluate with numpy values
                        constraint_val = constraint_func(X_val[:, node_idx], S_val).full().flatten()
                        lb, ub = constraint.get_bounds()
                        
                        # Calculate violation for inequality constraints
                        violation = 0.0
                        constraint_val_flat = np.atleast_1d(constraint_val)
                        lb_flat = np.atleast_1d(lb) if hasattr(lb, '__len__') else [lb] * len(constraint_val_flat)
                        ub_flat = np.atleast_1d(ub) if hasattr(ub, '__len__') else [ub] * len(constraint_val_flat)
                        
                        for val, l, u in zip(constraint_val_flat, lb_flat, ub_flat):
                            if val < l:
                                violation += (l - val)**2
                            elif val > u:
                                violation += (val - u)**2
                        violation = np.sqrt(violation)
                        
                    elif constraint.get_constraint_type() == ConstraintType.INPUT:
                        # Create CasADi symbolic variables for function evaluation
                        u_sym = ca.MX.sym('u', U_val.shape[0])
                        
                        # Create constraint function
                        constraint_expr = constraint.evaluate(u_sym)
                        constraint_func = ca.Function('constraint', [u_sym], [constraint_expr])
                        
                        # Evaluate with numpy values
                        constraint_val = constraint_func(U_val[:, node_idx]).full().flatten()
                        
                        if constraint.is_equality:
                            expected_val = constraint.get_bounds()[0]
                            violation = np.linalg.norm(constraint_val - expected_val)
                        else:
                            lb, ub = constraint.get_bounds()
                            violation = 0.0
                            constraint_val_flat = np.atleast_1d(constraint_val)
                            lb_flat = np.atleast_1d(lb) if hasattr(lb, '__len__') else [lb] * len(constraint_val_flat)
                            ub_flat = np.atleast_1d(ub) if hasattr(ub, '__len__') else [ub] * len(constraint_val_flat)
                            
                            for val, l, u in zip(constraint_val_flat, lb_flat, ub_flat):
                                if val < l:
                                    violation += (l - val)**2
                                elif val > u:
                                    violation += (val - u)**2
                            violation = np.sqrt(violation)
                            
                    elif constraint.get_constraint_type() == ConstraintType.AUXILIARY:
                        # Create CasADi symbolic variables for function evaluation
                        x_sym = ca.MX.sym('x', X_val.shape[0])
                        u_sym = ca.MX.sym('u', U_val.shape[0])
                        s_sym = ca.MX.sym('s', len(S_val))
                        
                        # Create constraint function - note: auxiliary constraints take node_idx as last parameter
                        constraint_expr = constraint.evaluate(x_sym, u_sym, s_sym, node_idx)
                        constraint_func = ca.Function('constraint', [x_sym, u_sym, s_sym], [constraint_expr])
                        
                        # Evaluate with numpy values
                        constraint_val = constraint_func(X_val[:, node_idx], U_val[:, node_idx], S_val).full().flatten()
                        lb, ub = constraint.get_bounds()
                        
                        violation = 0.0
                        constraint_val_flat = np.atleast_1d(constraint_val)
                        lb_flat = np.atleast_1d(lb) if hasattr(lb, '__len__') else [lb] * len(constraint_val_flat)
                        ub_flat = np.atleast_1d(ub) if hasattr(ub, '__len__') else [ub] * len(constraint_val_flat)
                        
                        for val, l, u in zip(constraint_val_flat, lb_flat, ub_flat):
                            if val < l:
                                violation += (l - val)**2
                            elif val > u:
                                violation += (val - u)**2
                        violation = np.sqrt(violation)
                    else:
                        continue

                    print(f"violation([{node_idx}] {constraint.name}): {violation}")
                        
                    if violation > 1e-6:  # Only report significant violations
                        violations.append({
                            'domain': domain_name,
                            'node': node_idx,
                            'constraint': constraint.name,
                            'type': constraint.get_constraint_type().value,
                            'violation': violation,
                            'is_equality': constraint.is_equality
                        })
            
        # Analyze edge constraints
        for edge_idx, edge_name in enumerate(hybrid_system.edges):
            edge = hybrid_system.edges[edge_name]

            to_domain = edge.to_domain
            to_domain_name = to_domain.name

            from_domain = edge.from_domain
            from_domain_name = from_domain.name

            if to_domain_name not in self.variables:
                raise ValueError(f"to domain as part of edge {edge_name} not in optimizer variables!")

            # From domain variables
            X, U, S = self.variables[from_domain_name]
            X_val = self.opti.debug.value(X)
            U_val = self.opti.debug.value(U)
            S_val = self.opti.debug.value(S)

            if isinstance(S_val, float):
                S_val = [S_val]

            # To domain variables
            X_to, U_to, S_to = self.variables[to_domain_name]
            X_to_val = self.opti.debug.value(X_to)
            S_to_val = self.opti.debug.value(S_to)

            for constraint in edge.constraints:
                # Create CasADi symbolic variables for function evaluation
                x_sym = ca.MX.sym('x', X_val.shape[0])
                u_sym = ca.MX.sym('u', U_val.shape[0])
                xnext_sym = ca.MX.sym('xnext', X_to_val.shape[0])
                s_sym = ca.MX.sym('s', len(S_val))

                # Create constraint function
                constraint_expr = constraint.evaluate(x_sym, u_sym, xnext_sym, s_sym)
                constraint_func = ca.Function('constraint', [x_sym, u_sym, xnext_sym, s_sym], [constraint_expr])

                # Edge constraints connect last state of from_domain to first state of to_domain
                constraint_val = constraint_func(
                    X_val[:, -1],  # Last state of from domain
                    U_val[:, -1],  # Last input of from domain
                    X_to_val[:, 0],  # First state of to domain
                    S_val  # Parameters of from domain
                ).full().flatten()

                lb, ub = constraint.get_bounds()
                violation = 0.0
                constraint_val_flat = np.atleast_1d(constraint_val)
                lb_flat = np.atleast_1d(lb) if hasattr(lb, '__len__') else [lb] * len(constraint_val_flat)
                ub_flat = np.atleast_1d(ub) if hasattr(ub, '__len__') else [ub] * len(constraint_val_flat)

                print(f"Edge name: {edge_name}")
                for val, l, u in zip(constraint_val_flat, lb_flat, ub_flat):
                    if val < l:
                        violation += (l - val)**2
                        print(f"lb violation: {(l - val)**2}")
                    elif val > u:
                        violation += (val - u)**2
                        print(f"ub violation: {(val - u)**2}")
                violation = np.sqrt(violation)

                if violation > 1e-6:
                    violations.append({
                        'domain': f"{from_domain_name} -> {to_domain_name}",
                        'node': 'edge',
                        'constraint': constraint.name,
                        'type': 'edge_dynamics',
                        'violation': violation,
                        'is_equality': constraint.is_equality
                    })
        
        # Sort violations by magnitude (largest first)
        violations.sort(key=lambda x: x['violation'], reverse=True)
        
        # Report the top violations
        if not violations:
            print("No significant constraint violations found (all violations < 1e-6)")
        else:
            print(f"Found {len(violations)} constraint violations > 1e-6")
            print("\nTop 10 constraint violations:")
            print("-" * 95)
            print(f"{'Violation':<12} {'Domain':<35} {'Node':<8} {'Constraint':<25} {'Type':<15}")
            print("-" * 95)
            
            for i, viol in enumerate(violations[:10]):
                domain_str = viol['domain']
                # Truncate domain string if too long, but keep the full info readable
                if len(domain_str) > 35:
                    # For edge constraints, try to fit better
                    if ' -> ' in domain_str:
                        parts = domain_str.split(' -> ')
                        if len(parts[0]) + len(parts[1]) + 4 > 35:  # 4 for ' -> '
                            domain_str = f"{parts[0][:15]} -> {parts[1][:15]}"
                        
                print(f"{viol['violation']:<12.2e} {domain_str:<35} {str(viol['node']):<8} "
                      f"{viol['constraint']:<25} {viol['type']:<15}")
                
            if len(violations) > 10:
                print(f"\n... and {len(violations) - 10} more violations")
    
    def _get_constraint_active_nodes(self, domain, constraint) -> List[int]:
        """Get the list of nodes where a constraint is active."""
        active_nodes = []
        for node_idx in range(domain.num_nodes):
            if constraint in domain.get_constraints_for_node(node_idx):
                active_nodes.append(node_idx)
        return active_nodes
    
    def _get_cost_active_nodes(self, domain, cost) -> List[int]:
        """Get the list of nodes where a cost is active."""
        active_nodes = []
        for node_idx in range(domain.num_nodes):
            if cost in domain.get_costs_for_node(node_idx):
                active_nodes.append(node_idx)
        return active_nodes

    def _apply_loaded_warm_start(self, fine_system, loaded_solution):
        """Apply a loaded solution as a warm start for the optimizer variables."""
        for domain_name in fine_system.domains.keys():
            if domain_name in loaded_solution:
                fine_x = loaded_solution[domain_name]['x']
                fine_u = loaded_solution[domain_name]['u']
                fine_S = loaded_solution[domain_name]['S']
                # Set initial values
                self.opti.set_initial(self.variables[domain_name][0], fine_x)
                self.opti.set_initial(self.variables[domain_name][1], fine_u)
                if fine_S is not None:
                    self.opti.set_initial(self.variables[domain_name][2], fine_S)
                print(f"Applied loaded warm start to domain {domain_name}")