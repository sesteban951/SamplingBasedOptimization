##
#
# Cartpole CMA-ES
#
##

# standard imports
import numpy as np
import cma

# jax imports
import jax
import jax.numpy as jnp

# custom imports
from utils.simulation.simulation import *
from utils.spline.bezier import *
from utils.spline.zoh import *
from utils.spline.linear import *
from utils.spline.cubic import *

# for base class
from abc import ABC, abstractmethod
from dataclasses import dataclass


#############################################################
# CMA-ES Configuration
#############################################################

@dataclass
class CMAES_Config:
    """Configuration for CMA-ES optimizer"""
    
    # planning horizon
    T: float                    # total time horizon
    N_knots: int                # number of spline knot points
    
    # CMA-ES parameters
    sigma0: float = 0.3         # initial step size
    popsize: int = None         # population size (None = auto)
    maxiter: int = 500          # max iterations
    
    # spline params
    spline_type: str = "Cubic"  # "ZOH" | "Linear" | "Cubic" | "Bezier"
    
    # CMA-ES options
    use_diagonal_cov: bool = False  # diagonal approximation (faster for high-D)
    verbose: bool = True             # print progress


#############################################################
# CMA-ES Base Optimizer
#############################################################

class CMAES_Optimizer(ABC):
    
    def __init__(self, model_config: Model_Config,
                       sim_config: ParallelSim_Config,
                       cmaes_config: CMAES_Config):
        
        # create the parallel sim object
        self.sim = ParallelSim(model_config, sim_config)
        
        # store config
        self.cmaes_config = cmaes_config
        
        # check params
        self._check_valid_params()
        
        # construct time array
        self.N = int(round(self.cmaes_config.T / self.sim.dt))
        self.t_sim = self.sim.dt * jnp.arange(self.N + 1)
        self.T_eff = self.N * self.sim.dt
        
        # dimension of optimization problem
        self.dim = self.cmaes_config.N_knots * self.sim.nu
        
        # initialize spline
        self._initialize_spline()
        
        # CMA-ES options
        self.cma_options = self._get_cma_options()
        
        print("CMA-ES Optimizer initialized.")
        print(f"  Dimension: {self.dim} (N_knots={self.cmaes_config.N_knots} × nu={self.sim.nu})")
        print(f"  Population size: {self.cma_options['popsize']}")
        print(f"  Spline type: {self.cmaes_config.spline_type}")
    
    
    def _check_valid_params(self):
        """Check that input params make sense"""
        if self.sim.B < 2:
            raise ValueError(f"Batch size must be at least 2")
        if self.cmaes_config.T <= 0.0:
            raise ValueError(f"Time horizon must be positive")
        if self.cmaes_config.N_knots < 2:
            raise ValueError(f"Need at least 2 knots")
    
    
    def _get_cma_options(self):
        """Configure CMA-ES options"""
        
        # auto-calculate population size if not specified
        if self.cmaes_config.popsize is None:
            # CMA-ES default: 4 + floor(3*ln(n))
            popsize = 4 + int(3 * np.log(self.dim))
            # Use available parallel resources
            popsize = min(self.sim.B, max(popsize, 50))
        else:
            popsize = self.cmaes_config.popsize
        
        # Make sure popsize doesn't exceed batch size
        if popsize > self.sim.B:
            print(f"Warning: popsize {popsize} > batch size {self.sim.B}")
            print(f"  Setting popsize = {self.sim.B}")
            popsize = self.sim.B
        
        options = {
            'popsize': popsize,
            'maxiter': self.cmaes_config.maxiter,
            'verb_disp': 1 if self.cmaes_config.verbose else 0,
            'verb_log': 0,
            'verbose': -1,  # Suppress most output
        }
        
        # Diagonal approximation for high dimensions
        if self.cmaes_config.use_diagonal_cov:
            options['CMA_diagonal'] = True
        
        return options
    
    
    def _initialize_spline(self):
        """Initialize spline with random knots within limits"""
        
        # Create initial knots WITH FULL BATCH SIZE
        y_size = (self.sim.B, self.cmaes_config.N_knots, self.sim.nu)
        Y0 = jnp.zeros(y_size)
        
        # Initialize within position or control limits
        if self.sim.use_pd:
            limits = self.sim.pos_limits
        else:
            limits = self.sim.ctrl_limits
        
        # Random initialization for all batches
        rng = jax.random.PRNGKey(0)
        for i in range(self.sim.nu):
            lo, hi = limits[i, 0], limits[i, 1]
            if abs(hi) < 1e-6 and abs(lo) < 1e-6:
                raise ValueError(f"No limits defined at actuator {i}")
            
            rng, subkey = jax.random.split(rng)
            Y0 = Y0.at[:, :, i].set(
                jax.random.uniform(subkey, 
                                 shape=(self.sim.B, self.cmaes_config.N_knots),
                                 minval=lo, 
                                 maxval=hi)
            )
        
        # Create spline object with full batch
        if self.cmaes_config.spline_type == "ZOH":
            self.spline = ZOH_Spline(Y0, self.T_eff)
        elif self.cmaes_config.spline_type == "Linear":
            self.spline = Linear_Spline(Y0, self.T_eff)
        elif self.cmaes_config.spline_type == "Cubic":
            self.spline = Cubic_Spline(Y0, self.T_eff)
        elif self.cmaes_config.spline_type == "Bezier":
            self.spline = Bezier_Spline(Y0, self.T_eff)
        else:
            raise NotImplementedError(f"Spline type {self.cmaes_config.spline_type} not implemented")
        
        # Store initial guess from first batch element (numpy for pycma)
        self.x0 = np.array(Y0[0].flatten())
        
        # Debug print
        print(f"  Spline initialized with shape: {self.spline.Y.shape}")
    
    
    @abstractmethod
    def cost(self, q, v, tau):
        """
        Cost function - must be implemented by subclass
        
        Args:
            q: shape (B, N+1, nq) - positions
            v: shape (B, N+1, nv) - velocities  
            tau: shape (B, N, nu) - controls
        Returns:
            costs: shape (B,) - cost for each trajectory
        """
        raise NotImplementedError("Must implement cost() in subclass")
    
    
    def _evaluate_population(self, population):
        """
        Evaluate entire CMA-ES population in parallel using JAX
        
        Args:
            population: list of numpy arrays, each shape (dim,)
        Returns:
            fitness: list of scalars (costs)
        """
        
        # Get population size
        B_eval = len(population)
        
        # Convert population to batch of knot matrices
        Y_batch = np.zeros((self.sim.B, self.cmaes_config.N_knots, self.sim.nu))
        for i, x in enumerate(population):
            Y_batch[i] = x.reshape(self.cmaes_config.N_knots, self.sim.nu)
        
        # Remaining batch slots are zeros (won't affect results)
        
        # Convert to JAX
        Y_batch_jax = jnp.array(Y_batch)
        
        # Update spline knots
        self.spline.update_knots(Y_batch_jax)
        
        # Evaluate spline at simulation times
        y_val = self.spline.evaluate(self.t_sim[:-1])  # (B, N, nu)
        
        # Rollout trajectories in parallel
        q_log, v_log, tau_log = self.sim.rollout(self.q0, self.v0, y_val)
        q_log.block_until_ready()
        
        # Compute costs
        J = self.cost(q_log, v_log, tau_log)  # (B,)
        J.block_until_ready()
        
        # Extract only the evaluated samples
        fitness = np.array(J[:B_eval])
        
        return fitness.tolist()
    
    
    def optimize(self, q0, v0):
        """
        Run CMA-ES optimization
        
        Args:
            q0: jnp.array, shape (nq,) or (B, nq) - initial positions
            v0: jnp.array, shape (nv,) or (B, nv) - initial velocities
        Returns:
            q_opt: shape (N+1, nq) - optimal trajectory positions
            v_opt: shape (N+1, nv) - optimal trajectory velocities
            tau_opt: shape (N, nu) - optimal controls
        """
        
        # Handle different input shapes - broadcast if needed
        if q0.ndim == 1:
            # Broadcast to batch dimension
            self.q0 = jnp.tile(q0[None, :], (self.sim.B, 1))
            self.v0 = jnp.tile(v0[None, :], (self.sim.B, 1))
        else:
            self.q0 = q0
            self.v0 = v0
        
        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(self.x0, 
                                      self.cmaes_config.sigma0,
                                      self.cma_options)
        
        # Track best solution
        best_cost = np.inf
        best_x = None
        
        print("\nStarting CMA-ES optimization...")
        print("=" * 70)
        
        # Main optimization loop
        iteration = 0
        while not es.stop():
            
            # Ask CMA-ES for new population
            population = es.ask()
            
            # Evaluate entire population in parallel using JAX
            fitness = self._evaluate_population(population)
            
            # Tell CMA-ES the results
            es.tell(population, fitness)
            
            # Track best
            min_fitness = min(fitness)
            if min_fitness < best_cost:
                best_cost = min_fitness
                best_idx = fitness.index(min_fitness)
                best_x = population[best_idx].copy()
            
            # Print progress
            if self.cmaes_config.verbose:
                iteration += 1
                print(f"Iteration {iteration:4d} | "
                      f"Best: {best_cost:.4f} | "
                      f"Mean: {np.mean(fitness):.4f} | "
                      f"Sigma: {es.sigma:.4f}")
        
        print("=" * 70)
        print(f"Optimization complete. Final cost: {best_cost:.4f}")
        print(f"Stop conditions: {es.stop()}")
        
        # Evaluate best solution to get full trajectory
        q_opt, v_opt, tau_opt = self._evaluate_single(best_x)
        
        return q_opt, v_opt, tau_opt
    
    
    def _evaluate_single(self, x):
        """Evaluate a single solution and return full trajectory"""
        
        # Reshape to knot matrix
        Y = jnp.array(x.reshape(self.cmaes_config.N_knots, self.sim.nu))
        
        # Create batch with this solution in first slot, zeros elsewhere
        Y_batch = jnp.zeros((self.sim.B, self.cmaes_config.N_knots, self.sim.nu))
        Y_batch = Y_batch.at[0].set(Y)
        
        # Update and evaluate
        self.spline.update_knots(Y_batch)
        y_val = self.spline.evaluate(self.t_sim[:-1])
        
        # Rollout
        q_log, v_log, tau_log = self.sim.rollout(self.q0, self.v0, y_val)
        q_log.block_until_ready()
        
        # Return first (best) trajectory
        return q_log[0], v_log[0], tau_log[0]




#############################################################
# Cartpole CMA-ES
#############################################################

class Cartpole_CMAES(CMAES_Optimizer):

    def __init__(self, model_config: Model_Config,
                       sim_config: ParallelSim_Config,
                       cmaes_config: CMAES_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cmaes_config)


    def cost(self, q, v, tau):
        """
        Cost function for trajectory optimization.

        Args:
            q: jnp.array, shape (B, N+1, nq) - generalized position trajectory.
            v: jnp.array, shape (B, N+1, nv) - generalized velocity trajectory.
            tau: jnp.array, shape (B, N, nu) - control input trajectory.
        Returns:
            J: jnp.array, shape (B,) - cost for each batch.
        """

        # goal state
        cart_pos_des = 0.0
        cos_des = 1.0
        sin_des = 0.0
        cart_vel_des = 0.0
        pole_vel_des = 0.0

        # cost weights
        w_cart_pos = 10.0
        w_pole_pos = 10.0
        w_cart_vel = 0.1
        w_pole_vel = 0.1
        w_tau = 0.01
        wf_cart_pos = 50.0 * w_cart_pos 
        wf_pole_pos = 50.0 * w_pole_pos
        wf_cart_vel = 50.0 * w_cart_vel
        wf_pole_vel = 50.0 * w_pole_vel
    
        # running state (t = 0 to N-1)
        cart_pos_t = q[:, :-1, 0]  # (B, N)
        theta_t =    q[:, :-1, 1]  # (B, N)
        cart_vel_t = v[:, :-1, 0]  # (B, N)
        pole_vel_t = v[:, :-1, 1]  # (B, N)
        cos_t = jnp.cos(theta_t)   # (B, N)
        sin_t = jnp.sin(theta_t)   # (B, N)

        # terminal states (t = N)
        cart_pos_T = q[:, -1, 0]   # (B,)
        theta_T =    q[:, -1, 1]   # (B,)
        cart_vel_T = v[:, -1, 0]   # (B,)
        pole_vel_T = v[:, -1, 1]   # (B,)
        cos_T = jnp.cos(theta_T)   # (B,)
        sin_T = jnp.sin(theta_T)   # (B,)

        # ---------------------------------------------------
        # RUNNING COST
        # ---------------------------------------------------

        # Quadratic costs
        cart_pos_running = w_cart_pos * jnp.square(cart_pos_t - cart_pos_des)
        pole_pos_running = w_pole_pos * (
            jnp.square(cos_t - cos_des) + jnp.square(sin_t - sin_des)
        )
        cart_vel_running = w_cart_vel * jnp.square(cart_vel_t - cart_vel_des)
        pole_vel_running = w_pole_vel * jnp.square(pole_vel_t - pole_vel_des)
        tau_running = w_tau * jnp.sum(jnp.square(tau), axis=-1)

        running_cost = jnp.sum(
              cart_pos_running
            + pole_pos_running
            + cart_vel_running
            + pole_vel_running
            + tau_running
            , axis=-1
        ) * self.sim.dt

        # ---------------------------------------------------
        # TERMINAL COST
        # ---------------------------------------------------

        cart_pos_terminal = wf_cart_pos * jnp.square(cart_pos_T - cart_pos_des)
        pole_pos_terminal = wf_pole_pos * (
            jnp.square(cos_T - cos_des) + jnp.square(sin_T - sin_des)
        )
        cart_vel_terminal = wf_cart_vel * jnp.square(cart_vel_T - cart_vel_des)
        pole_vel_terminal = wf_pole_vel * jnp.square(pole_vel_T - pole_vel_des)

        terminal_cost = (
            cart_pos_terminal
            + pole_pos_terminal
            + cart_vel_terminal
            + pole_vel_terminal
        )
        
        # total cost
        J = running_cost + terminal_cost

        return J


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import time
    import os

    # print device
    print(f"Using device: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU device: {gpu_info}")

    # fix the random seed
    np.random.seed(0)

    # model config
    model_config = Model_Config(
        xml_path="./models/cartpole/cartpole.xml",
        Kp=[500.0], 
        Kd=[50.0],  
        q_actuated_idx=[0],
        v_actuated_idx=[0],
        action_mode="pos",
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size=4096,
    )

    # CMA-ES config
    cmaes_config = CMAES_Config(
        T=3.0,
        N_knots=10,
        spline_type="Cubic",
        sigma0=0.3,
        popsize=2048,  # Use lots of parallel evals!
        maxiter=200,
        use_diagonal_cov=False,
        verbose=True
    )

    # create the CMA-ES optimizer
    cmaes_optimizer = Cartpole_CMAES(
        model_config=model_config,
        sim_config=sim_config,
        cmaes_config=cmaes_config
    )

    # optimize from initial state
    t0 = time.time()
    q_opt, v_opt, tau_opt = cmaes_optimizer.optimize(
        q0=jnp.array([0.0, np.pi]),   # cart at 0, pole hanging down
        v0=jnp.array([0.0, 0.0])      # zero velocity
    )
    times = cmaes_optimizer.t_sim
    tf = time.time()
    print(f"\nOptimization took {tf - t0:.2f} seconds.")

    # convert to numpy for saving
    times = np.array(times)
    q_opt = np.array(q_opt)
    v_opt = np.array(v_opt)
    tau_opt = np.array(tau_opt)

    # save results
    save_dir = "./results/cartpole_cmaes/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    np.savetxt(save_dir + "time.csv", times, delimiter=",")
    np.savetxt(save_dir + "q_opt.csv", q_opt, delimiter=",")
    np.savetxt(save_dir + "v_opt.csv", v_opt, delimiter=",")
    np.savetxt(save_dir + "tau_opt.csv", tau_opt, delimiter=",")
    print(f"Saved results to {save_dir}")