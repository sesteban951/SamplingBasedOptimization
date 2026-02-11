##
#
# Class that computes a bunch of usefuls dynamics quantities
# for a given Mujoco MJX model.
#

# standard imports
from dataclasses import dataclass

# jax imports
import jax
import jax.numpy as jnp

# mujoco imports
import mujoco 
import mujoco.mjx as mjx


#############################################################
# DYNAMICS PROPERTIES CLASS
#############################################################

# model config
@dataclass
class Dynamics_Config:

    # model parameters
    xml_path: str

    # number of parallerl environments
    num_envs: int


class Dynamics:
    """
    Helper class to perform parallel computation of useful
    dynamics quantities for a given Mujoco MJX model.

    Args:
        dynamics_config (Dynamics_Config): configuration dataclass for dynamics properties
    """

    def __init__(self, dynamics_config: Dynamics_Config):

        # load the model
        self._initialize_model(dynamics_config)

        # initialize jit functions
        self._initialize_jit_functions()

        print("Initialized the dynamics class")

    
    ################################## INITIALIZATION ##################################

    def _initialize_model(self, dynamics_config: Dynamics_Config):
        """
        Initializes the parallel Mujoco environments.

        Args: 
            dynamics_config (Dynamics_Config): configuration dataclass for dynamics properties
        """

        # mujoco model
        mj_model = mujoco.MjModel.from_xml_path(dynamics_config.xml_path)
        mj_data = mujoco.MjData(mj_model)

        # put the model on GPU
        self.mjx_model = mjx.put_model(mj_model)
        data0 = mjx.put_data(mj_model, mj_data)

        # load sizes
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        # batch dimension
        self.B = dynamics_config.num_envs

        # replicate data0 across batch dimension
        self.data0 = jax.vmap(
            lambda _: mjx.put_data(mj_model, mj_data)
        )(jnp.arange(self.B))

    
    def _initialize_jit_functions(self):
        """
        Initializes the jit functions for computing dynamics properties.
        """

        # jit the com state in world function
        self.com_state_in_world = jax.jit(self.com_state_in_world)


    ################################## UTILS ##################################

    def _com_state_in_world_single_env(self, model , data):
        """
        Computes the center of mass position and velocity in the 
        world frame for a single environment.
        """
        data = mjx.kinematics(model, data)
        data = mjx.com_pos(model, data)
        com_pos = data.subtree_com[0]

        data = mjx.com_vel(model, data)
        data = mjx.subtree_vel(model, data)  
        com_vel = data._impl.subtree_linvel[0]

        return com_pos, com_vel


    def com_state_in_world(self, q, v):
        """
        Computes the center of mass position and velocity in the world frame 
        for all parallel environments.

        Args:
            q (jnp.ndarray): (B, nq) array of generalized positions 
            v (jnp.ndarray): (B, nv) array of generalized velocities
        Returns:
            p_com (jnp.ndarray): (B, 3) array of COM positions
            v_com (jnp.ndarray): (B, 3) array of COM velocities
        """
        # build batched data from q and v, no mutation
        data = self.data0.replace(qpos=q, qvel=v)

        p_com, v_com = jax.vmap(
            lambda d: self._com_state_in_world_single_env(self.mjx_model, d)
        )(data)

        return p_com, v_com
    



#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import time

    # configure with a built-in mujoco model
    config = Dynamics_Config(
        xml_path="./models/g1/g1_21dof.xml",  # e.g. a humanoid or ant model
        # xml_path="./models/biped/biped.xml",  # e.g. a humanoid or ant model
        num_envs=4
    )

    # initialize
    dyn = Dynamics(config)

    # create dummy q and v
    q = jnp.zeros((dyn.B, dyn.nq))
    v = jnp.zeros((dyn.B, dyn.nv))

    q = q.at[:, 3].set(1.0)  # quat
    q = q.at[:, 4].set(0.0)  # 
    q = q.at[:, 5].set(0.0)  # 
    q = q.at[:, 6].set(0.0)  # 

    q = q.at[0, 0].set(0.1)  # env 1 pos
    q = q.at[0, 1].set(0.2)  # 
    q = q.at[0, 2].set(0.3)  # 
    q = q.at[0, 7].set(0.1)  # 
    q = q.at[0, 8].set(0.1)  # 

    q = q.at[1, 0].set(0.4)  # env 2 pos
    q = q.at[1, 1].set(0.5)  # 
    q = q.at[1, 2].set(0.6)  # 

    v = v.at[0, 0].set(4.0)  # env 1 vel
    v = v.at[0, 1].set(5.0)  # 
    v = v.at[0, 2].set(6.0)  # 

    v = v.at[0, 6].set(1.0)  #

    v = v.at[1, 0].set(4.0)  # env 2 vel
    v = v.at[1, 1].set(5.0)  # 
    v = v.at[1, 2].set(6.0)  # 

    # first call will trigger jit compilation (slow)
    t0 = time.time()
    p_com, v_com = dyn.com_state_in_world(q, v)
    t1 = time.time()
    print("p_com shape:", p_com.shape)  # expect (4, 3)
    print("p_com:\n", p_com)  # expect (4, 3)
    print("v_com shape:", v_com.shape)  # expect (4, 3)
    print("v_com:\n", v_com)  # expect (4, 3)
    print("time: ", t1 - t0, "seconds")

    _, vc1 = dyn.com_state_in_world(q, v)
    v_shift = v.at[:, 0].add(2.0)
    _, vc2 = dyn.com_state_in_world(q, v_shift)
    print(vc2[:,0] - vc1[:,0])  # should be ~2.0 for envs with same pose