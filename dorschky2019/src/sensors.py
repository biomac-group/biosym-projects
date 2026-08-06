import os
from typing import List, Dict, Any
import sympy as sp
from sympy import Matrix, lambdify
from sympy.physics.mechanics import Point, ReferenceFrame
import jax
import jax.numpy as jnp
from jax import tree_util

DEFAULT_IMU_CONFIG = [
    {"body": "pelvis", "offset": [0.0, 0.0, 0.0]},
    {"body": "femur_r", "offset": [0.0, 0.0, 0.0]},
    {"body": "tibia_r", "offset": [0.0, 0.0, 0.0]},
    {"body": "foot_r", "offset": [0.0, 0.0, 0.0]},
    {"body": "femur_l", "offset": [0.0, 0.0, 0.0]},
    {"body": "tibia_l", "offset": [0.0, 0.0, 0.0]},
    {"body": "foot_l", "offset": [0.0, 0.0, 0.0]},
]

class IMUSensorModel:
    """
    Custom IMU sensor model mapping musculoskeletal states and constants
    to simulated accelerometer and gyroscope signals.
    """

    def __init__(self, model: Any, config: List[Dict[str, Any]] = None):
        """
        Initialize the IMU sensor model.

        :param model: The loaded BiosymModel instance.
        :param config: A list of dicts specifying sensor placements. Each dict should have:
                       - "body": Name of the body (e.g. "pelvis")
                       - "offset": local [x, y, z] offset relative to body origin.
        """
        self.model = model
        self.config = config if config is not None else DEFAULT_IMU_CONFIG
        self.n_sensors = len(self.config)
        self._derive_symbolic_equations()

    def _derive_symbolic_equations(self):
        """
        Derive symbolic equations for the IMU sensors and compile them to JAX.
        """
        print("Deriving symbolic equations for IMU sensors...")
        symbolic_list = []

        # Proper gravity vector in global frame (g is a constant defined in model)
        g_vector = (
            self.model.g.symbols[0] * self.model.ground_frame.x
            + self.model.g.symbols[1] * self.model.ground_frame.y
            + self.model.g.symbols[2] * self.model.ground_frame.z
        )

        for cfg in self.config:
            body_name = cfg["body"]
            offset = cfg.get("offset", [0.0, 0.0, 0.0])

            # Ensure body exists in the model
            if body_name not in self.model.reference_frames:
                raise ValueError(f"Body '{body_name}' not found in the model reference frames.")

            body_frame = self.model.reference_frames[body_name]
            body_origin = self.model.body_origins[body_name]

            # Define sensor point fixed on the body
            sensor_pt = Point(f"{body_name}_imu_sensor")
            offset_vec = offset[0] * body_frame.x + offset[1] * body_frame.y + offset[2] * body_frame.z
            sensor_pt.set_pos(body_origin, offset_vec)
            sensor_pt.set_vel(body_frame, 0)
            
            # Kinematics: compute position, velocity and acceleration relative to ground frame
            sensor_pt.v2pt_theory(body_origin, self.model.ground_frame, body_frame)
            acc_vec = sensor_pt.acc(self.model.ground_frame)

            # Proper acceleration in global frame: acceleration minus gravity
            proper_acc_vec = acc_vec - g_vector

            # Proper acceleration components in the local body frame
            acc_local = [
                proper_acc_vec.dot(body_frame.x),
                proper_acc_vec.dot(body_frame.y),
                proper_acc_vec.dot(body_frame.z)
            ]

            # Angular velocity of body in global frame
            ang_vel = body_frame.ang_vel_in(self.model.ground_frame)
            omega_local = [
                ang_vel.dot(body_frame.x),
                ang_vel.dot(body_frame.y),
                ang_vel.dot(body_frame.z)
            ]

            # Add to the full output vector: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            symbolic_list.extend(acc_local)
            symbolic_list.extend(omega_local)

        # Convert to matrix
        imu_matrix = Matrix(symbolic_list)

        # Replace dynamicsymbols (q, qd, and their derivatives) with flat vector variables v
        print("Replacing dynamic symbols in IMU equations...")
        imu_matrix_replaced = self.model._replace_dyn(imu_matrix, _replace_d_q=True)

        # Lambdify using JAX
        print("Lambdifying IMU equations to JAX...")
        self.imu_fun_uncompiled = lambdify(
            self.model._symbols,
            imu_matrix_replaced,
            modules="jax",
            cse=True,
            docstring_limit=2
        )

        # Build JIT-compiled and vectorized JAX functions
        self._build_jax_interfaces()

    def _build_jax_interfaces(self):
        """
        Build JIT-compiled single-point and vectorized (batch) functions.
        """
        # Uncompiled lambdified function
        fun = self.imu_fun_uncompiled

        @jax.jit
        def imu_single(states_model: jnp.ndarray, constants_model: jnp.ndarray) -> jnp.ndarray:
            """
            Compute IMU signals for a single time node.
            :param states_model: Flat state vector (positions, velocities, accelerations) of shape (n_states,)
            :param constants_model: Flat constants vector of shape (n_constants,)
            :return: Flat array of shape (n_sensors * 6,)
            """
            flat_inputs = jnp.concatenate([states_model, constants_model])
            # Lambdified function outputs (N_outputs, 1) or similar. Squeeze to 1D.
            return jnp.squeeze(fun(*flat_inputs))

        @jax.jit
        def imu_batch(states_model_batch: jnp.ndarray, constants_model: jnp.ndarray) -> jnp.ndarray:
            """
            Compute IMU signals for a batch of time nodes.
            :param states_model_batch: Array of shape (n_nodes, n_states)
            :param constants_model: Flat constants vector of shape (n_constants,)
            :return: Array of shape (n_nodes, n_sensors * 6)
            """
            # Use vmap to vectorize over the state nodes, keeping constants fixed
            single_fn = lambda s: imu_single(s, constants_model)
            return jax.vmap(single_fn)(states_model_batch)

        self.compute_single = imu_single
        self.compute_batch = imu_batch

    def get_sensor_names(self) -> List[str]:
        """
        Get the list of signal names generated by the sensors in order.
        """
        names = []
        for cfg in self.config:
            b = cfg["body"]
            names.extend([
                f"{b}_acc_x", f"{b}_acc_y", f"{b}_acc_z",
                f"{b}_gyr_x", f"{b}_gyr_y", f"{b}_gyr_z"
            ])
        return names
