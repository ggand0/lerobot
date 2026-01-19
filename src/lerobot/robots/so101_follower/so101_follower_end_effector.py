#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .so101_follower import SO101Follower
from .config_so101_follower import SO101FollowerEndEffectorConfig

logger = logging.getLogger(__name__)


class SO101FollowerEndEffector(SO101Follower):
    """
    SO101Follower robot with end-effector space control using MuJoCo IK.

    This robot inherits from SO101Follower but transforms actions from
    end-effector space to joint space using MuJoCo's damped least-squares IK.
    """

    config_class = SO101FollowerEndEffectorConfig
    name = "so101_follower_end_effector"

    # Joint order in MuJoCo model (degrees)
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    N_ARM_JOINTS = 5

    def __init__(self, config: SO101FollowerEndEffectorConfig):
        super().__init__(config)
        # Override bus with DEGREES mode for kinematics compatibility
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.cameras = make_cameras_from_configs(config.cameras)
        self.config = config

        # Initialize MuJoCo model for IK
        if self.config.mujoco_model_path is None:
            raise ValueError(
                "mujoco_model_path must be provided in the configuration for end-effector control. "
                "Please set mujoco_model_path in your SO101FollowerEndEffectorConfig."
            )

        model_path = Path(self.config.mujoco_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        # Get end-effector site ID
        self.ee_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.config.end_effector_site
        )
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{self.config.end_effector_site}' not found in MuJoCo model")

        # Pre-allocate Jacobians
        self.jacp = np.zeros((3, self.mj_model.nv))
        self.jacr = np.zeros((3, self.mj_model.nv))

        # Joint limits from model (radians)
        self.joint_limits_lower = np.array([self.mj_model.jnt_range[i, 0] for i in range(self.N_ARM_JOINTS)])
        self.joint_limits_upper = np.array([self.mj_model.jnt_range[i, 1] for i in range(self.N_ARM_JOINTS)])

        # Store bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        logger.info(f"Initialized MuJoCo IK with model: {model_path}")
        logger.info(f"EE site: {self.config.end_effector_site} (id={self.ee_site_id})")
        logger.info(f"Locked joints: {self.config.locked_joints}")
        logger.info(f"Locked joint positions: {getattr(self.config, 'locked_joint_positions', {})}")

    def _sync_mujoco(self, joint_positions_rad: np.ndarray):
        """Sync MuJoCo model state with joint positions (radians)."""
        n_joints = min(len(joint_positions_rad), self.N_ARM_JOINTS)
        self.mj_data.qpos[:n_joints] = joint_positions_rad[:n_joints]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _get_ee_position(self) -> np.ndarray:
        """Get current end-effector position from MuJoCo model."""
        return self.mj_data.site_xpos[self.ee_site_id].copy()

    def _compute_ik(
        self,
        target_pos: np.ndarray,
        current_joints_rad: np.ndarray,
    ) -> np.ndarray:
        """Compute target joint positions using damped least-squares IK.

        Args:
            target_pos: Target end-effector position (3,) in meters.
            current_joints_rad: Current joint positions (5,) in radians.

        Returns:
            Target joint positions (5,) in radians.
        """
        # Sync model with current joints
        self._sync_mujoco(current_joints_rad)

        # Position error
        current_pos = self._get_ee_position()
        pos_error = target_pos - current_pos

        # Compute Jacobian
        mujoco.mj_jacSite(
            self.mj_model, self.mj_data, self.jacp, self.jacr, self.ee_site_id
        )

        # Active joints (exclude locked ones)
        locked = self.config.locked_joints or []
        active_joints = [i for i in range(self.N_ARM_JOINTS) if i not in locked]
        n_active = len(active_joints)
        Jp = self.jacp[:, active_joints]

        # Damped least-squares
        JTJ = Jp.T @ Jp
        damping_matrix = self.config.ik_damping ** 2 * np.eye(n_active)

        try:
            dq_active = np.linalg.solve(JTJ + damping_matrix, Jp.T @ pos_error)
        except np.linalg.LinAlgError:
            dq_active = np.linalg.pinv(Jp) @ pos_error

        # Clamp velocity
        dq_active = np.clip(dq_active, -self.config.ik_max_dq, self.config.ik_max_dq)

        # Build target joint positions
        target_joints = current_joints_rad.copy()
        for i, joint_idx in enumerate(active_joints):
            target_joints[joint_idx] += dq_active[i]

        # Clamp to joint limits
        target_joints = np.clip(target_joints, self.joint_limits_lower, self.joint_limits_upper)

        return target_joints

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space using MuJoCo IK.

        Args:
            action: Dictionary with keys 'delta_x', 'delta_y', 'delta_z' for end-effector control
                   or a numpy array with [delta_x, delta_y, delta_z, gripper]

        Returns:
            The joint-space action that was sent to the motors
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Convert action to numpy array if dict
        if isinstance(action, dict):
            # Check if this is a joint-space action (from teleoperation)
            joint_keys = [f"{motor}.pos" for motor in self.bus.motors]
            if any(k in action for k in joint_keys):
                # Pass through joint-space actions directly to parent class
                return SO101Follower.send_action(self, action)
            elif all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                delta_xyz = np.array(
                    [action["delta_x"], action["delta_y"], action["delta_z"]],
                    dtype=np.float32,
                )
                gripper = action.get("gripper", 1.0)
                action = np.append(delta_xyz, gripper)
            else:
                logger.warning(
                    f"Expected action keys 'delta_x', 'delta_y', 'delta_z' or joint keys, got {list(action.keys())}"
                )
                action = np.zeros(4, dtype=np.float32)

        # ALWAYS read current joint positions from robot (not cached)
        # Caching causes internal state to diverge from reality
        current_pos_dict = self.bus.sync_read("Present_Position")
        current_joints_deg = np.array([current_pos_dict[name] for name in self.JOINT_NAMES])
        current_joints_rad = np.deg2rad(current_joints_deg)

        # Get current EE position from MuJoCo FK
        self._sync_mujoco(current_joints_rad)
        current_ee_pos = self._get_ee_position()

        # Compute target EE position
        delta_xyz = action[:3] * self.config.action_scale
        target_ee_pos = current_ee_pos + delta_xyz

        # Apply bounds
        if self.end_effector_bounds is not None:
            target_ee_pos = np.clip(
                target_ee_pos,
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )

        # Compute IK to get target joint positions (radians)
        target_joints_rad = self._compute_ik(target_ee_pos, current_joints_rad)

        # Enforce locked joint positions from config (IK just preserves current, we need target)
        locked_joints = self.config.locked_joints or []
        locked_joint_positions = getattr(self.config, 'locked_joint_positions', {})
        for joint_idx in locked_joints:
            if joint_idx < len(target_joints_rad):
                # Try both int and string keys (JSON uses string keys)
                target_deg = locked_joint_positions.get(joint_idx,
                             locked_joint_positions.get(str(joint_idx), 90.0))
                target_joints_rad[joint_idx] = np.deg2rad(target_deg)

        target_joints_deg = np.rad2deg(target_joints_rad)

        # Build joint action dict
        joint_action = {
            f"{name}.pos": target_joints_deg[i] for i, name in enumerate(self.JOINT_NAMES)
        }

        # Handle gripper (action in [0, 2] where 1 = no-op)
        current_gripper = current_pos_dict["gripper"]
        gripper_delta = (action[-1] - 1) * self.config.max_gripper_pos
        joint_action["gripper.pos"] = np.clip(
            current_gripper + gripper_delta,
            5,
            self.config.max_gripper_pos,
        )

        # Send to parent class
        return super().send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                obs_dict[cam_key] = cam.async_read()
            except TimeoutError as e:
                logger.error(f"Camera {cam_key} timeout - USB may need reset. Unplug and replug the camera.")
                raise RuntimeError(
                    f"Camera {cam_key} stopped responding. "
                    f"Please unplug and replug the USB camera, then restart."
                ) from e
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        """Reset internal state."""
        pass  # No cached state to reset anymore
