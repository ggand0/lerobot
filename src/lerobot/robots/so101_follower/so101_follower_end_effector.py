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
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .so101_follower import SO101Follower
from .config_so101_follower import SO101FollowerEndEffectorConfig

logger = logging.getLogger(__name__)


class SO101FollowerEndEffector(SO101Follower):
    """
    SO101Follower robot with end-effector space control using placo IK.

    This robot inherits from SO101Follower but transforms actions from
    end-effector space to joint space using placo's kinematics solver.

    Always reads actual motor positions from the bus — no state caching.
    """

    config_class = SO101FollowerEndEffectorConfig
    name = "so101_follower_end_effector"

    # Joint order (degrees) — excludes gripper
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

        # Initialize the kinematics module for the so101 robot
        if self.config.urdf_path is None:
            raise ValueError(
                "urdf_path must be provided in the configuration for end-effector control. "
                "Please set urdf_path in your SO101FollowerEndEffectorConfig."
            )

        self.kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
            joint_names=self.JOINT_NAMES,
        )

        # Store bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        logger.info(f"Initialized placo IK with URDF: {self.config.urdf_path}")
        logger.info(f"EE frame: {self.config.target_frame_name}")
        logger.info(f"Locked joints: {self.config.locked_joints}")
        logger.info(f"Locked joint positions: {self.config.locked_joint_positions}")

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
        Transform action from end-effector space to joint space using placo IK.

        Always reads actual motor positions from the bus — no caching.

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

        # Compute FK to get current EE pose (4x4 transform) — placo works in degrees
        current_ee_pose = self.kinematics.forward_kinematics(current_joints_deg)
        current_ee_pos = current_ee_pose[:3, 3]

        # Compute target EE position
        delta_xyz = action[:3] * self.config.action_scale
        target_ee_pos_unclamped = current_ee_pos + delta_xyz

        # Apply bounds
        target_ee_pos = target_ee_pos_unclamped.copy()
        if self.end_effector_bounds is not None:
            target_ee_pos = np.clip(
                target_ee_pos,
                self.end_effector_bounds["min"],
                self.end_effector_bounds["max"],
            )
            if self.config.debug_ik:
                ee_clipped = target_ee_pos - target_ee_pos_unclamped
                if np.any(np.abs(ee_clipped) > 0.001):
                    logger.warning(f"EE BOUNDS CLIPPING: clipped by {ee_clipped}m, bounds={self.end_effector_bounds}")

        # Build desired 4x4 pose (keep current orientation, set new position)
        desired_ee_pose = current_ee_pose.copy()
        desired_ee_pose[:3, 3] = target_ee_pos

        # Compute IK to get target joint positions (degrees) — placo works in degrees
        target_joints_deg = self.kinematics.inverse_kinematics(
            current_joints_deg, desired_ee_pose
        )

        # Enforce locked joint positions from config
        locked_joints = self.config.locked_joints or []
        locked_joint_positions = self.config.locked_joint_positions or {}
        for joint_idx in locked_joints:
            if joint_idx < self.N_ARM_JOINTS:
                # Try both int and string keys (JSON uses string keys)
                target_deg = locked_joint_positions.get(joint_idx,
                             locked_joint_positions.get(str(joint_idx), 90.0))
                target_joints_deg[joint_idx] = target_deg

        # Build joint action dict (5 arm joints)
        joint_action = {
            f"{name}.pos": target_joints_deg[i] for i, name in enumerate(self.JOINT_NAMES)
        }

        # Debug logging (optional)
        if self.config.debug_ik:
            joint_deltas_deg = target_joints_deg[:self.N_ARM_JOINTS] - current_joints_deg
            logger.info(
                f"SEND_ACTION: action={action[:3]}, delta_xyz_scaled={delta_xyz}, "
                f"current_ee={current_ee_pos}, target_ee={target_ee_pos}, "
                f"current_joints_deg={current_joints_deg}, target_joints_deg={target_joints_deg[:self.N_ARM_JOINTS]}, "
                f"joint_deltas_deg={joint_deltas_deg}"
            )

        # Handle gripper: [-1, 1] where -1=close, 0=no-op, 1=open
        # Legacy teleop uses [0, 2]: 0=close, 1=no-op, 2=open
        current_gripper = current_pos_dict["gripper"]
        gripper_action = action[-1]

        # Convert legacy [0, 2] format
        if gripper_action > 1.0:
            gripper_action = gripper_action - 1.0

        gripper_delta = gripper_action * self.config.max_gripper_pos
        new_gripper_pos = np.clip(
            current_gripper + gripper_delta,
            5,
            self.config.max_gripper_pos,
        )

        joint_action["gripper.pos"] = new_gripper_pos

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

        # Capture images from cameras with retry logic
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            max_retries = 5
            last_error = None
            for attempt in range(max_retries):
                try:
                    obs_dict[cam_key] = cam.async_read()
                    break
                except TimeoutError as e:
                    last_error = e
                    logger.warning(f"Camera {cam_key} timeout attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        # Try to recover camera
                        time.sleep(0.2 * (attempt + 1))
                        try:
                            # Attempt to restart async read thread
                            if hasattr(cam, 'thread') and cam.thread is not None:
                                if not cam.thread.is_alive():
                                    logger.warning(f"Camera {cam_key} read thread dead, reconnecting...")
                                    cam.disconnect()
                                    time.sleep(0.5)
                                    cam.connect()
                        except Exception as reconnect_err:
                            logger.warning(f"Camera reconnect failed: {reconnect_err}")
            else:
                logger.error(f"Camera {cam_key} timeout after {max_retries} attempts - USB may need reset.")
                raise RuntimeError(
                    f"Camera {cam_key} stopped responding after {max_retries} retries. "
                    f"Please unplug and replug the USB camera, then restart."
                ) from last_error
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
