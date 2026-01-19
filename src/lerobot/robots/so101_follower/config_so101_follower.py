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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@RobotConfig.register_subclass("so101_follower_end_effector")
@dataclass
class SO101FollowerEndEffectorConfig(SO101FollowerConfig):
    """Configuration for the SO101FollowerEndEffector robot."""

    # Path to MuJoCo XML model for kinematics (required for IK)
    mujoco_model_path: str | None = None

    # End-effector site name in MuJoCo model
    end_effector_site: str = "gripperframe"

    # IK parameters
    ik_damping: float = 0.1  # Damping for singularity robustness
    ik_max_dq: float = 0.5  # Max joint velocity per step (radians)

    # Joints to lock during IK (0=shoulder_pan, 1=shoulder_lift, 2=elbow_flex, 3=wrist_flex, 4=wrist_roll)
    locked_joints: list[int] = field(default_factory=lambda: [3, 4])  # Lock wrist_flex and wrist_roll by default

    # Target positions (degrees) for locked joints during teleoperation
    # Maps joint index to target angle. If not specified, defaults to 90°.
    locked_joint_positions: dict[int, float] = field(default_factory=lambda: {3: 90.0, 4: 90.0})

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    # Action scale: meters per action unit (same as sim training)
    action_scale: float = 0.02
