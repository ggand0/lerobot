# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

"""DrQ-v2 configuration for LeRobot.

This configuration matches the RoboBase DrQ-v2 architecture exactly,
enabling fine-tuning of sim-trained checkpoints on real robots.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE


@dataclass
class ConcurrencyConfig:
    """Configuration for the concurrency of the actor and learner.
    Possible values are:
    - "threads": Use threads for the actor and learner.
    - "processes": Use processes for the actor and learner.
    """

    actor: str = "threads"
    learner: str = "threads"


@dataclass
class ActorLearnerConfig:
    """Configuration for actor-learner distributed training."""

    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@PreTrainedConfig.register_subclass("drqv2")
@dataclass
class DrQV2Config(PreTrainedConfig):
    """DrQ-v2 (Data-regularized Q-learning version 2) configuration.

    DrQ-v2 is an off-policy RL algorithm that uses image augmentation
    (random shift) as a regularizer for sample-efficient learning from pixels.

    This implementation matches RoboBase's DrQ-v2 architecture to enable
    loading pre-trained checkpoints from simulation.
    """

    # Normalization - RoboBase uses IDENTITY (no normalization) for low_dim_state
    # The pretrained actor expects raw values in radians, not normalized values
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # Handled in encoder
            "STATE": NormalizationMode.IDENTITY,   # RoboBase uses no normalization
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,  # Actions already in [-1, 1]
        }
    )

    # Dataset stats (not used for images - encoder handles normalization)
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_STATE: {"min": [0.0], "max": [1.0]},
            ACTION: {"min": [-1.0], "max": [1.0]},
        }
    )

    # Device settings
    device: str = "cuda"
    storage_device: str = "cpu"

    # Encoder settings (matches RoboBase EncoderCNNMultiViewDownsampleWithStrides)
    num_downsample_convs: int = 1
    num_post_downsample_convs: int = 3
    encoder_channels: int = 32
    encoder_kernel_size: int = 3
    encoder_padding: int = 0
    normalise_encoder_inputs: bool = True  # x / 255.0 - 0.5

    # View fusion settings
    view_fusion_mode: str = "flatten"  # "flatten", "average", or "sum"

    # Network settings (matches RoboBase MLPWithBottleneckFeatures)
    bottleneck_size: int = 50
    norm_after_bottleneck: bool = True
    tanh_after_bottleneck: bool = True
    mlp_nodes: list[int] = field(default_factory=lambda: [256, 256])

    # RNN settings (for temporal processing)
    use_rnn: bool = False  # Set True if checkpoint has RNN
    num_rnn_layers: int = 1
    rnn_hidden_size: int = 128

    # Critic settings
    num_critics: int = 2

    # DrQ-v2 specific settings
    stddev_schedule: str = "linear(1.0,0.1,500000)"
    stddev_clip: float = 0.3
    use_augmentation: bool = True
    augmentation_pad: int = 4

    # Training settings
    discount: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    encoder_lr: float = 1e-4
    weight_decay: float = 0.0
    critic_target_tau: float = 0.01
    num_explore_steps: int = 2000
    actor_grad_clip: float | None = None
    critic_grad_clip: float | None = None

    # Frame stacking
    frame_stack: int = 3

    # Online training settings
    online_steps: int = 1000000
    online_buffer_capacity: int = 100000
    offline_buffer_capacity: int = 50000  # Capacity for offline demonstration buffer
    online_step_before_learning: int = 100
    policy_update_freq: int = 1

    # Compatibility with learner.py
    num_discrete_actions: int | None = None  # DrQ-v2 doesn't use discrete actions
    shared_encoder: bool = True  # DrQ-v2 uses shared encoder
    vision_encoder_name: str | None = None  # DrQ-v2 uses custom encoder
    freeze_vision_encoder: bool = False  # DrQ-v2 trains the encoder
    grad_clip_norm: float = 1.0  # Gradient clipping
    utd_ratio: int = 1  # Update-to-data ratio (critic updates per env step)
    async_prefetch: bool = False  # Async prefetching for replay buffer

    # Pretrained model path (used when loading from checkpoint)
    pretrained_path: str | None = None

    # Skip loading pretrained critic weights (for sim-to-real transfer)
    # When True, only loads actor and encoder weights, training critic from scratch
    skip_pretrained_critic: bool = False

    # Actor-learner config for distributed training (HIL-SERL)
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)

    # Concurrency config for distributed training
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self):
        from lerobot.optim.optimizers import MultiAdamConfig

        return MultiAdamConfig(
            weight_decay=self.weight_decay,
            optimizer_groups={
                "encoder": {"lr": self.encoder_lr},
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
            },
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        has_image = any(key.startswith(OBS_IMAGE) for key in self.input_features)
        if not has_image:
            raise ValueError(
                "DrQ-v2 requires at least one image observation "
                "(key starting with 'observation.image') in the input features"
            )
        if "action" not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        """Get list of image feature keys.

        Returns keys starting with 'observation.image' from input_features.
        Falls back to default if input_features not set (e.g., when loading from checkpoint).
        """
        if not self.input_features:
            # Default fallback for checkpoint loading
            return ["observation.image"]
        return [key for key in self.input_features if key.startswith(OBS_IMAGE)]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
