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

"""DrQ-v2 Policy implementation for LeRobot.

This implementation matches the RoboBase DrQ-v2 architecture exactly,
enabling fine-tuning of sim-trained checkpoints on real robots.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_drqv2 import DrQV2Config


# ============================================================================
# Utilities (ported from RoboBase)
# ============================================================================


def weight_init(m: nn.Module) -> None:
    """Weight initialization used by RoboBase."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def schedule(schdl: str, step: int) -> float:
    """Parse schedule string and return value at given step.

    Supports:
    - Float values: "0.5" -> 0.5
    - Linear schedules: "linear(1.0,0.1,500000)" -> linear from 1.0 to 0.1 over 500k steps
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(f"Unknown schedule: {schdl}")


class TruncatedNormal(pyd.Normal):
    """Truncated normal distribution with optional clipping on samples."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Soft update of target network parameters."""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ============================================================================
# Random Shift Augmentation (key DrQ-v2 feature)
# ============================================================================


class RandomShiftsAug(nn.Module):
    """Random shift augmentation from DrQ-v2.

    Pads the image with replicate padding and randomly crops back to original size.
    This is the key regularization technique that makes DrQ-v2 sample-efficient.
    """

    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        assert h == w, f"RandomShiftsAug requires square images, got {h}x{w}"
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


# ============================================================================
# Encoder (ported from RoboBase EncoderCNNMultiViewDownsampleWithStrides)
# ============================================================================


class DrQV2Encoder(nn.Module):
    """CNN encoder matching RoboBase's EncoderCNNMultiViewDownsampleWithStrides.

    Key features:
    - Input normalization: x / 255.0 - 0.5 (not ImageNet mean/std)
    - Per-camera conv networks
    - Strided downsampling followed by non-strided convs
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int, int],  # (V, C, H, W)
        num_downsample_convs: int = 1,
        num_post_downsample_convs: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
        padding: int = 0,
        normalise_inputs: bool = True,
    ):
        super().__init__()
        self.input_shape = input_shape
        self._normalise_inputs = normalise_inputs
        num_cameras = input_shape[0]

        self.convs_per_cam = nn.ModuleList()
        final_channels = 0

        for i in range(num_cameras):
            resolution = np.array(input_shape[2:])  # H, W
            net = []
            input_channels = input_shape[1]
            output_channels = channels

            # Downsampling convs (stride=2)
            for _ in range(num_downsample_convs):
                net.append(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                    )
                )
                net.append(nn.Identity())  # Placeholder for norm
                net.append(nn.ReLU())
                input_channels = output_channels
                resolution = np.floor((resolution + 2 * padding - kernel_size) / 2) + 1

            # Post-downsample convs (stride=1)
            for _ in range(num_post_downsample_convs):
                net.append(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    )
                )
                net.append(nn.Identity())  # Placeholder for norm
                net.append(nn.ReLU())
                input_channels = output_channels
                resolution = np.floor((resolution + 2 * padding - kernel_size) / 1) + 1

            self.convs_per_cam.append(nn.Sequential(*net))
            final_channels = int(input_channels * resolution.prod())

        self._output_shape = (num_cameras, final_channels)
        self.apply(weight_init)

    @property
    def output_shape(self) -> tuple[int, int]:
        return self._output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, V, C, H, W)

        Returns:
            Tensor of shape (B, V, features)
        """
        if self._normalise_inputs:
            x = x / 255.0 - 0.5

        outs = []
        for _x, net in zip(x.unbind(1), self.convs_per_cam):
            outs.append(net(_x).view(_x.size(0), -1))
        fused = torch.stack(outs, 1)
        return fused


# ============================================================================
# View Fusion (ported from RoboBase FusionMultiCamFeature)
# ============================================================================


class ViewFusion(nn.Module):
    """Fuses multi-camera features without learnable parameters.

    Modes:
    - flatten: concatenate all view features
    - average: average across views
    - sum: sum across views
    """

    def __init__(self, input_shape: tuple[int, int], mode: str = "flatten"):
        super().__init__()
        self.input_shape = input_shape
        self._mode = mode

        if mode == "flatten":
            self._output_shape = (np.prod(input_shape),)
        elif mode in ["average", "sum"]:
            self._output_shape = (input_shape[1],)
        else:
            raise ValueError(f"Mode {mode} is not supported.")

    @property
    def output_shape(self) -> tuple[int]:
        return self._output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, V, features)

        Returns:
            Tensor of shape (B, fused_features)
        """
        if self._mode == "flatten":
            x = x.flatten(-2)
        elif self._mode == "average":
            x = x.mean(-2)
        elif self._mode == "sum":
            x = x.sum(-2)
        return x


# ============================================================================
# MLP Networks (ported from RoboBase MLPWithBottleneckFeatures)
# ============================================================================


class Reshape(nn.Module):
    """Reshape layer for output formatting."""

    def __init__(self, shapes: tuple, last_dim_to_keep: int = 1):
        super().__init__()
        self.shapes = shapes
        self.last_dim_to_keep = last_dim_to_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[: -self.last_dim_to_keep] + self.shapes)


class MLPWithBottleneck(nn.Module):
    """MLP network with bottleneck layers for different input modalities.

    This matches RoboBase's MLPWithBottleneckFeatures architecture.
    """

    def __init__(
        self,
        input_shapes: dict[str, tuple],
        output_shape: int | tuple,
        keys_to_bottleneck: list[str],
        bottleneck_size: int = 50,
        norm_after_bottleneck: bool = True,
        tanh_after_bottleneck: bool = True,
        mlp_nodes: list[int] = None,
    ):
        super().__init__()
        if mlp_nodes is None:
            mlp_nodes = [256, 256]

        self.input_shapes = input_shapes
        self.keys_to_bottleneck = keys_to_bottleneck
        self._output_shape = (output_shape,) if isinstance(output_shape, int) else tuple(output_shape)

        # Build bottleneck layers for specified keys
        input_preprocess_modules = {}
        for k in keys_to_bottleneck:
            if k not in input_shapes:
                continue
            net = [nn.Linear(input_shapes[k][-1], bottleneck_size)]
            if norm_after_bottleneck:
                net.append(nn.LayerNorm(bottleneck_size))
            if tanh_after_bottleneck:
                net.append(nn.Tanh())
            input_preprocess_modules[k] = nn.Sequential(*net)
        self.input_preprocess_modules = nn.ModuleDict(input_preprocess_modules)

        # Calculate input size to main MLP
        inputs = [
            v[-1] for k, v in input_shapes.items() if k not in keys_to_bottleneck
        ]
        inputs_for_bottleneck = [
            bottleneck_size for k in input_shapes if k in keys_to_bottleneck
        ]
        in_size = int(np.sum(inputs + inputs_for_bottleneck))

        # Build main MLP
        main_mlp = []
        for nodes in mlp_nodes:
            main_mlp.append(nn.Linear(in_size, nodes))
            main_mlp.append(nn.Identity())  # Placeholder for norm
            main_mlp.append(nn.ReLU())
            in_size = nodes
        self.main_mlp = nn.Sequential(*main_mlp)

        # Output layer
        out_mlp = [
            nn.Linear(in_size, int(np.prod(self._output_shape))),
            Reshape(self._output_shape, 1),
        ]
        self.out_mlp = nn.Sequential(*out_mlp)

        self.apply(weight_init)

    @property
    def output_shape(self) -> tuple:
        return self._output_shape

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Dictionary of input tensors

        Returns:
            Output tensor
        """
        inputs = [v for k, v in x.items() if k not in self.keys_to_bottleneck]
        feats = []
        for k, mlp in self.input_preprocess_modules.items():
            feats.append(mlp(x[k]))
        feats.extend(inputs)
        feats = torch.cat(feats, dim=-1)
        return self.out_mlp(self.main_mlp(feats))


# ============================================================================
# Actor and Critic
# ============================================================================


class DrQV2Actor(nn.Module):
    """Actor network for DrQ-v2."""

    def __init__(self, actor_model: MLPWithBottleneck):
        super().__init__()
        self.actor_model = actor_model

    def forward(
        self,
        low_dim_obs: torch.Tensor | None,
        fused_view_feats: torch.Tensor | None,
        std: float,
    ) -> TruncatedNormal:
        """Forward pass.

        Args:
            low_dim_obs: Low-dimensional observations (B, low_dim)
            fused_view_feats: Fused visual features (B, vis_dim)
            std: Standard deviation for action distribution

        Returns:
            TruncatedNormal distribution over actions
        """
        net_ins = {}
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats

        mu = self.actor_model(net_ins)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)


class DrQV2Critic(nn.Module):
    """Critic ensemble for DrQ-v2."""

    def __init__(self, critic_models: nn.ModuleList):
        super().__init__()
        self.qs = critic_models

    def forward(
        self,
        low_dim_obs: torch.Tensor | None,
        fused_view_feats: torch.Tensor | None,
        action: torch.Tensor,
        time_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            low_dim_obs: Low-dimensional observations
            fused_view_feats: Fused visual features
            action: Actions
            time_obs: Time observations (optional)

        Returns:
            Q-values from all critics stacked along last dimension
        """
        net_ins = {}
        if low_dim_obs is not None:
            net_ins["low_dim_obs"] = low_dim_obs
        if fused_view_feats is not None:
            net_ins["fused_view_feats"] = fused_view_feats
        net_ins["action"] = action
        if time_obs is not None:
            net_ins["time_obs"] = time_obs

        qs = []
        for q in self.qs:
            qs.append(q(net_ins))
        return torch.stack(qs, dim=-1)


# ============================================================================
# DrQ-v2 Policy
# ============================================================================


class DrQV2Policy(PreTrainedPolicy):
    """DrQ-v2 Policy for LeRobot.

    This implementation matches RoboBase's DrQ-v2 architecture exactly,
    enabling loading of sim-trained checkpoints for real robot fine-tuning.
    """

    config_class = DrQV2Config
    name = "drqv2"

    def __init__(
        self,
        config: DrQV2Config,
        dataset_stats: dict | None = None,
    ):
        super().__init__(config)
        self.config = config
        config.validate_features()

        # Determine input shapes from config
        self._setup_input_shapes()

        # Build networks
        self._build_encoder()
        self._build_view_fusion()
        self._build_actor()
        self._build_critic()

        # Augmentation
        if config.use_augmentation:
            self.aug = RandomShiftsAug(pad=config.augmentation_pad)
        else:
            self.aug = nn.Identity()

        # Training state
        self._step = 0

        # Compatibility aliases for learner.py
        self.critic_ensemble = self.critic  # learner.py expects critic_ensemble
        # Dummy log_alpha for temperature (DrQ-v2 doesn't use it, but learner expects it)
        # Must have requires_grad=True for backward() to work, but loss is always 0
        self.log_alpha = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.temperature = 0.0  # Compatibility

    def _setup_input_shapes(self):
        """Setup input shapes from config."""
        config = self.config

        # Get image shape from first image feature
        image_keys = config.image_features
        if not image_keys:
            raise ValueError("DrQ-v2 requires at least one image input")

        # Assume all images have same shape - get from config
        # Shape: (num_cameras, C * frame_stack, H, W)
        self.num_cameras = len(image_keys)
        self.image_channels = 3 * config.frame_stack
        # Default to 84x84 (common for DrQ-v2)
        self.image_height = 84
        self.image_width = 84

        # Low-dim state shape
        if "observation.state" in config.input_features:
            state_feature = config.input_features.get("observation.state")
            # Handle both dict (JSON) and PolicyFeature (parsed config) formats
            if hasattr(state_feature, "shape"):
                self.low_dim_size = state_feature.shape[0]
            elif isinstance(state_feature, dict):
                self.low_dim_size = state_feature.get("shape", [0])[0]
            else:
                self.low_dim_size = 0
        else:
            self.low_dim_size = 0

        # Action shape
        action_feature = config.output_features.get("action")
        if hasattr(action_feature, "shape"):
            self.action_dim = action_feature.shape[0]
        elif isinstance(action_feature, dict):
            self.action_dim = action_feature.get("shape", [4])[0]
        else:
            self.action_dim = 4

    def _build_encoder(self):
        """Build the CNN encoder."""
        config = self.config
        input_shape = (
            self.num_cameras,
            self.image_channels,
            self.image_height,
            self.image_width,
        )
        self.encoder = DrQV2Encoder(
            input_shape=input_shape,
            num_downsample_convs=config.num_downsample_convs,
            num_post_downsample_convs=config.num_post_downsample_convs,
            channels=config.encoder_channels,
            kernel_size=config.encoder_kernel_size,
            padding=config.encoder_padding,
            normalise_inputs=config.normalise_encoder_inputs,
        )

    def _build_view_fusion(self):
        """Build the view fusion module."""
        self.view_fusion = ViewFusion(
            input_shape=self.encoder.output_shape,
            mode=self.config.view_fusion_mode,
        )

    def _build_actor(self):
        """Build the actor network."""
        config = self.config

        # Input shapes for actor
        input_shapes = {}
        if self.low_dim_size > 0:
            input_shapes["low_dim_obs"] = (self.low_dim_size,)
        input_shapes["fused_view_feats"] = self.view_fusion.output_shape

        actor_model = MLPWithBottleneck(
            input_shapes=input_shapes,
            output_shape=self.action_dim,
            keys_to_bottleneck=["fused_view_feats", "low_dim_obs"],
            bottleneck_size=config.bottleneck_size,
            norm_after_bottleneck=config.norm_after_bottleneck,
            tanh_after_bottleneck=config.tanh_after_bottleneck,
            mlp_nodes=config.mlp_nodes,
        )
        self.actor = DrQV2Actor(actor_model)

    def _build_critic(self):
        """Build the critic ensemble and target."""
        config = self.config

        # Input shapes for critic (includes action)
        input_shapes = {}
        if self.low_dim_size > 0:
            input_shapes["low_dim_obs"] = (self.low_dim_size,)
        input_shapes["fused_view_feats"] = self.view_fusion.output_shape
        input_shapes["action"] = (self.action_dim,)

        critic_models = nn.ModuleList()
        for _ in range(config.num_critics):
            critic_model = MLPWithBottleneck(
                input_shapes=input_shapes,
                output_shape=1,
                keys_to_bottleneck=["fused_view_feats", "low_dim_obs"],
                bottleneck_size=config.bottleneck_size,
                norm_after_bottleneck=config.norm_after_bottleneck,
                tanh_after_bottleneck=config.tanh_after_bottleneck,
                mlp_nodes=config.mlp_nodes,
            )
            critic_models.append(critic_model)

        self.critic = DrQV2Critic(critic_models)
        self.critic_target = deepcopy(self.critic)

        # Freeze target
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def get_std(self, step: int) -> float:
        """Get exploration std for given step."""
        return schedule(self.config.stddev_schedule, step)

    def select_action(
        self,
        batch: dict[str, torch.Tensor],
        step: int = 0,
        eval_mode: bool = True,
    ) -> torch.Tensor:
        """Select action given observations.

        Args:
            batch: Dictionary containing observations
            step: Current training step (for exploration noise)
            eval_mode: If True, use deterministic policy

        Returns:
            Action tensor
        """
        # Random exploration at start of training
        if step < self.config.num_explore_steps and not eval_mode:
            return torch.rand(batch[self.config.image_features[0]].size(0), self.action_dim) * 2 - 1

        std = self.get_std(step)

        with torch.no_grad():
            # Extract observations
            low_dim_obs = None
            if "observation.state" in batch:
                low_dim_obs = batch["observation.state"]

            # Get image observations and encode
            rgb_obs = self._extract_rgb_obs(batch)
            multi_view_feats = self.encoder(rgb_obs.float())
            fused_feats = self.view_fusion(multi_view_feats)

            # Get action distribution
            dist = self.actor(low_dim_obs, fused_feats, std)

            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample()

        return action

    def _extract_rgb_obs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and stack RGB observations.

        Args:
            batch: Dictionary containing observations

        Returns:
            Tensor of shape (B, V, C, H, W)
        """
        image_keys = self.config.image_features
        images = []
        for key in image_keys:
            img = batch[key]
            # Ensure proper shape (B, C, H, W)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            images.append(img)

        # Stack along view dimension: (B, V, C, H, W)
        return torch.stack(images, dim=1)

    def get_optim_params(self) -> dict:
        """Returns optimizer parameter groups."""
        return {
            "encoder": list(self.encoder.parameters()),
            "actor": list(self.actor.parameters()),
            "critic": list(self.critic.parameters()),
        }

    def reset(self):
        """Reset any caches. DrQ-v2 is memoryless so nothing to reset."""
        pass

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        model: str = "critic",
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        This method follows the LeRobot SAC interface where different model components
        can be updated separately via the `model` parameter.

        Args:
            batch: Dictionary containing:
                - action: Action tensor (B, action_dim)
                - reward: Reward tensor (B,) or (B, 1)
                - state: Dict of current observations
                - next_state: Dict of next observations
                - done: Done mask tensor (B,) or (B, 1)
            model: Which model to compute loss for ("critic" or "actor")

        Returns:
            Dictionary with loss tensors
        """
        if model == "critic":
            return {"loss_critic": self.compute_loss_critic(batch)}
        elif model == "actor":
            return {"loss_actor": self.compute_loss_actor(batch)}
        elif model == "temperature":
            # DrQ-v2 doesn't use temperature - return zero loss for compatibility
            # Multiply by log_alpha to make it require grad (but result is still 0)
            return {"loss_temperature": self.log_alpha.exp() * 0.0}
        else:
            raise ValueError(f"Unknown model type: {model}")

    def compute_loss_critic(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute critic loss using TD learning.

        DrQ-v2 critic loss:
        1. Apply random shift augmentation to images
        2. Encode augmented observations
        3. Compute target Q using target network
        4. MSE loss between predicted and target Q
        """
        # Extract batch components
        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        # Ensure proper shapes
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if done.dim() == 1:
            done = done.unsqueeze(1)

        # Get current step for stddev schedule
        step = self._step

        # Process current observations with augmentation
        low_dim_obs = None
        if "observation.state" in observations:
            low_dim_obs = observations["observation.state"]

        rgb_obs = self._extract_rgb_obs_from_dict(observations)

        # Apply random shift augmentation during training
        if self.training and self.config.use_augmentation:
            b, v, c, h, w = rgb_obs.shape
            rgb_obs = self.aug(rgb_obs.float().view(b * v, c, h, w)).view(b, v, c, h, w)

        # Encode current observations
        multi_view_feats = self.encoder(rgb_obs.float())
        fused_feats = self.view_fusion(multi_view_feats)

        # Compute predicted Q values
        q_values = self.critic(low_dim_obs, fused_feats, actions)  # (B, 1, num_critics)
        q_values = q_values.squeeze(1)  # (B, num_critics)

        # Compute target Q values (no gradient)
        with torch.no_grad():
            # Process next observations with augmentation
            next_low_dim_obs = None
            if "observation.state" in next_observations:
                next_low_dim_obs = next_observations["observation.state"]

            next_rgb_obs = self._extract_rgb_obs_from_dict(next_observations)

            # Apply augmentation to next observations too
            if self.training and self.config.use_augmentation:
                b, v, c, h, w = next_rgb_obs.shape
                next_rgb_obs = self.aug(next_rgb_obs.float().view(b * v, c, h, w)).view(b, v, c, h, w)

            # Encode next observations
            next_multi_view_feats = self.encoder(next_rgb_obs.float())
            next_fused_feats = self.view_fusion(next_multi_view_feats)

            # Get next actions from actor with scheduled noise
            std = self.get_std(step)
            dist = self.actor(next_low_dim_obs, next_fused_feats, std)
            next_actions = dist.sample(clip=self.config.stddev_clip)

            # Compute target Q values using target critic
            target_q_values = self.critic_target(next_low_dim_obs, next_fused_feats, next_actions)
            target_q_values = target_q_values.squeeze(1)  # (B, num_critics)

            # Take minimum across critics
            min_target_q = target_q_values.min(dim=-1, keepdim=True)[0]  # (B, 1)

            # Compute TD target: r + gamma * (1 - done) * min_Q_target
            td_target = rewards + self.config.discount * (1 - done) * min_target_q

        # Repeat TD target for each critic
        td_target = td_target.repeat(1, self.config.num_critics)

        # Compute MSE loss across all critics
        critic_loss = F.mse_loss(q_values, td_target, reduction="none")
        critic_loss = critic_loss.mean(dim=1).sum()  # Mean per sample, sum across batch for each critic

        return critic_loss

    def compute_loss_actor(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute actor loss using policy gradient.

        DrQ-v2 actor loss: -min(Q(s, pi(s)))
        No entropy term (unlike SAC) - exploration is via scheduled noise.
        """
        observations = batch["state"]
        step = self._step

        # Extract observations
        low_dim_obs = None
        if "observation.state" in observations:
            low_dim_obs = observations["observation.state"]

        rgb_obs = self._extract_rgb_obs_from_dict(observations)

        # Encode observations (detach from encoder to prevent actor gradient through encoder)
        with torch.no_grad():
            multi_view_feats = self.encoder(rgb_obs.float())
        fused_feats = self.view_fusion(multi_view_feats)
        fused_feats = fused_feats.detach()

        # Get actions from actor
        std = self.get_std(step)
        dist = self.actor(low_dim_obs, fused_feats, std)
        actions = dist.sample(clip=self.config.stddev_clip)

        # Compute Q values for actor-sampled actions
        # Detach features since encoder should only be trained via critic
        q_values = self.critic(
            low_dim_obs.detach() if low_dim_obs is not None else None,
            fused_feats.detach(),
            actions,
        )
        q_values = q_values.squeeze(1)  # (B, num_critics)

        # Actor loss: maximize Q (minimize -Q)
        min_q = q_values.min(dim=-1)[0]  # (B,)
        actor_loss = -min_q.mean()

        return actor_loss

    def _extract_rgb_obs_from_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and stack RGB observations from observation dict.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Tensor of shape (B, V, C, H, W)
        """
        # Use explicit keys if set (for checkpoint loading), otherwise use config
        if hasattr(self, "_explicit_image_keys"):
            image_keys = self._explicit_image_keys
        else:
            image_keys = self.config.image_features

        images = []

        # Try configured keys first, then fall back to any image key
        for key in image_keys:
            if key in obs_dict:
                img = obs_dict[key]
                # Ensure proper shape (B, C, H, W)
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)

        # If no configured keys found, try any key starting with observation.image
        if not images:
            for key in obs_dict:
                if key.startswith("observation.image"):
                    img = obs_dict[key]
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    images.append(img)

        if not images:
            raise ValueError(f"No image keys found in observations. Expected: {image_keys}, got: {list(obs_dict.keys())}")

        # Stack along view dimension: (B, V, C, H, W)
        return torch.stack(images, dim=1)

    def set_step(self, step: int):
        """Set current training step for stddev schedule."""
        self._step = step

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict action chunk. DrQ-v2 predicts single actions, not chunks."""
        return self.select_action(batch)

    def update_target_networks(self):
        """Soft update of target critic network."""
        soft_update_params(self.critic, self.critic_target, self.config.critic_target_tau)

    def update_temperature(self):
        """Update temperature - no-op for DrQ-v2 (no temperature learning)."""
        pass  # DrQ-v2 doesn't use temperature

    @classmethod
    def from_robobase_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "DrQV2Policy":
        """Load policy from RoboBase checkpoint.

        This method infers all dimensions from the checkpoint weights and builds
        a policy that exactly matches the checkpoint architecture.

        Args:
            checkpoint_path: Path to RoboBase .pt checkpoint
            device: Device to load to

        Returns:
            DrQV2Policy instance with loaded weights
        """
        import torch

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent_state = ckpt["agent"]
        cfg = ckpt.get("cfg", {})

        # Create policy from checkpoint
        policy = cls._from_checkpoint_weights(agent_state, cfg, device)

        return policy

    @classmethod
    def _from_checkpoint_weights(
        cls,
        state_dict: dict[str, torch.Tensor],
        cfg: dict,
        device: str,
    ) -> "DrQV2Policy":
        """Build policy from checkpoint weights by inferring dimensions."""
        # Extract config parameters
        method_cfg = cfg.get("method", {}) if hasattr(cfg, "get") else {}
        encoder_cfg = method_cfg.get("encoder_model", {}) if method_cfg else {}
        actor_cfg = method_cfg.get("actor_model", {}) if method_cfg else {}

        # Infer dimensions from weights
        # Encoder input channels from first conv weight
        encoder_key = "encoder.convs_per_cam.0.0.weight"
        if encoder_key in state_dict:
            # Shape: (out_channels, in_channels, kH, kW)
            in_channels = state_dict[encoder_key].shape[1]
            encoder_channels = state_dict[encoder_key].shape[0]
        else:
            in_channels = 9  # 3 frames * 3 RGB
            encoder_channels = 32

        # Infer fused feature dim from actor bottleneck weight
        actor_feat_key = "actor.actor_model.input_preprocess_modules.fused_view_feats.0.weight"
        if actor_feat_key not in state_dict:
            actor_feat_key = "actor_model.input_preprocess_modules.fused_view_feats.0.weight"
        if actor_feat_key in state_dict:
            # Shape: (bottleneck_size, fused_feat_dim)
            fused_feat_dim = state_dict[actor_feat_key].shape[1]
            bottleneck_size = state_dict[actor_feat_key].shape[0]
        else:
            fused_feat_dim = 39200
            bottleneck_size = 50

        # Infer low-dim state size
        lowdim_key = "actor.actor_model.input_preprocess_modules.low_dim_obs.0.weight"
        if lowdim_key not in state_dict:
            lowdim_key = "actor_model.input_preprocess_modules.low_dim_obs.0.weight"
        if lowdim_key in state_dict:
            low_dim_size = state_dict[lowdim_key].shape[1]
        else:
            low_dim_size = 0

        # Infer action dim from actor output
        actor_out_key = "actor.actor_model.out_mlp.0.weight"
        if actor_out_key not in state_dict:
            actor_out_key = "actor_model.out_mlp.0.weight"
        if actor_out_key in state_dict:
            action_dim = state_dict[actor_out_key].shape[0]
        else:
            action_dim = 4

        # Infer MLP hidden dims
        mlp_key = "actor.actor_model.main_mlp.0.weight"
        if mlp_key not in state_dict:
            mlp_key = "actor_model.main_mlp.0.weight"
        if mlp_key in state_dict:
            mlp_hidden = state_dict[mlp_key].shape[0]
            mlp_nodes = [mlp_hidden, mlp_hidden]  # Assume same size
        else:
            mlp_nodes = [256, 256]

        # Count number of critics
        num_critics = sum(1 for k in state_dict if k.startswith("critic.qs.") and ".0." in k) // 10  # rough count
        if num_critics == 0:
            num_critics = 2

        print(f"Inferred dimensions from checkpoint:")
        print(f"  - Input channels: {in_channels} (frame_stack={in_channels // 3})")
        print(f"  - Encoder channels: {encoder_channels}")
        print(f"  - Fused feature dim: {fused_feat_dim}")
        print(f"  - Low-dim state size: {low_dim_size}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Bottleneck size: {bottleneck_size}")
        print(f"  - MLP nodes: {mlp_nodes}")
        print(f"  - Num critics: {num_critics}")

        # Build policy with inferred dimensions
        policy = cls._build_from_dims(
            encoder_channels=encoder_cfg.get("channels", encoder_channels) if encoder_cfg else encoder_channels,
            num_downsample_convs=encoder_cfg.get("num_downsample_convs", 1) if encoder_cfg else 1,
            num_post_downsample_convs=encoder_cfg.get("num_post_downsample_convs", 3) if encoder_cfg else 3,
            kernel_size=encoder_cfg.get("kernel_size", 3) if encoder_cfg else 3,
            frame_stack=in_channels // 3,
            fused_feat_dim=fused_feat_dim,
            low_dim_size=low_dim_size,
            action_dim=action_dim,
            bottleneck_size=actor_cfg.get("bottleneck_size", bottleneck_size) if actor_cfg else bottleneck_size,
            norm_after_bottleneck=actor_cfg.get("norm_after_bottleneck", True) if actor_cfg else True,
            tanh_after_bottleneck=actor_cfg.get("tanh_after_bottleneck", True) if actor_cfg else True,
            mlp_nodes=list(actor_cfg.get("mlp_nodes", mlp_nodes)) if actor_cfg else mlp_nodes,
            num_critics=method_cfg.get("num_critics", num_critics) if method_cfg else num_critics,
            stddev_schedule=method_cfg.get("stddev_schedule", "linear(1.0,0.1,500000)") if method_cfg else "linear(1.0,0.1,500000)",
            stddev_clip=method_cfg.get("stddev_clip", 0.3) if method_cfg else 0.3,
            use_augmentation=method_cfg.get("use_augmentation", True) if method_cfg else True,
            device=device,
        )

        # Load weights
        policy._load_robobase_weights(state_dict)

        return policy.to(device)

    @classmethod
    def _build_from_dims(
        cls,
        encoder_channels: int,
        num_downsample_convs: int,
        num_post_downsample_convs: int,
        kernel_size: int,
        frame_stack: int,
        fused_feat_dim: int,
        low_dim_size: int,
        action_dim: int,
        bottleneck_size: int,
        norm_after_bottleneck: bool,
        tanh_after_bottleneck: bool,
        mlp_nodes: list[int],
        num_critics: int,
        stddev_schedule: str,
        stddev_clip: float,
        use_augmentation: bool,
        device: str,
    ) -> "DrQV2Policy":
        """Build policy directly from inferred dimensions without needing input_features."""
        # Create a minimal config
        config = DrQV2Config(
            encoder_channels=encoder_channels,
            num_downsample_convs=num_downsample_convs,
            num_post_downsample_convs=num_post_downsample_convs,
            encoder_kernel_size=kernel_size,
            frame_stack=frame_stack,
            bottleneck_size=bottleneck_size,
            norm_after_bottleneck=norm_after_bottleneck,
            tanh_after_bottleneck=tanh_after_bottleneck,
            mlp_nodes=mlp_nodes,
            num_critics=num_critics,
            stddev_schedule=stddev_schedule,
            stddev_clip=stddev_clip,
            use_augmentation=use_augmentation,
            device=device,
        )

        # Create policy without validation
        policy = object.__new__(cls)
        PreTrainedPolicy.__init__(policy, config)
        policy.config = config

        # Store dimensions
        policy.num_cameras = 1
        policy.image_channels = 3 * frame_stack
        policy.image_height = 84
        policy.image_width = 84
        policy.low_dim_size = low_dim_size
        policy.action_dim = action_dim

        # Build encoder
        input_shape = (1, 3 * frame_stack, 84, 84)
        policy.encoder = DrQV2Encoder(
            input_shape=input_shape,
            num_downsample_convs=num_downsample_convs,
            num_post_downsample_convs=num_post_downsample_convs,
            channels=encoder_channels,
            kernel_size=kernel_size,
            padding=0,
            normalise_inputs=True,
        )

        # Build view fusion
        policy.view_fusion = ViewFusion(
            input_shape=policy.encoder.output_shape,
            mode="flatten",
        )

        # Build actor with correct dimensions
        input_shapes = {"fused_view_feats": (fused_feat_dim,)}
        if low_dim_size > 0:
            input_shapes["low_dim_obs"] = (low_dim_size,)

        actor_model = MLPWithBottleneck(
            input_shapes=input_shapes,
            output_shape=action_dim,
            keys_to_bottleneck=["fused_view_feats", "low_dim_obs"],
            bottleneck_size=bottleneck_size,
            norm_after_bottleneck=norm_after_bottleneck,
            tanh_after_bottleneck=tanh_after_bottleneck,
            mlp_nodes=mlp_nodes,
        )
        policy.actor = DrQV2Actor(actor_model)

        # Build critic with correct dimensions
        critic_input_shapes = {"fused_view_feats": (fused_feat_dim,), "action": (action_dim,)}
        if low_dim_size > 0:
            critic_input_shapes["low_dim_obs"] = (low_dim_size,)

        critic_models = nn.ModuleList()
        for _ in range(num_critics):
            critic_model = MLPWithBottleneck(
                input_shapes=critic_input_shapes,
                output_shape=1,
                keys_to_bottleneck=["fused_view_feats", "low_dim_obs"],
                bottleneck_size=bottleneck_size,
                norm_after_bottleneck=norm_after_bottleneck,
                tanh_after_bottleneck=tanh_after_bottleneck,
                mlp_nodes=mlp_nodes,
            )
            critic_models.append(critic_model)

        policy.critic = DrQV2Critic(critic_models)
        policy.critic_target = deepcopy(policy.critic)
        for param in policy.critic_target.parameters():
            param.requires_grad = False

        # Augmentation
        if use_augmentation:
            policy.aug = RandomShiftsAug(pad=4)
        else:
            policy.aug = nn.Identity()

        policy._step = 0

        # Compatibility aliases for learner.py
        policy.critic_ensemble = policy.critic
        policy.log_alpha = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        policy.temperature = 0.0

        # Store explicit image keys for checkpoint loading
        policy._explicit_image_keys = ["observation.image"]

        return policy

    def _load_robobase_weights(self, state_dict: dict[str, torch.Tensor]):
        """Load weights from RoboBase state dict.

        Maps RoboBase key names to LeRobot key names.
        """
        new_state_dict = {}

        for key, value in state_dict.items():
            # Skip hidden states (RNN buffers)
            if "hidden_state" in key:
                continue

            new_key = self._map_robobase_key(key)
            if new_key is not None:
                new_state_dict[new_key] = value

        # Load with strict=False to allow missing keys
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    def _map_robobase_key(self, key: str) -> str | None:
        """Map RoboBase key to LeRobot key."""
        # Encoder weights
        if key.startswith("encoder."):
            return key  # Same structure

        # Actor weights
        if key.startswith("actor.actor_model."):
            return key.replace("actor.actor_model.", "actor.actor_model.")
        if key.startswith("actor_model."):
            return "actor.actor_model." + key[len("actor_model."):]

        # Critic weights
        if key.startswith("critic.qs."):
            return key  # Same structure
        if key.startswith("critic_target.qs."):
            return key  # Same structure

        return None
