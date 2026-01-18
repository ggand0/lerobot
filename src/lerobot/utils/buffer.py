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

import functools
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.transition import Transition


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to euler angles (XYZ convention)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ):
        """
        Replay buffer for storing transitions.
        It will allocate tensors on the specified device, when the first transition is added.
        NOTE: If you encounter memory issues, you can try to use the `optimize_memory` flag to save memory or
        and use the `storage_device` flag to store the buffer on a different device.
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory

        # Track episode boundaries for memory optimization
        self.episode_ends = torch.zeros(capacity, dtype=torch.bool, device=storage_device)

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        self.image_augmentation_function = image_augmentation_function

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # Pre-allocate tensors for storage
        self.states = {
            key: torch.empty((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.empty((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.empty((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)

        # Initialize storage for complementary_info
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # Pre-allocate tensors for each key in complementary_info
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.squeeze(0).shape
                    self.complementary_info[key] = torch.empty(
                        (self.capacity, *value_shape), device=self.storage_device
                    )
                elif isinstance(value, (int, float)):
                    # Handle scalar values similar to reward
                    self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Saves a transition, ensuring tensors are stored on the designated storage device."""
        # Initialize storage if this is the first transition
        if not self.initialized:
            self._initialize_storage(state=state, action=action, complementary_info=complementary_info)

        # Store the transition in pre-allocated tensors
        for key in self.states:
            self.states[key][self.position].copy_(state[key].squeeze(dim=0))

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_states[key][self.position].copy_(next_state[key].squeeze(dim=0))

        self.actions[self.position].copy_(action.squeeze(dim=0))
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncateds[self.position] = truncated

        # Handle complementary_info if provided and storage is initialized
        if complementary_info is not None and self.has_complementary_info:
            # Store the complementary_info
            for key in self.complementary_info_keys:
                if key in complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][self.position].copy_(value.squeeze(dim=0))
                    elif isinstance(value, (int, float)):
                        self.complementary_info[key][self.position] = value

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions and collate them into batched tensors."""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        batch_size = min(batch_size, self.size)
        high = max(0, self.size - 1) if self.optimize_memory and self.size < self.capacity else self.size

        # Random indices for sampling - create on the same device as storage
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)

        # Identify image keys that need augmentation
        image_keys = [k for k in self.states if k.startswith("observation.image")] if self.use_drq else []

        # Create batched state and next_state
        batch_state = {}
        batch_next_state = {}

        # First pass: load all state tensors to target device
        for key in self.states:
            batch_state[key] = self.states[key][idx].to(self.device)

            if not self.optimize_memory:
                # Standard approach - load next_states directly
                batch_next_state[key] = self.next_states[key][idx].to(self.device)
            else:
                # Memory-optimized approach - get next_state from the next index
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # Optimization: Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # Calculate offsets for the current image key:
                # For each key, we have 2*batch_size images (batch_size for states, batch_size for next_states)
                # States start at index i*2*batch_size and take up batch_size slots
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next states start after the states at index (i*2+1)*batch_size and also take up batch_size slots
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # Sample other tensors
        batch_actions = self.actions[idx].to(self.device)
        batch_rewards = self.rewards[idx].to(self.device)
        batch_dones = self.dones[idx].to(self.device).float()
        batch_truncateds = self.truncateds[idx].to(self.device).float()

        # Sample complementary_info if available
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][idx].to(self.device)

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)

        Yields:
            BatchTransition: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size)

            # Yield all items from the iterator
            with suppress(StopIteration):
                yield from iterator

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Create an iterator that continuously yields prefetched batches in a
        background thread. The design is intentionally simple and avoids busy
        waiting / complex state management.

        Args:
            batch_size (int): Size of batches to sample.
            queue_size (int): Maximum number of prefetched batches to keep in
                memory.

        Yields:
            BatchTransition: A batch sampled from the replay buffer.
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """Continuously put sampled batches into the queue until shutdown."""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size)
                    # The timeout ensures the thread unblocks if the queue is full
                    # and the shutdown event gets set meanwhile.
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # Queue is full – loop again (will re-check shutdown_event)
                    continue
                except Exception:
                    # Surface any unexpected error and terminate the producer.
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # If the producer already set the shutdown flag we exit.
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # Drain the queue quickly to help the thread exit if it's blocked on `put`.
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # Give the producer thread a bit of time to finish.
            producer_thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            queue_size (int): Number of initial batches to prefetch

        Yields:
            BatchTransition: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        image_size: tuple[int, int] | None = None,
        compute_full_proprioception: bool = False,
        mujoco_model_path: str | None = None,
        ee_site_name: str = "gripper",
        fps: float = 30.0,
        convert_actions_to_ee: bool = False,
        ee_action_scale: float = 0.02,
        target_action_dim: int | None = None,
        frame_stack: int = 1,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Uses streaming to avoid loading all transitions into memory at once.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity. If None, uses dataset length.
            action_mask (Sequence[int] | None): Indices of action dimensions to keep.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.
            image_size (tuple[int, int] | None): Target size (H, W) to resize images. If None, no resize.
            compute_full_proprioception (bool): If True, expand state to 18-dim with vel and FK.
            mujoco_model_path (str | None): Path to MuJoCo model for FK computation.
            ee_site_name (str): End-effector site name in MuJoCo model.
            fps (float): Dataset FPS for velocity computation.
            convert_actions_to_ee (bool): If True, convert joint actions to EE delta actions.
            ee_action_scale (float): Scale for EE delta actions (to denormalize).
            target_action_dim (int | None): Target action dimension (e.g., 4 for xyz+gripper).
            frame_stack (int): Number of frames to stack for observations. Default 1 (no stacking).

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to ReplayBuffer.")

        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        num_frames = len(lerobot_dataset)
        if num_frames == 0:
            return replay_buffer

        # Initialize MuJoCo for FK computation if computing full proprioception
        mj_model = None
        mj_data = None
        ee_site_id = None
        dt = 1.0 / fps
        prev_joint_pos = None

        # Initialize MuJoCo if needed for full proprioception or action conversion
        if compute_full_proprioception or convert_actions_to_ee:
            if mujoco_model_path is None:
                raise ValueError(
                    "mujoco_model_path must be provided when compute_full_proprioception=True or convert_actions_to_ee=True"
                )
            import mujoco
            mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path)
            mj_data = mujoco.MjData(mj_model)
            ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
            if ee_site_id < 0:
                raise ValueError(f"Site '{ee_site_name}' not found in MuJoCo model")
            print(f"Initialized MuJoCo FK from {mujoco_model_path} (site: {ee_site_name})")
            if convert_actions_to_ee:
                print(f"Converting actions to EE space (scale: {ee_action_scale}, target_dim: {target_action_dim})")

        # Check dataset features from first sample
        sample = lerobot_dataset[0]
        has_done_key = "next.done" in sample
        has_reward_key = "next.reward" in sample
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")
        if not has_reward_key:
            print("'next.reward' key not found in dataset. Using 0.0 as default reward...")

        # Frame stacking setup
        use_frame_stacking = frame_stack > 1
        if use_frame_stacking:
            from collections import deque as frame_deque
            # Identify image keys and state keys for stacking
            image_keys_to_stack = [k for k in state_keys if ".images." in k]
            state_key = "observation.state" if "observation.state" in state_keys else None
            # Initialize frame buffers (will be reset per episode)
            frame_buffers: dict[str, deque] = {}
            print(f"Frame stacking enabled: {frame_stack} frames")
            print(f"  Image keys to stack: {image_keys_to_stack}")
            print(f"  State key to stack: {state_key}")

        def reset_frame_buffers(initial_state: dict[str, torch.Tensor]) -> None:
            """Reset frame buffers with copies of initial state."""
            if not use_frame_stacking:
                return
            for key in image_keys_to_stack:
                frame_buffers[key] = frame_deque(maxlen=frame_stack)
                for _ in range(frame_stack):
                    frame_buffers[key].append(initial_state[key].clone())
            if state_key and state_key in initial_state:
                frame_buffers[state_key] = frame_deque(maxlen=frame_stack)
                for _ in range(frame_stack):
                    frame_buffers[state_key].append(initial_state[state_key].clone())

        def update_frame_buffers(current_state: dict[str, torch.Tensor]) -> None:
            """Add current state to frame buffers."""
            if not use_frame_stacking:
                return
            for key in image_keys_to_stack:
                if key in current_state:
                    frame_buffers[key].append(current_state[key].clone())
            if state_key and state_key in current_state:
                frame_buffers[state_key].append(current_state[state_key].clone())

        def get_stacked_state(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Get frame-stacked version of state."""
            if not use_frame_stacking:
                return raw_state
            stacked = {}
            for key, val in raw_state.items():
                if key in image_keys_to_stack and key in frame_buffers:
                    # Stack images along channel dimension: (1, C, H, W) x N -> (1, C*N, H, W)
                    frames = list(frame_buffers[key])
                    stacked[key] = torch.cat(frames, dim=1)  # dim=1 is channel dim with batch
                elif key == state_key and state_key in frame_buffers:
                    # Stack states along feature dimension: (1, D) x N -> (1, D*N)
                    frames = list(frame_buffers[key])
                    stacked[key] = torch.cat(frames, dim=1)  # dim=1 is feature dim with batch
                else:
                    stacked[key] = val
            return stacked

        # Stream through dataset - only keep current and next sample in memory
        prev_sample = None
        prev_state = None
        prev_stacked_state = None
        prev_episode_idx = None
        prev_ee_pos = None  # For action conversion
        episode_frame_count = 0

        for i in tqdm(range(num_frames), desc="Converting dataset to replay buffer"):
            current_sample = lerobot_dataset[i]
            current_episode_idx = current_sample["episode_index"].item()

            # Build current state
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                # Resize images if image_size is specified
                if image_size is not None and val.ndim == 3 and val.shape[0] in (1, 3, 4):
                    # Assume CHW format for images
                    val = F.interpolate(
                        val.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False
                    ).squeeze(0)

                # Compute full proprioception for observation.state
                if compute_full_proprioception and key == "observation.state" and mj_model is not None:
                    import mujoco
                    joint_pos = val.numpy()  # 6-dim joint positions (degrees)
                    num_dof = len(joint_pos)

                    # Convert to radians for MuJoCo
                    joint_pos_rad = np.deg2rad(joint_pos)

                    # Compute velocity (finite difference)
                    # Reset on episode boundary
                    is_new_episode = prev_episode_idx is not None and prev_episode_idx != current_episode_idx
                    if prev_joint_pos is None or is_new_episode:
                        joint_vel = np.zeros(num_dof)
                    else:
                        joint_vel = (joint_pos - prev_joint_pos) / dt
                    prev_joint_pos = joint_pos.copy()

                    # Compute FK using MuJoCo
                    mj_data.qpos[:num_dof] = joint_pos_rad
                    mujoco.mj_forward(mj_model, mj_data)
                    ee_xyz = mj_data.site_xpos[ee_site_id].copy()
                    xmat = mj_data.site_xmat[ee_site_id].reshape(3, 3)
                    ee_euler = rotation_matrix_to_euler(xmat)

                    # Concatenate into 18-dim full proprioceptive state
                    full_state = np.concatenate([joint_pos, joint_vel, ee_xyz, ee_euler])
                    val = torch.from_numpy(full_state).float()

                current_state[key] = val.unsqueeze(0).to(storage_device)

            # Handle frame stacking
            is_new_episode = prev_episode_idx is not None and prev_episode_idx != current_episode_idx
            if use_frame_stacking:
                if episode_frame_count == 0 or is_new_episode:
                    # New episode - reset buffers with current state
                    reset_frame_buffers(current_state)
                    episode_frame_count = 1
                else:
                    # Same episode - update buffers
                    update_frame_buffers(current_state)
                    episode_frame_count += 1
                current_stacked_state = get_stacked_state(current_state)
            else:
                current_stacked_state = current_state

            # Process previous sample now that we have current (for next_state)
            if prev_sample is not None:
                # Determine if previous frame was done
                if has_done_key:
                    done = bool(prev_sample["next.done"].item())
                else:
                    # Done if episode changed
                    done = prev_episode_idx != current_episode_idx

                # Get reward
                if has_reward_key:
                    reward = float(prev_sample["next.reward"].item())
                else:
                    reward = 0.0

                # next_state is current_stacked_state if same episode, else prev_stacked_state
                if done:
                    next_stacked_state = prev_stacked_state
                else:
                    next_stacked_state = current_stacked_state

                # Get action
                action = prev_sample["action"]

                # Convert joint actions to EE delta actions if requested
                if convert_actions_to_ee and mj_model is not None:
                    import mujoco
                    # Get joint positions from prev and current observation.state
                    prev_joints = prev_sample["observation.state"].numpy()[:6]
                    curr_joints = current_sample["observation.state"].numpy()[:6]

                    # Compute EE positions via FK
                    prev_joints_rad = np.deg2rad(prev_joints)
                    mj_data.qpos[:6] = prev_joints_rad
                    mujoco.mj_forward(mj_model, mj_data)
                    prev_ee = mj_data.site_xpos[ee_site_id].copy()

                    curr_joints_rad = np.deg2rad(curr_joints)
                    mj_data.qpos[:6] = curr_joints_rad
                    mujoco.mj_forward(mj_model, mj_data)
                    curr_ee = mj_data.site_xpos[ee_site_id].copy()

                    # EE delta (unnormalized)
                    ee_delta = curr_ee - prev_ee

                    # Normalize by action scale to get action in [-1, 1] range
                    ee_delta_normalized = ee_delta / ee_action_scale

                    # Clip to [-1, 1]
                    ee_delta_normalized = np.clip(ee_delta_normalized, -1.0, 1.0)

                    # Gripper action: compute from gripper joint position change
                    # Get gripper position from observation.state (last joint, index 5)
                    # If observation.state has 6 values, last is gripper; otherwise use action
                    if len(prev_joints) >= 6:
                        prev_gripper = prev_joints[5]  # Gripper joint position in degrees
                        curr_gripper = curr_joints[5]
                        gripper_delta = curr_gripper - prev_gripper
                        # Normalize: typical gripper range is 0-100 degrees, delta of ~10 deg = 0.2 action
                        gripper_action = np.clip(gripper_delta / 50.0, -1.0, 1.0)
                    else:
                        # Fallback: use original action's gripper value
                        gripper_val = action[-1].item() if action.numel() > 0 else 0.0
                        gripper_action = (gripper_val / 50.0) - 1.0
                    gripper_normalized = np.clip(gripper_action, -1.0, 1.0)

                    # Construct 4-dim EE action: [dx, dy, dz, gripper]
                    ee_action = np.array([
                        ee_delta_normalized[0],
                        ee_delta_normalized[1],
                        ee_delta_normalized[2],
                        gripper_normalized
                    ], dtype=np.float32)

                    action = torch.from_numpy(ee_action)

                action = action.unsqueeze(0).to(storage_device)

                # Get complementary info
                complementary_info = None
                if has_complementary_info:
                    complementary_info = {}
                    for key in complementary_info_keys:
                        short_key = key.replace("complementary_info.", "")
                        val = prev_sample[key]
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        complementary_info[short_key] = val.to(storage_device)

                # Initialize storage on first transition (using stacked states)
                if not replay_buffer.initialized:
                    init_state = {k: v.to(device) for k, v in prev_stacked_state.items()}
                    init_action = action.to(device)
                    init_comp_info = None
                    if complementary_info:
                        init_comp_info = {k: v.to(device) for k, v in complementary_info.items()}
                    replay_buffer._initialize_storage(
                        state=init_state, action=init_action, complementary_info=init_comp_info
                    )

                # Add transition (using stacked states)
                replay_buffer.add(
                    state=prev_stacked_state,
                    action=action,
                    reward=reward,
                    next_state=next_stacked_state,
                    done=done,
                    truncated=False,
                    complementary_info=complementary_info,
                )

            # Shift current to previous
            prev_sample = current_sample
            prev_state = current_state
            prev_stacked_state = current_stacked_state
            prev_episode_idx = current_episode_idx

        # Handle last frame (always done)
        if prev_sample is not None:
            if has_reward_key:
                reward = float(prev_sample["next.reward"].item())
            else:
                reward = 0.0

            action = prev_sample["action"]

            # Convert joint actions to EE delta actions for the last frame
            # Since there's no next sample, use zero EE delta
            if convert_actions_to_ee and mj_model is not None:
                # Gripper action: use gripper value from action
                gripper_val = action[-1].item()
                gripper_normalized = (gripper_val / 50.0) - 1.0  # 0->-1, 100->1
                gripper_normalized = np.clip(gripper_normalized, -1.0, 1.0)

                # Zero EE delta for last frame (no movement)
                ee_action = np.array([0.0, 0.0, 0.0, gripper_normalized], dtype=np.float32)
                action = torch.from_numpy(ee_action)

            action = action.unsqueeze(0).to(storage_device)

            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    short_key = key.replace("complementary_info.", "")
                    val = prev_sample[key]
                    if val.ndim == 0:
                        val = val.unsqueeze(0)
                    complementary_info[short_key] = val.to(storage_device)

            # Initialize if this is the only frame (using stacked state)
            if not replay_buffer.initialized:
                init_state = {k: v.to(device) for k, v in prev_stacked_state.items()}
                init_action = action.to(device)
                init_comp_info = None
                if complementary_info:
                    init_comp_info = {k: v.to(device) for k, v in complementary_info.items()}
                replay_buffer._initialize_storage(
                    state=init_state, action=init_action, complementary_info=init_comp_info
                )

            replay_buffer.add(
                state=prev_stacked_state,
                action=action,
                reward=reward,
                next_state=prev_stacked_state,  # Last frame's next_state is itself (stacked)
                done=True,
                truncated=False,
                complementary_info=complementary_info,
            )

        return replay_buffer

    def save(self, path: str) -> None:
        """
        Save the replay buffer to disk.

        Args:
            path: Path to save the buffer (without extension, will create .pt file)
        """
        if not self.initialized:
            raise RuntimeError("Cannot save an uninitialized buffer.")

        save_dict = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "truncateds": self.truncateds,
            "episode_ends": self.episode_ends,
            "position": self.position,
            "size": self.size,
            "capacity": self.capacity,
            "state_keys": list(self.state_keys),  # Convert to list for pickling
            "optimize_memory": self.optimize_memory,
            "has_complementary_info": self.has_complementary_info,
            "complementary_info_keys": self.complementary_info_keys,
            "complementary_info": self.complementary_info if self.has_complementary_info else {},
        }

        # Only save next_states if not using memory optimization
        if not self.optimize_memory:
            save_dict["next_states"] = self.next_states

        torch.save(save_dict, f"{path}.pt")

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cuda:0",
        storage_device: str = "cpu",
        image_augmentation_function=None,
        use_drq: bool = True,
    ) -> "ReplayBuffer":
        """
        Load a replay buffer from disk.

        Args:
            path: Path to the saved buffer (without .pt extension)
            device: Device for sampling tensors
            storage_device: Device for storage (ignored, uses saved device)
            image_augmentation_function: Optional augmentation function
            use_drq: Whether to use DrQ augmentation

        Returns:
            ReplayBuffer: Loaded replay buffer
        """
        save_dict = torch.load(f"{path}.pt", weights_only=False)

        # Create buffer with saved configuration
        replay_buffer = cls(
            capacity=save_dict["capacity"],
            device=device,
            state_keys=save_dict["state_keys"],
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=save_dict["optimize_memory"],
        )

        # Restore state
        replay_buffer.states = save_dict["states"]
        replay_buffer.actions = save_dict["actions"]
        replay_buffer.rewards = save_dict["rewards"]
        replay_buffer.dones = save_dict["dones"]
        replay_buffer.truncateds = save_dict["truncateds"]
        replay_buffer.episode_ends = save_dict["episode_ends"]
        replay_buffer.position = save_dict["position"]
        replay_buffer.size = save_dict["size"]
        replay_buffer.has_complementary_info = save_dict["has_complementary_info"]
        replay_buffer.complementary_info_keys = save_dict["complementary_info_keys"]
        replay_buffer.complementary_info = save_dict["complementary_info"]

        if not save_dict["optimize_memory"]:
            replay_buffer.next_states = save_dict["next_states"]
        else:
            replay_buffer.next_states = replay_buffer.states

        replay_buffer.initialized = True

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object.
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        sample_action = self.actions[0]
        act_info = guess_feature_info(t=sample_action, name="action")
        features["action"] = act_info

        # Add "reward" and "done"
        features["next.reward"] = {"dtype": "float32", "shape": (1,)}
        features["next.done"] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # Add complementary_info keys if available
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                sample_val = self.complementary_info[key][0]
                if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
                    sample_val = sample_val.unsqueeze(0)
                f_info = guess_feature_info(t=sample_val, name=f"complementary_info.{key}")
                features[f"complementary_info.{key}"] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # Start writing images if needed
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # Convert transitions into episodes and frames
        episode_index = 0
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(episode_index=episode_index)

        frame_idx_in_episode = 0
        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            frame_dict = {}

            # Fill the data for state keys
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # Fill action, reward, done
            frame_dict["action"] = self.actions[actual_idx].cpu()
            frame_dict["next.reward"] = torch.tensor([self.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict["next.done"] = torch.tensor([self.dones[actual_idx]], dtype=torch.bool).cpu()

            # Add complementary_info if available
            if self.has_complementary_info:
                for key in self.complementary_info_keys:
                    val = self.complementary_info[key][actual_idx]
                    # Convert tensors to CPU
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        frame_dict[f"complementary_info.{key}"] = val.cpu()
                    # Non-tensor values can be used directly
                    else:
                        frame_dict[f"complementary_info.{key}"] = val

            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict, task=task_name)

            # Move to next frame
            frame_idx_in_episode += 1

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode()
                episode_index += 1
                frame_idx_in_episode = 0
                lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(
                    episode_index=episode_index
                )

        # Save any remaining frames in the buffer
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode()

        lerobot_dataset.stop_image_writer()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
    ) -> list[Transition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.

        Args:
            dataset (LeRobotDataset):
                The dataset to convert. Each item in the dataset is expected to have
                at least the following keys:
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                plus whatever your 'state_keys' specify.

            state_keys (Sequence[str] | None):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        transitions = []
        num_frames = len(dataset)

        # Check if the dataset has "next.done" and "next.reward" keys
        sample = dataset[0]
        has_done_key = "next.done" in sample
        has_reward_key = "next.reward" in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")
        if not has_reward_key:
            print("'next.reward' key not found in dataset. Using 0.0 as default reward...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample["action"].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            # Use next.reward if available, otherwise default to 0.0
            if has_reward_key:
                reward = float(current_sample["next.reward"].item())  # ensure float
            else:
                reward = 0.0  # Default reward for teleoperation datasets without reward annotations

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample["next.done"].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # TODO: (azouitine) Check if it's necessary to convert to tensor
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            transitions.append(transition)

        return transitions


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or scalar value.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to appropriate dtype for numeric.
    """

    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """
    Concatenates two BatchTransition objects into one.

    This function merges the right BatchTransition into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransition): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransition): The second batch to append to the first one.

    Returns:
        BatchTransition: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # Concatenate basic fields
    left_batch_transitions["action"] = torch.cat(
        [left_batch_transitions["action"], right_batch_transition["action"]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # Only process if right_info exists
    if right_info is not None:
        # Initialize left complementary_info if needed
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # Concatenate each field
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions
