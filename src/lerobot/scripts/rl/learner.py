# !/usr/bin/env python

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
"""
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.scripts.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.drqv2.modeling_drqv2 import DrQV2Policy
from lerobot.robots import so100_follower, so101_follower  # noqa: F401
from lerobot.scripts.rl import learner_service
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

LOG_PREFIX = "[LEARNER]"


#################################################
# MAIN ENTRY POINTS AND CORE ALGORITHM FUNCTIONS #
#################################################


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Save config to output dir for actor to use
    config_save_path = os.path.join(cfg.output_dir, "train_config.json")
    if not os.path.exists(config_save_path):
        import json
        with open(config_save_path, "w") as f:
            json.dump(cfg.to_dict(), f, indent=4, default=str)

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.utils.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


#################################################
# Core algorithm functions #
#################################################


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

    # Log checkpoint config at startup
    logging.info(f"[LEARNER] Checkpoint config: save_checkpoint={saving_checkpoint}, save_freq={save_freq}, output_dir={cfg.output_dir}")

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    # Load pretrained weights if specified (e.g., Genesis DrQ-v2 checkpoint)
    # Check config object first, then fall back to reading from saved JSON config
    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    if pretrained_path is None:
        # Config parser may not preserve pretrained_path, read from saved JSON
        config_json_path = os.path.join(cfg.output_dir, "train_config.json")
        if os.path.exists(config_json_path):
            import json
            with open(config_json_path) as f:
                saved_cfg = json.load(f)
            pretrained_path = saved_cfg.get("policy", {}).get("pretrained_path")

    if pretrained_path:
        logging.info(f"[LEARNER] Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

        # Check if we should skip loading critic weights (for sim-to-real transfer)
        skip_pretrained_critic = getattr(cfg.policy, "skip_pretrained_critic", False)
        if skip_pretrained_critic:
            logging.info("[LEARNER] Skipping pretrained critic weights (training from scratch)")

        if "agent" in checkpoint:
            # RoboBase/Genesis format
            agent_state = checkpoint["agent"]
            if skip_pretrained_critic:
                # Filter out critic weights - they don't transfer well from sim to real
                agent_state = {k: v for k, v in agent_state.items()
                              if not k.startswith(("critic.", "critic_target."))}
                logging.info(f"[LEARNER] Filtered to {len(agent_state)} weights (excluding critic)")

            if hasattr(policy, "_load_robobase_weights"):
                policy._load_robobase_weights(agent_state)
                logging.info("[LEARNER] Loaded RoboBase pretrained weights")
            else:
                logging.warning("[LEARNER] Policy does not support RoboBase weight loading")
        else:
            # Direct state dict format
            state_dict = checkpoint
            if skip_pretrained_critic:
                state_dict = {k: v for k, v in state_dict.items()
                             if not k.startswith(("critic.", "critic_target."))}
            policy.load_state_dict(state_dict, strict=False)
            logging.info("[LEARNER] Loaded pretrained state dict")

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # Initialize iterators
    online_iterator = None
    offline_iterator = None

    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    while True:
        # Exit the training loop if shutdown is requested
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # Process all available transitions to the replay buffer, send by the actor server
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # Process all available interaction messages sent by the actor server
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # Wait until the replay buffer has enough samples to start training
        if len(replay_buffer) < online_step_before_learning:
            # Log waiting status periodically
            if optimization_step == 0 and not hasattr(add_actor_information_and_train, '_waiting_logged'):
                logging.info(
                    f"[LEARNER] Waiting for actor to connect and send {online_step_before_learning} transitions... "
                    f"(current: {len(replay_buffer)}/{online_step_before_learning})"
                )
                add_actor_information_and_train._waiting_logged = True
            # Wait with timeout to avoid busy-waiting while respecting shutdown
            if shutdown_event is not None:
                shutdown_event.wait(timeout=0.1)
            else:
                time.sleep(0.1)
            continue

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            # Sample from the iterators
            batch = next(online_iterator)

            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator)
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            actions = batch["action"]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = {
                "action": actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "complementary_info": batch["complementary_info"],
            }

            # Use the forward method for critic loss
            critic_output = policy.forward(forward_batch, model="critic")

            # Main critic optimization
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            # For DrQ-v2, also zero encoder gradients (encoder trained through critic loss)
            if "encoder" in optimizers:
                optimizers["encoder"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()
            # For DrQ-v2, step encoder optimizer after critic
            if "encoder" in optimizers:
                torch.nn.utils.clip_grad_norm_(
                    parameters=policy.encoder.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["encoder"].step()

            # Discrete critic optimization (if available)
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["discrete_critic"].step()

            # Update target networks (main and discrete)
            policy.update_target_networks()

        # Sample for the last update in the UTD ratio
        batch = next(online_iterator)

        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            "action": actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }

        critic_output = policy.forward(forward_batch, model="critic")

        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        # For DrQ-v2, also zero encoder gradients (encoder trained through critic loss)
        if "encoder" in optimizers:
            optimizers["encoder"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()
        # For DrQ-v2, step encoder optimizer after critic
        if "encoder" in optimizers:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.encoder.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["encoder"].step()

        # Initialize training info dictionary
        training_infos = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
        }
        # Add encoder grad norm for DrQ-v2
        if "encoder" in optimizers:
            training_infos["encoder_grad_norm"] = encoder_grad_norm

        # Discrete critic optimization (if available)
        if policy.config.num_discrete_actions is not None:
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()

            # Add discrete critic info to training info
            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

        # Actor and temperature optimization (at specified frequency)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor optimization
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # Add actor info to training info
                training_infos["loss_actor"] = loss_actor.item()
                training_infos["actor_grad_norm"] = actor_grad_norm

                # Temperature optimization (SAC only - DrQ-v2 uses scheduled noise)
                if "temperature" in optimizers:
                    temperature_output = policy.forward(forward_batch, model="temperature")
                    loss_temperature = temperature_output["loss_temperature"]
                    optimizers["temperature"].zero_grad()
                    loss_temperature.backward()
                    temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                    ).item()
                    optimizers["temperature"].step()

                    # Add temperature info to training info
                    training_infos["loss_temperature"] = loss_temperature.item()
                    training_infos["temperature_grad_norm"] = temp_grad_norm
                    training_infos["temperature"] = policy.temperature

                    # Update temperature
                    policy.update_temperature()

        # Push policy to actors if needed
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        # Update target networks (main and discrete)
        policy.update_target_networks()

        # Log training metrics at specified intervals
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            # Log training metrics to console
            loss_str = f"critic={training_infos.get('loss_critic', 0):.4f}"
            if "loss_actor" in training_infos:
                loss_str += f" actor={training_infos['loss_actor']:.4f}"
            if "loss_temperature" in training_infos:
                loss_str += f" temp={training_infos['loss_temperature']:.4f}"
            if "temperature" in training_infos:
                loss_str += f" α={training_infos['temperature']:.4f}"
            logging.info(
                f"[LEARNER] Step {optimization_step}: {loss_str} "
                f"buffer={training_infos['replay_buffer_size']}"
            )

            # Log training metrics to wandb
            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        # Calculate optimization frequency (only log periodically to reduce spam)
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        # Log optimization frequency
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # Save checkpoint at specified intervals
        should_save = optimization_step % save_freq == 0 or optimization_step == online_steps
        if should_save:
            logging.info(f"[LEARNER] Checkpoint check: step={optimization_step}, save_freq={save_freq}, saving_checkpoint={saving_checkpoint}")
            if saving_checkpoint:
                try:
                    save_training_checkpoint(
                        cfg=cfg,
                        optimization_step=optimization_step,
                        online_steps=online_steps,
                        interaction_message=interaction_message,
                        policy=policy,
                        optimizers=optimizers,
                        replay_buffer=replay_buffer,
                        offline_replay_buffer=offline_replay_buffer,
                        dataset_repo_id=dataset_repo_id,
                        fps=fps,
                    )
                    logging.info(f"[LEARNER] Checkpoint saved at step {optimization_step}")
                except Exception as e:
                    logging.error(f"[LEARNER] Failed to save checkpoint at step {optimization_step}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    service = learner_service.LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(learner_service.SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    """
    Save training checkpoint and associated data.

    This function performs the following steps:
    1. Creates a checkpoint directory with the current optimization step
    2. Saves the policy model, configuration, and optimizer states
    3. Saves the current interaction step for resuming training
    4. Updates the "last" checkpoint symlink to point to this checkpoint
    5. Saves the replay buffer as a dataset for later use
    6. If an offline replay buffer exists, saves it as a separate dataset

    Args:
        cfg: Training configuration
        optimization_step: Current optimization step
        online_steps: Total number of online steps
        interaction_message: Dictionary containing interaction information
        policy: Policy model to save
        optimizers: Dictionary of optimizers
        replay_buffer: Replay buffer to save as dataset
        offline_replay_buffer: Optional offline replay buffer to save
        dataset_repo_id: Repository ID for dataset
        fps: Frames per second for dataset
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # Create checkpoint directory
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # Save checkpoint
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save interaction step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)

    # TODO : temporary save replay buffer here, remove later when on the robot
    # We want to control this with the keyboard inputs
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Save dataset
    # NOTE: Handle the case where the dataset repo id is not specified in the config
    # eg. RL training without demonstrations data
    # NOTE: Frame stacking produces multi-channel images that can't be saved as standard LeRobotDataset
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    try:
        replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)
    except ValueError as e:
        if "Shape of" in str(e):
            logging.warning(f"[LEARNER] Skipping buffer-to-dataset conversion (frame stacking incompatible): {e}")
        else:
            raise

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        try:
            offline_replay_buffer.to_lerobot_dataset(
                cfg.dataset.repo_id,
                fps=fps,
                root=dataset_offline_dir,
            )
        except ValueError as e:
            if "Shape of" in str(e):
                logging.warning(f"[LEARNER] Skipping offline buffer-to-dataset conversion (frame stacking incompatible): {e}")
            else:
                raise

    logging.info("Resume training")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.
    - For DrQ-v2: The **encoder**, which is trained through the critic loss.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.
    - For DrQ-v2, the encoder has a separate optimizer since it's trained through the critic loss.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature", optionally "encoder") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    # Check if this is a DrQ-v2 policy (has encoder that needs separate optimization)
    is_drqv2 = isinstance(policy, DrQV2Policy)

    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
    }
    # SAC uses learned temperature, DrQ-v2 uses scheduled noise
    if hasattr(policy, "log_alpha"):
        optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
        optimizers["temperature"] = optimizer_temperature
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic

    # For DrQ-v2, add encoder optimizer (encoder is trained through critic loss)
    if is_drqv2:
        encoder_lr = getattr(cfg.policy, "encoder_lr", cfg.policy.critic_lr)
        optimizers["encoder"] = torch.optim.Adam(params=policy.encoder.parameters(), lr=encoder_lr)
        logging.info(f"[LEARNER] DrQ-v2 detected: added encoder optimizer with lr={encoder_lr}")

    return optimizers, lr_scheduler


#################################################
# Training setup functions #
#################################################


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration

    Returns:
        TrainRLServerPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        # No checkpoint exists yet - this is fine, just start fresh but keep resume=true
        # so the output dir isn't treated as an error and actor can connect
        logging.info(
            colored(
                "No checkpoint found but resume=True: starting fresh (output dir already exists)",
                color="yellow",
                attrs=["bold"],
            )
        )
        return cfg

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Preserve certain config values from the current config (not checkpoint)
    # This allows changing save_freq, log_freq, etc. without recreating checkpoints
    checkpoint_cfg.save_freq = cfg.save_freq
    checkpoint_cfg.log_freq = cfg.log_freq
    checkpoint_cfg.save_checkpoint = cfg.save_checkpoint

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    """
    if not cfg.resume:
        return None, None

    # Construct path to the last checkpoint directory
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # Use the utility function from train_utils which loads the optimizer state
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # Load interaction step separately from training_state.pt
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    # Check if there's a saved online dataset to resume from
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    if cfg.resume and os.path.exists(dataset_path):
        logging.info("Resume training: loading online dataset from checkpoint")
        # NOTE: In RL is possible to not have a dataset.
        repo_id = None
        if cfg.dataset is not None:
            repo_id = cfg.dataset.repo_id
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
            video_backend=cfg.dataset.video_backend if cfg.dataset else "pyav",
        )
        return ReplayBuffer.from_lerobot_dataset(
            lerobot_dataset=dataset,
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            optimize_memory=True,
        )

    # Start with empty buffer (either fresh start or resume without saved dataset)
    if cfg.resume:
        logging.info("Resume training: no saved online dataset found, starting with empty buffer")
    return ReplayBuffer(
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset.

    Uses caching to avoid re-converting the dataset on every run.
    The cache is stored at {output_dir}/offline_buffer.pt

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    # Check for cached buffer
    cache_path = os.path.join(cfg.output_dir, "offline_buffer")
    cache_file = f"{cache_path}.pt"

    if os.path.exists(cache_file):
        logging.info(f"Loading cached offline replay buffer from {cache_file}")
        offline_replay_buffer = ReplayBuffer.load(
            cache_path,
            device=device,
            storage_device=storage_device,
        )
        logging.info(f"Loaded {len(offline_replay_buffer)} transitions from cache")
        return offline_replay_buffer

    # No cache - need to convert from dataset
    # Check if local dataset exists (for resume), otherwise download from hub
    dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
    if cfg.resume and os.path.exists(dataset_offline_path):
        logging.info("load offline dataset from local path")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
            video_backend=cfg.dataset.video_backend,
        )
    else:
        logging.info("make_dataset offline buffer (downloading from hub)")
        offline_dataset = make_dataset(cfg)

    logging.info("Convert to a offline replay buffer")
    # Get image resize size from env config if available
    image_size = None
    if hasattr(cfg, "env") and hasattr(cfg.env, "wrapper") and hasattr(cfg.env.wrapper, "resize_size"):
        resize_size = cfg.env.wrapper.resize_size
        if resize_size is not None:
            image_size = tuple(resize_size)
            logging.info(f"Resizing images to {image_size} during buffer conversion")

    # Get full proprioception config
    compute_full_proprioception = False
    convert_to_radians = False
    unnormalize_images = False
    mujoco_model_path = None
    ee_site_name = "gripper"
    fps = 30.0

    if hasattr(cfg, "env") and hasattr(cfg.env, "wrapper"):
        compute_full_proprioception = getattr(cfg.env.wrapper, "add_full_proprioception", False)
        convert_to_radians = getattr(cfg.env.wrapper, "use_radians", False)
        # When normalize_images=False in live env, we need to unnormalize dataset images
        # (dataset stores images in [0,1] but encoder expects [0,255])
        normalize_images = getattr(cfg.env.wrapper, "normalize_images", True)
        unnormalize_images = not normalize_images
        if unnormalize_images:
            logging.info("Unnormalizing dataset images from [0,1] to [0,255] (normalize_images=False)")

    # Get MuJoCo model path from robot config (needed for full proprioception and action conversion)
    if hasattr(cfg, "env") and hasattr(cfg.env, "robot"):
        robot_cfg = cfg.env.robot
        # Try both direct access and nested config
        mujoco_model_path = getattr(robot_cfg, "mujoco_model_path", None)
        if mujoco_model_path is None and hasattr(robot_cfg, "config"):
            mujoco_model_path = getattr(robot_cfg.config, "mujoco_model_path", None)
        ee_site_name = getattr(robot_cfg, "end_effector_site", None)
        if ee_site_name is None and hasattr(robot_cfg, "config"):
            ee_site_name = getattr(robot_cfg.config, "end_effector_site", "gripper")
        if ee_site_name is None:
            ee_site_name = "gripper"

    if hasattr(cfg, "env"):
        fps = getattr(cfg.env, "fps", 30.0)

    if compute_full_proprioception:
        logging.info(f"Computing full proprioception with MuJoCo FK (model: {mujoco_model_path}, site: {ee_site_name})")
        if convert_to_radians:
            logging.info("Converting joint positions/velocities to radians (for RoboBase/Genesis pretrained models)")

    # Check if action conversion is needed (dataset has joint actions but policy expects EE actions)
    convert_actions_to_ee = False
    ee_action_scale = 0.02
    target_action_dim = None

    # Get policy action dim
    policy_action_dim = None
    if hasattr(cfg.policy, "output_features") and "action" in cfg.policy.output_features:
        action_feature = cfg.policy.output_features["action"]
        # Handle both dict and PolicyFeature object
        if hasattr(action_feature, "shape"):
            policy_action_shape = action_feature.shape
        elif isinstance(action_feature, dict):
            policy_action_shape = action_feature.get("shape", None)
        else:
            policy_action_shape = None
        if policy_action_shape:
            policy_action_dim = policy_action_shape[0] if isinstance(policy_action_shape, (list, tuple)) else policy_action_shape

    # Get dataset action dim and check if actions are valid
    dataset_action_dim = offline_dataset[0]["action"].shape[0] if len(offline_dataset) > 0 else None

    # Check if dataset actions are all zeros (corrupted/missing actions)
    # Only check XYZ components (first 3), as gripper (4th) may be non-zero
    actions_are_zero = False
    if len(offline_dataset) > 0:
        import torch
        sample_actions = torch.stack([offline_dataset[i]["action"] for i in range(min(100, len(offline_dataset)))])
        # Check only XYZ components (first 3 dimensions) - gripper may have valid non-zero values
        xyz_actions = sample_actions[:, :3] if sample_actions.shape[1] >= 3 else sample_actions
        if xyz_actions.abs().max() < 1e-6:
            actions_are_zero = True
            logging.warning("[LEARNER] Dataset XYZ actions are all zeros - will compute from state changes")

    if policy_action_dim and dataset_action_dim and policy_action_dim != dataset_action_dim:
        logging.info(f"Action dimension mismatch: dataset={dataset_action_dim}, policy={policy_action_dim}")
        # If policy expects 4-dim (EE) and dataset has 6-dim (joints), convert
        if policy_action_dim == 4 and dataset_action_dim == 6:
            convert_actions_to_ee = True
            target_action_dim = policy_action_dim
            # Get action scale from robot config
            if hasattr(cfg.env, "robot"):
                ee_action_scale = getattr(cfg.env.robot, "action_scale", 0.02)
            logging.info(f"Converting joint actions to EE actions (scale: {ee_action_scale})")

    # Also convert if actions are zeros (need to compute from FK)
    if actions_are_zero and mujoco_model_path is not None:
        convert_actions_to_ee = True
        target_action_dim = policy_action_dim if policy_action_dim else 4
        if hasattr(cfg.env, "robot"):
            ee_action_scale = getattr(cfg.env.robot, "action_scale", 0.02)
        logging.info(f"Computing actions from state changes via FK (scale: {ee_action_scale})")

    # Get frame_stack from policy config (for DrQ-v2 and similar policies)
    frame_stack = getattr(cfg.policy, "frame_stack", 1)
    if frame_stack > 1:
        logging.info(f"Frame stacking enabled for offline buffer: {frame_stack} frames")

    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        image_size=image_size,
        compute_full_proprioception=compute_full_proprioception,
        convert_to_radians=convert_to_radians,
        mujoco_model_path=mujoco_model_path,
        ee_site_name=ee_site_name,
        fps=fps,
        convert_actions_to_ee=convert_actions_to_ee,
        ee_action_scale=ee_action_scale,
        target_action_dim=target_action_dim,
        frame_stack=frame_stack,
        unnormalize_images=unnormalize_images,
    )

    # Save to cache for future runs
    logging.info(f"Saving offline replay buffer cache to {cache_file}")
    offline_replay_buffer.save(cache_path)
    logging.info("Cache saved successfully")

    return offline_replay_buffer


#################################################
# Utilities/Helpers functions #
#################################################


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations, normalize=True)
        next_observation_features = policy.actor.encoder.get_cached_image_features(
            next_observations, normalize=True
        )

    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # Add encoder if it exists (needed for DrQ-v2 which trains encoder end-to-end)
    if hasattr(policy, "encoder") and policy.encoder is not None:
        state_dicts["encoder"] = move_state_dict_to_device(
            policy.encoder.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including encoder in state dict push")

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log episode info
    ep_num = message.get("Episode number", "?")
    ep_reward = message.get("Episodic reward", 0)
    int_step = message.get("Interaction step", 0)
    int_rate = message.get("Intervention rate", 0)
    logging.info(f"[LEARNER] Episode {ep_num} | step={int_step} | reward={ep_reward:.2f} | intervention={int_rate:.1%}")

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition["action"],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition)

            # Add to offline buffer if it's an intervention
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                "is_intervention"
            ):
                offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
