#!/usr/bin/env python
"""Convert RoboBase DrQ-v2 checkpoint to LeRobot format.

Usage:
    python scripts/convert_robobase_checkpoint.py \
        --input /path/to/robobase/checkpoint.pt \
        --output /path/to/lerobot/checkpoint \
        --device cuda
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def map_robobase_key_to_lerobot(key: str) -> str | None:
    """Map RoboBase state dict key to LeRobot key.

    RoboBase structure:
        encoder.convs_per_cam.0.* -> encoder.convs_per_cam.0.*
        actor.actor_model.* -> actor.actor_model.*
        actor_model.* -> actor.actor_model.* (duplicate in some checkpoints)
        critic.qs.0.* -> critic.qs.0.*
        critic_target.qs.0.* -> critic_target.qs.0.*
    """
    # Skip hidden states (RNN buffers, not weights)
    if "hidden_state" in key:
        return None

    # Encoder - same structure
    if key.startswith("encoder."):
        return key

    # Actor - handle both prefixes
    if key.startswith("actor.actor_model."):
        return key
    if key.startswith("actor_model."):
        return "actor." + key

    # Critic - same structure
    if key.startswith("critic.qs."):
        return key
    if key.startswith("critic_target.qs."):
        return key

    # View fusion (if present)
    if key.startswith("view_fusion."):
        return key

    return None


def convert_checkpoint(
    input_path: str,
    output_path: str,
    device: str = "cpu",
) -> dict:
    """Convert RoboBase checkpoint to LeRobot format.

    Args:
        input_path: Path to RoboBase .pt checkpoint
        output_path: Path to save LeRobot checkpoint (directory)
        device: Device to load checkpoint on

    Returns:
        Dictionary with conversion stats
    """
    print(f"Loading RoboBase checkpoint from {input_path}")
    ckpt = torch.load(input_path, map_location=device, weights_only=False)

    agent_state = ckpt["agent"]
    cfg = ckpt.get("cfg", {})

    # Extract config info
    method_cfg = cfg.get("method", {}) if hasattr(cfg, "get") else {}
    encoder_cfg = method_cfg.get("encoder_model", {}) if method_cfg else {}
    actor_cfg = method_cfg.get("actor_model", {}) if method_cfg else {}

    print(f"Checkpoint has {len(agent_state)} keys")

    # Map weights
    new_state_dict = {}
    skipped_keys = []
    mapped_keys = []

    for key, value in agent_state.items():
        new_key = map_robobase_key_to_lerobot(key)
        if new_key is not None:
            new_state_dict[new_key] = value
            mapped_keys.append((key, new_key))
        else:
            skipped_keys.append(key)

    print(f"Mapped {len(mapped_keys)} keys, skipped {len(skipped_keys)} keys")

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as safetensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(new_state_dict, safetensors_path)
    print(f"Saved weights to {safetensors_path}")

    # Helper to convert OmegaConf types to native Python
    def to_native(val):
        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
            return list(val)
        return val

    # Save config as JSON
    config_dict = {
        "type": "drqv2",
        "num_downsample_convs": to_native(encoder_cfg.get("num_downsample_convs", 1)) if encoder_cfg else 1,
        "num_post_downsample_convs": to_native(encoder_cfg.get("num_post_downsample_convs", 3)) if encoder_cfg else 3,
        "encoder_channels": to_native(encoder_cfg.get("channels", 32)) if encoder_cfg else 32,
        "encoder_kernel_size": to_native(encoder_cfg.get("kernel_size", 3)) if encoder_cfg else 3,
        "bottleneck_size": to_native(actor_cfg.get("bottleneck_size", 50)) if actor_cfg else 50,
        "norm_after_bottleneck": to_native(actor_cfg.get("norm_after_bottleneck", True)) if actor_cfg else True,
        "tanh_after_bottleneck": to_native(actor_cfg.get("tanh_after_bottleneck", True)) if actor_cfg else True,
        "mlp_nodes": to_native(actor_cfg.get("mlp_nodes", [256, 256])) if actor_cfg else [256, 256],
        "num_critics": to_native(method_cfg.get("num_critics", 2)) if method_cfg else 2,
        "stddev_schedule": to_native(method_cfg.get("stddev_schedule", "linear(1.0,0.1,500000)")) if method_cfg else "linear(1.0,0.1,500000)",
        "stddev_clip": to_native(method_cfg.get("stddev_clip", 0.3)) if method_cfg else 0.3,
        "use_augmentation": to_native(method_cfg.get("use_augmentation", True)) if method_cfg else True,
        "discount": to_native(method_cfg.get("discount", 0.99)) if method_cfg else 0.99,
        "critic_target_tau": to_native(method_cfg.get("critic_target_tau", 0.01)) if method_cfg else 0.01,
    }

    import json
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")

    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Mapped keys: {len(mapped_keys)}")
    print(f"Skipped keys: {len(skipped_keys)}")

    if skipped_keys:
        print("\nSkipped keys (hidden states, etc.):")
        for k in skipped_keys[:10]:
            print(f"  - {k}")
        if len(skipped_keys) > 10:
            print(f"  ... and {len(skipped_keys) - 10} more")

    return {
        "mapped_keys": len(mapped_keys),
        "skipped_keys": len(skipped_keys),
        "config": config_dict,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboBase DrQ-v2 checkpoint to LeRobot format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to RoboBase .pt checkpoint",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for LeRobot checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load checkpoint on (default: cpu)",
    )

    args = parser.parse_args()
    convert_checkpoint(args.input, args.output, args.device)


if __name__ == "__main__":
    main()
