"""Simple script to load and iterate through a LeRobot dataset.

Example usage:
    python examples/use_lerobot_dataset.py --repo_id lerobot/aloha_sim_transfer_cube_human
    python examples/use_lerobot_dataset.py --repo_id your_hf_username/your_dataset --local_files_only
"""

import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Load and inspect a LeRobot dataset")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobot dataset repo ID (e.g., 'lerobot/aloha_sim_transfer_cube_human' or local path)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=3,
        help="Number of batches to iterate through",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=10,
        help="Action horizon (number of future actions to predict)",
    )
    parser.add_argument(
        "--skip_norm_stats",
        action="store_true",
        help="Skip normalization (useful if norm stats not available)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Use an existing config name instead of creating a simple one",
    )
    args = parser.parse_args()

    # Create a simple model config
    model_config = pi0_config.Pi0Config(
        action_dim=7,  # Adjust based on your dataset
        action_horizon=args.action_horizon,
        max_token_len=48,
    )

    # Option 1: Use an existing config from config.py
    if args.config_name:
        config = _config.get_config(args.config_name)
        config = config.replace(batch_size=args.batch_size)
        if config.data.repo_id != args.repo_id:
            logging.warning(
                f"Config repo_id ({config.data.repo_id}) differs from provided ({args.repo_id}). "
                f"Using provided repo_id."
            )
            # Create a new data config with the provided repo_id
            config = config.replace(
                data=config.data.replace(repo_id=args.repo_id)
            )
    else:
        # Option 2: Create a simple config manually
        # You'll need to adjust the DataConfig based on your dataset structure
        data_config_factory = _config.SimpleDataConfig(
            repo_id=args.repo_id,
            base_config=_config.DataConfig(
                repo_id=args.repo_id,
                action_sequence_keys=("action",),  # Adjust based on your dataset
                prompt_from_task=False,  # Set to True if your dataset has task labels
            ),
        )

        config = _config.TrainConfig(
            name="test_lerobot",
            exp_name="test",
            model=model_config,
            data=data_config_factory,
            batch_size=args.batch_size,
            assets_base_dir=str(Path.home() / ".cache" / "openpi" / "assets"),
            checkpoint_base_dir=str(Path.home() / ".cache" / "openpi" / "checkpoints"),
        )

    logging.info(f"Loading dataset: {args.repo_id}")
    logging.info(f"Model config: action_dim={model_config.action_dim}, action_horizon={model_config.action_horizon}")

    # Create data loader
    try:
        data_loader = _data_loader.create_data_loader(
            config,
            shuffle=False,
            num_batches=args.num_batches,
            skip_norm_stats=args.skip_norm_stats,
        )
        logging.info("Data loader created successfully!")
    except Exception as e:
        logging.error(f"Failed to create data loader: {e}")
        logging.error(
            "\nTroubleshooting tips:\n"
            "1. Make sure your dataset is in LeRobot format\n"
            "2. Check that the repo_id is correct\n"
            "3. If using a local dataset, ensure it's in the correct location\n"
            "4. You may need to create a custom DataConfigFactory in config.py\n"
            "   See examples like LeRobotAlohaDataConfig or LeRobotLiberoDataConfig"
        )
        raise

    # Get data config info
    data_cfg = data_loader.data_config()
    logging.info(f"Data config repo_id: {data_cfg.repo_id}")
    logging.info(f"Action sequence keys: {data_cfg.action_sequence_keys}")

    # Iterate through batches
    logging.info(f"\nIterating through {args.num_batches} batches...")
    for i, (observation, actions) in enumerate(data_loader):
        logging.info(f"\n--- Batch {i+1} ---")
        logging.info(f"Observation keys: {list(observation.to_dict().keys())}")
        
        # Print observation shapes
        for key, value in observation.to_dict().items():
            if isinstance(value, dict):
                logging.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'shape'):
                        logging.info(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
            elif hasattr(value, 'shape'):
                logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Print action info
        logging.info(f"Actions: shape={actions.shape}, dtype={actions.dtype}")
        logging.info(f"  Action range: [{jnp.min(actions):.3f}, {jnp.max(actions):.3f}]")
        logging.info(f"  Action mean: {jnp.mean(actions):.3f}, std: {jnp.std(actions):.3f}")

        if i >= args.num_batches - 1:
            break

    logging.info("\n✓ Successfully loaded and iterated through the dataset!")


if __name__ == "__main__":
    main()
