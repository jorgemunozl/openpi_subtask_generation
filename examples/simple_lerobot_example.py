"""
Minimal example to load a LeRobot dataset directly.

This is a simpler example that shows how to use the dataset loading functions
without going through the full training config system.
"""

import logging

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logging.basicConfig(level=logging.INFO)


def simple_example():
    """Simple example loading a LeRobot dataset."""
    # Dataset configuration
    repo_id = "lerobot/aloha_sim_transfer_cube_human"  # Replace with your dataset
    action_horizon = 10
    
    # Model config (needed to determine input/output shapes)
    model_config = pi0_config.Pi0Config(
        action_dim=7,  # Adjust based on your dataset
        action_horizon=action_horizon,
        max_token_len=48,
    )
    
    # Create a simple data config
    data_config = _config.DataConfig(
        repo_id=repo_id,
        action_sequence_keys=("action",),  # Key name in your dataset
        prompt_from_task=False,  # Set True if dataset has task labels
    )
    
    logging.info(f"Loading dataset: {repo_id}")
    
    # Create the dataset (without transforms for simplicity)
    try:
        dataset = _data_loader.create_torch_dataset(
            data_config=data_config,
            action_horizon=action_horizon,
            model_config=model_config,
        )
        logging.info(f"Dataset created! Length: {len(dataset)}")
    except Exception as e:
        logging.error(f"Failed to create dataset: {e}")
        raise
    
    # Get a few samples
    logging.info("\nInspecting samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        logging.info(f"\nSample {i}:")
        logging.info(f"  Keys: {list(sample.keys())}")
        
        # Print shapes
        for key, value in sample.items():
            if isinstance(value, dict):
                logging.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'shape'):
                        logging.info(f"    {sub_key}: {sub_value.shape}")
            elif hasattr(value, 'shape'):
                logging.info(f"  {key}: {value.shape}")
            elif isinstance(value, (str, int, float)):
                logging.info(f"  {key}: {value}")


def with_transforms_example():
    """Example with data transforms applied."""
    repo_id = "lerobot/aloha_sim_transfer_cube_human"
    action_horizon = 10
    
    model_config = pi0_config.Pi0Config(
        action_dim=7,
        action_horizon=action_horizon,
        max_token_len=48,
    )
    
    # Use a pre-configured data config factory (like Aloha)
    data_config_factory = _config.LeRobotAlohaDataConfig(
        repo_id=repo_id,
        default_prompt="Transfer cube",
    )
    
    # Create the full config (needed for transforms)
    config = _config.TrainConfig(
        name="test",
        exp_name="test",
        model=model_config,
        data=data_config_factory,
        batch_size=4,
    )
    
    logging.info(f"Loading dataset with transforms: {repo_id}")
    
    # Create data loader (this applies all transforms)
    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=2,
        skip_norm_stats=True,  # Skip if norm stats not available
    )
    
    # Iterate through batches
    logging.info("\nIterating through batches:")
    for i, (observation, actions) in enumerate(data_loader):
        logging.info(f"\nBatch {i+1}:")
        logging.info(f"  Observation keys: {list(observation.to_dict().keys())}")
        logging.info(f"  Actions shape: {actions.shape}")
        logging.info(f"  Actions range: [{jnp.min(actions):.3f}, {jnp.max(actions):.3f}]")


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Example (no transforms)")
    print("=" * 60)
    try:
        simple_example()
    except Exception as e:
        logging.error(f"Simple example failed: {e}")
        logging.info("\nTrying with transforms example instead...")
    
    print("\n" + "=" * 60)
    print("Example with Transforms")
    print("=" * 60)
    try:
        with_transforms_example()
    except Exception as e:
        logging.error(f"Transforms example failed: {e}")
        logging.info("\nMake sure:")
        logging.info("1. The dataset exists and is accessible")
        logging.info("2. You have the correct action_dim for your dataset")
        logging.info("3. The action_sequence_keys match your dataset structure")
