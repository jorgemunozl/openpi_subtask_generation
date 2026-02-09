# Using LeRobot Datasets

This directory contains examples for loading and using LeRobot datasets with the openpi data loader.

## Quick Start

### Simple Example (Minimal Setup)

```bash
python examples/simple_lerobot_example.py
```

This script demonstrates:
- Loading a LeRobot dataset directly
- Inspecting dataset samples
- Using transforms (optional)

### Full Example (With CLI)

```bash
# Using a HuggingFace dataset
python examples/use_lerobot_dataset.py --repo_id lerobot/aloha_sim_transfer_cube_human

# Using a local dataset
python examples/use_lerobot_dataset.py --repo_id /path/to/local/dataset --skip_norm_stats

# Customize batch size and number of batches
python examples/use_lerobot_dataset.py \
    --repo_id your_hf_username/your_dataset \
    --batch_size 8 \
    --num_batches 5 \
    --action_horizon 20
```

## Dataset Format Requirements

Your LeRobot dataset should have:
- **Images**: Stored with `dtype="image"` in the dataset
- **State**: Robot state/proprioception (e.g., joint positions)
- **Actions**: Action sequences (key name configurable via `action_sequence_keys`)
- **Optional**: Task/prompt labels (if using `prompt_from_task=True`)

## Configuration

### Option 1: Use Existing Config

If your dataset matches an existing robot type (Aloha, Libero, etc.), use the corresponding config:

```python
from openpi.training.config import LeRobotAlohaDataConfig

data_config = LeRobotAlohaDataConfig(
    repo_id="your_dataset",
    default_prompt="Your task description",
)
```

### Option 2: Create Custom Config

For custom datasets, create a `DataConfigFactory` similar to the examples in `src/openpi/training/config.py`:

```python
from openpi.training.config import DataConfigFactory, DataConfig
from openpi.transforms import Group, RepackTransform

class MyCustomDataConfig(DataConfigFactory):
    def create(self, assets_dirs, model_config):
        repack_transform = Group(
            inputs=[
                RepackTransform({
                    "images": {"camera1": "observation.image"},  # Map your keys
                    "state": "observation.state",
                    "actions": "action",
                })
            ]
        )
        
        return self.create_base_config(assets_dirs, model_config).replace(
            repack_transforms=repack_transform,
            action_sequence_keys=("action",),  # Your action key name
        )
```

## Troubleshooting

### Import Error: `lerobot_dataset` not found
- Make sure lerobot is installed: `pip install lerobot` or check `pyproject.toml`
- The import was uncommented in `data_loader.py` - verify it's active

### Dataset Not Found
- Check that `repo_id` is correct
- For local datasets, ensure the path is correct
- For HuggingFace datasets, ensure you're logged in if needed: `huggingface-cli login`

### Normalization Stats Missing
- Use `--skip_norm_stats` flag for testing
- For training, compute norm stats using `scripts/compute_norm_stats.py`

### Wrong Action Dimensions
- Check your dataset's action shape
- Adjust `action_dim` in the model config to match
- Verify `action_sequence_keys` matches your dataset structure

### Key Mapping Issues
- Inspect your dataset keys: `dataset[0].keys()`
- Adjust `repack_transforms` in your DataConfig to map correctly
- See existing configs in `config.py` for examples

## Examples in Codebase

- `LeRobotAlohaDataConfig`: For Aloha robot datasets
- `LeRobotLiberoDataConfig`: For Libero manipulation datasets  
- `LeRobotX2robotDataConfig`: For X2Robot datasets
- `LeRobotDROIDDataConfig`: For DROID datasets

Each config shows how to:
- Map dataset keys to model inputs
- Apply robot-specific transforms
- Handle action spaces (delta vs absolute)
