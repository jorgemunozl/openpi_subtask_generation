import dataclasses
import functools
import json
import logging
import platform
import pathlib
from typing import Any, Literal

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.pi05_config as pi05_config
import openpi.models.tokenizer as _tokenizer
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from typing_extensions import override


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    real_action_dim: int,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, real_action_dim=real_action_dim, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _choose_prompt_column(column_names: list[str], string_columns: list[str]) -> str:
    preferred = ("task", "text", "instruction", "prompt")
    for name in preferred:
        if name in column_names:
            return name
    if string_columns:
        return string_columns[0]
    raise ValueError("No suitable text column found in parquet file.")


def _load_prompt_mapping(path: str, *, column: str | None = None) -> dict[int, str] | list[str]:
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        try:
            import pyarrow.parquet as pq
            import pyarrow.types as pt
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires pyarrow. Install it or convert the parquet to JSON."
            ) from exc

        table = pq.read_table(path)
        column_names = list(table.column_names)
        string_columns = [name for name in column_names if pt.is_string(table.schema.field(name).type)]
        selected = column if column else _choose_prompt_column(column_names, string_columns)
        if selected not in column_names:
            raise ValueError(f'Column "{selected}" not found in parquet file {path}.')
        values = table[selected].to_pylist()
        return ["" if v is None else str(v) for v in values]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        return {int(k): str(v) for k, v in data.items()}
    raise ValueError(f"Unsupported mapping format in {path}; expected list or dict.")


def _load_repack_mapping(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclasses.dataclass(frozen=True)
class HighLowTaskDataConfig(_config.DataConfigFactory):
    repo_id: str
    general_tasks_path: str
    tasks_path: str
    general_tasks_column: str | None = None
    tasks_column: str | None = None
    general_task_key: str = "general_task_index"
    task_key: str = "task_index"
    repack_mapping_path: str | None = None
    action_sequence_keys: tuple[str, ...] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        general_tasks = _load_prompt_mapping(self.general_tasks_path, column=self.general_tasks_column)
        tasks = _load_prompt_mapping(self.tasks_path, column=self.tasks_column)

        repack_transforms = _transforms.Group()
        if self.repack_mapping_path is not None:
            repack_transforms = _transforms.Group(
                inputs=[_transforms.RepackTransform(_load_repack_mapping(self.repack_mapping_path))]
            )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.PromptFromTaskIndices(
                    general_tasks=general_tasks,
                    tasks=tasks,
                    general_task_key=self.general_task_key,
                    task_key=self.task_key,
                )
            ]
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizeHighLowPrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass
class Args:
    repo_id: str
    general_tasks_path: str
    tasks_path: str
    general_tasks_column: str | None = None
    tasks_column: str | None = None
    exp_name: str
    config_name: str = "pi05_highlow"
    batch_size: int = 32
    num_train_steps: int = 30_000
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    seed: int = 42
    fsdp_devices: int = 1
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 1000
    overwrite: bool = False
    resume: bool = False
    project_name: str = "openpi"
    assets_base_dir: str = "/x2robot_v2/xinyuanfang/projects_v2/openpi/assets"
    checkpoint_base_dir: str = "/x2robot_v2/xinyuanfang/projects_v2/openpi/checkpoints"
    weight_type: Literal["none", "checkpoint", "paligemma"] = "none"
    weight_path: str | None = None
    repack_mapping_path: str | None = None
    general_task_key: str = "general_task_index"
    task_key: str = "task_index"
    action_sequence_keys: str = "actions"
    wandb_enabled: bool = True


def _build_weight_loader(args: Args) -> _weight_loaders.WeightLoader:
    if args.weight_type == "checkpoint":
        if args.weight_path is None:
            raise ValueError("--weight_path is required when --weight_type=checkpoint")
        return _weight_loaders.CheckpointWeightLoader(args.weight_path)
    if args.weight_type == "paligemma":
        return _weight_loaders.PaliGemmaWeightLoader()
    return _weight_loaders.NoOpWeightLoader()


def main(args: Args) -> None:
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if args.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {args.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    data_config = HighLowTaskDataConfig(
        repo_id=args.repo_id,
        general_tasks_path=args.general_tasks_path,
        tasks_path=args.tasks_path,
        general_tasks_column=args.general_tasks_column,
        tasks_column=args.tasks_column,
        general_task_key=args.general_task_key,
        task_key=args.task_key,
        repack_mapping_path=args.repack_mapping_path,
        action_sequence_keys=tuple(k.strip() for k in args.action_sequence_keys.split(",")),
    )

    config = _config.TrainConfig(
        name=args.config_name,
        project_name=args.project_name,
        exp_name=args.exp_name,
        model=pi05_config.Pi05Config(
            action_dim=args.action_dim,
            action_horizon=args.action_horizon,
            max_token_len=args.max_token_len,
        ),
        weight_loader=_build_weight_loader(args),
        data=data_config,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # If no norm stats are found, skip normalization to avoid hard failure.
    data_cfg = config.data.create(config.assets_dirs, config.model)
    skip_norm_stats = data_cfg.norm_stats is None
    if skip_norm_stats:
        logging.info("No normalization stats found; skipping normalization.")

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        skip_norm_stats=skip_norm_stats,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    real_action_dim = config.model.action_dim
    ptrain_step = jax.jit(
        functools.partial(train_step, config, real_action_dim=real_action_dim),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(tyro.cli(Args))
