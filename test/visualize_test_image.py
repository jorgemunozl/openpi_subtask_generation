#!/usr/bin/env python3
import argparse
import os

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.model import Observation


DEFAULT_IMAGE_MAP = {
    "base_0_rgb": "to.png",
    "left_wrist_0_rgb": "lef.png",
    "right_wrist_0_rgb": "righ.png",
}


def _load_images(img_dir: str, image_map: dict[str, str]) -> dict[str, jnp.ndarray]:
    images = {}
    for key, filename in image_map.items():
        path = os.path.join(img_dir, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image for {key}: {path}")
        images[key] = jnp.asarray(image[None, ...], dtype=jnp.float32) / 127.5 - 1.0
    return images


def _to_uint8(image: jnp.ndarray) -> np.ndarray:
    image = np.asarray(image)
    image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return image


def _save_images(output_dir: str, prefix: str, images: dict[str, jnp.ndarray]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key, image in images.items():
        out_path = os.path.join(output_dir, f"{prefix}{key}.png")
        cv2.imwrite(out_path, _to_uint8(image[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize images after preprocess_observation.")
    parser.add_argument(
        "--img-dir",
        default="/home/lperez/main/nh/openpi/test",
        help="Directory containing test.png/leftImg.png/rightImg.png",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/lperez/main/nh/openpi/test/visualized",
        help="Where to write the original and preprocessed images",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Apply train-time augmentations in preprocess_observation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train-time augmentations",
    )
    args = parser.parse_args()

    images = _load_images(args.img_dir, DEFAULT_IMAGE_MAP)
    observation = Observation(
        images=images,
        image_masks={},
        state=jnp.zeros((1, 32), dtype=jnp.float32),
    )

    rng = jax.random.key(args.seed) if args.train else None
    processed = _model.preprocess_observation(
        rng,
        observation,
        train=args.train,
        image_keys=list(images.keys()),
    )

    _save_images(args.output_dir, "orig_", images)
    _save_images(args.output_dir, "preproc_", processed.images)
    print(f"Wrote images to: {args.output_dir}")


if __name__ == "__main__":
    main()
