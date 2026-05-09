"""Download a Roboflow dataset export for training.

Example:
  export ROBOFLOW_API_KEY=...
  uv run python scripts/download_roboflow.py \
      --workspace practicum-ziryz \
      --project fridge-dataset-oi7ld \
      --version 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from roboflow import Roboflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True, type=str)
    p.add_argument('--project', required=True, type=str)
    p.add_argument('--version', required=True, type=int)
    p.add_argument('--format', default='yolov8', type=str,
                   help='Roboflow export format. Keep yolov8 for this project.')
    p.add_argument('--output-dir', default='data/roboflow', type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        raise RuntimeError('Missing ROBOFLOW_API_KEY environment variable')

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download(args.format, location=str(output_dir), overwrite=True)

    data_yaml = Path(dataset.location) / 'data.yaml'
    if not data_yaml.exists():
        raise RuntimeError(
            f'Roboflow reported success but data.yaml was not found in {dataset.location}'
        )

    print(f'Downloaded Roboflow dataset to: {dataset.location}')


if __name__ == '__main__':
    main()