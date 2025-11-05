"""Convert TimesFM safetensors checkpoint to PyTorch .pth format.

Usage:
  python -m MERIT.scripts.convert_timesfm_checkpoint \
      --source /path/to/checkpoint_dir_or_safetensors \
      --output /path/to/output.pth

Requires the `timesfm` python package.
"""

import argparse
from pathlib import Path

import torch


def convert_timesfm_checkpoint(src: Path, dst: Path) -> None:
    try:
        from timesfm import TimesFM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "timesfm 库未安装，请先 `pip install timesfm` 或激活相应环境后再执行转换"
        ) from exc

    if src.is_file() and src.suffix == ".safetensors":
        model_dir = src.parent
    elif src.is_dir():
        model_dir = src
    else:
        raise FileNotFoundError(f"未识别的 TimesFM checkpoint 路径: {src}")

    print(f"Loading TimesFM checkpoint from: {model_dir}")
    model = TimesFM.from_pretrained(str(model_dir))
    state_dict = model.state_dict()

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, dst)
    print(f"Converted checkpoint saved to: {dst}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TimesFM safetensors to .pth")
    parser.add_argument("--source", type=str, required=True, help="TimesFM checkpoint目录或safetensors文件")
    parser.add_argument("--output", type=str, required=True, help="输出的 .pth 文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_timesfm_checkpoint(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()

