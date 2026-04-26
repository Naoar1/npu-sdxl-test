import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = (ROOT / "common" / "sdxl_tpu_common.py").read_text(encoding="utf-8")


def notebook_for(phase):
    runner = "run_b1" if phase == "b1" else "run_b2"
    title = "SDXL Convert B1 TPU" if phase == "b1" else "SDXL Convert B2 TPU UNet"
    code = (
        "CIVITAI_VERSION_ID = '__CIVITAI_VERSION_ID__'\n"
        "MODEL_NAME = '__MODEL_NAME__'\n"
        "REALISTIC = '__REALISTIC__'\n"
        "MIN_SOC = '__MIN_SOC__'\n"
        f"COMMON_CODE = {COMMON!r}\n"
        "exec(COMMON_CODE, globals())\n"
        f"{runner}(MODEL_NAME, MIN_SOC, REALISTIC, CIVITAI_VERSION_ID)\n"
    )
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "Generated from `kaggle/common/sdxl_tpu_common.py`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_phase(phase):
    out_dir = ROOT / ("notebook_b1_tpu" if phase == "b1" else "notebook_b2_tpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notebook.ipynb").write_text(
        json.dumps(notebook_for(phase), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_phase("b1")
    write_phase("b2")
