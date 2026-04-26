import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = (ROOT / "common" / "sdxl_tpu_common.py").read_text(encoding="utf-8")


def notebook_for(phase, use_tpu=True):
    runner = "run_b1" if phase == "b1" else "run_b2"
    runtime = "TPU" if use_tpu else "CPU"
    title = f"SDXL Convert {phase.upper()} {runtime}"
    code = (
        "CIVITAI_VERSION_ID = '__CIVITAI_VERSION_ID__'\n"
        "MODEL_NAME = '__MODEL_NAME__'\n"
        "REALISTIC = '__REALISTIC__'\n"
        "MIN_SOC = '__MIN_SOC__'\n"
        f"USE_TPU_BURNER = {str(use_tpu)!r}\n"
        f"COMMON_CODE = {COMMON!r}\n"
        "exec(COMMON_CODE, globals())\n"
        f"{runner}(MODEL_NAME, MIN_SOC, REALISTIC, CIVITAI_VERSION_ID, USE_TPU_BURNER.lower() == 'true')\n"
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


def write_phase(phase, use_tpu=True):
    suffix = "tpu" if use_tpu else "cpu"
    out_dir = ROOT / f"notebook_{phase}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notebook.ipynb").write_text(
        json.dumps(notebook_for(phase, use_tpu=use_tpu), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    for runtime in (True, False):
        write_phase("b1", use_tpu=runtime)
        write_phase("b2", use_tpu=runtime)
