import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SLUG_RE = re.compile(r"^sdxl-(cpu|tpu)-(a|b1|b2)-(.+)-(8gen3|8gen4)-(r[01])-(\d{8}-\d{6})$")


def run(cmd, check=True, cwd=None, capture=False, env=None):
    print("+ " + " ".join(cmd[:2]) + (" ..." if len(cmd) > 2 else ""), flush=True)
    if capture:
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if check and p.returncode != 0:
            print(p.stdout)
            raise SystemExit(p.returncode)
        return p.stdout
    p = subprocess.run(cmd, cwd=cwd, env=env)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return ""


def sanitize_model(name):
    out = re.sub(r"[^a-z0-9-]+", "-", name.lower())
    out = re.sub(r"-+", "-", out).strip("-")
    if not out:
        raise SystemExit("model_name sanitized to empty")
    return out


def bool_slug(value):
    return "r1" if str(value).lower() == "true" else "r0"


def pipeline_slugs(model, soc, realistic, stamp=None, runtime="tpu"):
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base = f"{sanitize_model(model)}-{soc}-{bool_slug(realistic)}-{stamp}"
    return {
        "a": f"sdxl-{runtime}-a-{base}",
        "b1": f"sdxl-{runtime}-b1-{base}",
        "b2": f"sdxl-{runtime}-b2-{base}",
        "stamp": stamp,
    }


def parse_slug(slug):
    m = SLUG_RE.match(slug)
    if not m:
        return None
    runtime, stage, model, soc, real, stamp = m.groups()
    return {
        "runtime": runtime,
        "stage": stage,
        "model": model,
        "soc": soc,
        "realistic": "true" if real == "r1" else "false",
        "stamp": stamp,
        "key": (runtime, model, soc, real, stamp),
    }


def kaggle_status(ref):
    out = run(["kaggle", "kernels", "status", ref], check=False, capture=True)
    print(out.strip())
    m = re.search(r"(complete|running|error|cancel_acknowledged|cancel_requested|queued|new_script)", out, re.I)
    return m.group(1).lower() if m else "unknown"


def list_pipeline_kernels(username):
    commands = [
        ["kaggle", "kernels", "list", "--mine", "--page-size", "200", "--csv"],
        ["kaggle", "kernels", "list", "--mine", "--search", "sdxl", "--page-size", "200", "--csv"],
        ["kaggle", "kernels", "list", "--user", username, "--page-size", "200", "--csv"],
    ]
    refs = set()
    raw_outputs = []
    for cmd in commands:
        out = run(cmd, check=False, capture=True)
        raw_outputs.append(out)
        for row in csv.reader(out.splitlines()):
            for cell in row:
                cell = cell.strip().strip('"')
                if not cell:
                    continue
                m = re.search(r"kaggle\.com/code/([^/\s,]+)/([^/\s,]+)", cell)
                if m:
                    owner, slug = m.group(1), m.group(2)
                elif "/" in cell:
                    owner, slug = cell.split("/", 1)
                else:
                    owner, slug = username, cell
                slug = slug.split("?", 1)[0].strip()
                if owner == username and parse_slug(slug):
                    refs.add(f"{owner}/{slug}")
        refs.update(
            f"{username}/{slug}"
            for slug in re.findall(
                r"sdxl-(?:cpu|tpu)-(?:a|b1|b2)-[a-z0-9-]+-(?:8gen3|8gen4)-r[01]-\d{8}-\d{6}",
                out,
            )
            if parse_slug(slug)
        )
    if not refs:
        previews = []
        for out in raw_outputs:
            for line in out.splitlines()[1:8]:
                previews.append(line[:240])
        print("No sdxl refs parsed from Kaggle list output. Preview:")
        for line in previews[:20]:
            print(line)
    grouped = {}
    for ref in sorted(refs):
        slug = ref.split("/", 1)[1]
        info = parse_slug(slug)
        if not info:
            continue
        grouped.setdefault(info["key"], {})[info["stage"]] = ref
    return grouped


def patch_notebook(path, subs):
    nb = json.load(open(path, encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for key, value in subs.items():
            src = src.replace(key, value)
        cell["source"] = src.splitlines(keepends=True)
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def reset_dir_from_repo(src_dir, work_dir):
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(src_dir, work_dir)


def prepare_a(work_root, username, slug, title, inputs):
    src = ROOT / "notebook_a_gpu"
    dst = work_root / "notebook_a_gpu"
    reset_dir_from_repo(src, dst)
    meta = json.load(open(dst / "kernel-metadata.json", encoding="utf-8"))
    meta["id"] = f"{username}/{slug}"
    meta["title"] = title
    json.dump(meta, open(dst / "kernel-metadata.json", "w", encoding="utf-8"), indent=1)
    key = os.environ.get("CIVITAI_API_KEY", "")
    if not key:
        raise SystemExit("CIVITAI_API_KEY is required when skip_phase1=false")
    patch_notebook(
        dst / "notebook.ipynb",
        {
            "__CIVITAI_VERSION_ID__": inputs["civitai_version_id"],
            "__MODEL_NAME__": inputs["model_name"],
            "__REALISTIC__": inputs["realistic"],
            "__CIVITAI_API_KEY__": key,
        },
    )
    return dst


def prepare_b(work_root, stage, username, slug, source_ref, inputs):
    runtime = inputs.get("runtime", "tpu")
    src = ROOT / f"notebook_{stage}_{runtime}"
    dst = work_root / src.name
    reset_dir_from_repo(src, dst)
    meta = json.load(open(dst / "kernel-metadata.json", encoding="utf-8"))
    meta["id"] = f"{username}/{slug}"
    meta["title"] = slug
    if stage == "b1":
        meta["kernel_sources"] = [source_ref]
    else:
        meta["kernel_sources"] = [source_ref]
    json.dump(meta, open(dst / "kernel-metadata.json", "w", encoding="utf-8"), indent=1)
    patch_notebook(
        dst / "notebook.ipynb",
        {
            "__CIVITAI_VERSION_ID__": inputs.get("civitai_version_id", ""),
            "__MODEL_NAME__": inputs["model_name"],
            "__REALISTIC__": inputs["realistic"],
            "__MIN_SOC__": inputs["min_soc"],
        },
    )
    return dst


def push_kernel(folder, accelerator=None):
    cmd = ["kaggle", "kernels", "push", "-p", str(folder)]
    run(cmd)


def output_to(ref, target, diagnostics_only=False):
    target.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(target), "--force"], check=False)
    print_tree(target)
    tails = [
        "logs/b1_phase45.qnn_status.tsv",
        "logs/b1_phase45.qnn_progress.tsv",
        "logs/b1_phase45.qnn_stage.tsv",
        "logs/b2_unet.qnn_status.tsv",
        "logs/b2_unet.qnn_progress.tsv",
        "logs/b2_unet.qnn_stage.tsv",
        "logs/phase3.log",
        "logs/package_zip.log",
    ]
    for rel in tails:
        p = target / rel
        if p.exists():
            print(f"===== tail {rel} =====")
            try:
                print("\n".join(p.read_text(errors="replace").splitlines()[-60:]))
            except UnicodeDecodeError:
                pass
    if diagnostics_only:
        return


def fail_on_kaggle_error():
    return os.environ.get("FAIL_ON_KAGGLE_ERROR", "true").lower() == "true"


def handle_kaggle_error(stage, ref, state, out):
    output_to(ref, out)
    msg = f"{stage} failed state={state}"
    if fail_on_kaggle_error():
        raise SystemExit(msg)
    print(f"{msg}; FAIL_ON_KAGGLE_ERROR=false so watcher exits without failing")


def require_b1_bundle_ready(out):
    manifests = list(out.rglob("b1_bundle/manifest.json"))
    if not manifests:
        raise RuntimeError("B1 output is complete but b1_bundle/manifest.json is missing")
    manifest_path = manifests[0]
    bundle = manifest_path.parent
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    required = [
        "unet_sdxl/model.onnx",
        "input_list_unet_sdxl.txt",
        "output",
    ]
    missing = [rel for rel in required if not (bundle / rel).exists()]
    if missing:
        raise RuntimeError("B1 bundle is incomplete; missing: " + ", ".join(missing))
    unet_files = manifest.get("unet_sdxl_files", [])
    if "unet_sdxl/model.onnx" not in unet_files:
        raise RuntimeError("B1 bundle manifest is from an old/incomplete notebook: unet_sdxl_files missing")
    missing_unet = [rel for rel in unet_files[:200] if not (bundle / rel).exists()]
    if missing_unet:
        raise RuntimeError("B1 bundle unet_sdxl files missing from output: " + ", ".join(missing_unet[:20]))
    print(f"B1 bundle ready manifest={manifest_path} unet_files={len(unet_files)}")


def print_tree(root):
    for p in sorted(root.rglob("*")):
        if p.is_file():
            print(f"{p} {p.stat().st_size} bytes")


def newest_pipeline(grouped):
    if not grouped:
        return None, None
    key = sorted(grouped.keys(), key=lambda k: k[-1])[-1]
    return key, grouped[key]


def inputs_from_key(key):
    runtime, model, soc, real, _stamp = key
    return {
        "runtime": runtime,
        "model_name": model,
        "min_soc": soc,
        "realistic": "true" if real == "r1" else "false",
        "civitai_version_id": os.environ.get("CIVITAI_VERSION_ID", "2883731"),
    }


def start(username, inputs):
    slugs = pipeline_slugs(
        inputs["model_name"], inputs["min_soc"], inputs["realistic"], runtime=inputs.get("runtime", "tpu")
    )
    work_root = REPO / ".kaggle_work" / slugs["stamp"]
    work_root.mkdir(parents=True, exist_ok=True)
    if inputs["skip_phase1"] == "true":
        phase1_ref = inputs["phase1_slug"]
        if "/" not in phase1_ref:
            phase1_ref = f"{username}/{phase1_ref}"
        state = kaggle_status(phase1_ref)
        if state != "complete":
            raise SystemExit(f"phase1_ref is not complete: {phase1_ref} state={state}")
        folder = prepare_b(work_root, "b1", username, slugs["b1"], phase1_ref, inputs)
        push_kernel(folder)
        print(f"STARTED_B1={username}/{slugs['b1']}")
    else:
        folder = prepare_a(work_root, username, slugs["a"], slugs["a"], inputs)
        push_kernel(folder, accelerator="NvidiaTeslaT4")
        print(f"STARTED_A={username}/{slugs['a']}")


def watch(username):
    grouped = list_pipeline_kernels(username)
    key, refs = newest_pipeline(grouped)
    if not key:
        print("No sdxl pipeline kernels found")
        return
    inputs = inputs_from_key(key)
    runtime, model, soc, real, stamp = key
    print(f"Watching pipeline runtime={runtime} model={model} soc={soc} realistic={real} stamp={stamp} refs={refs}")
    work_root = REPO / ".kaggle_work" / stamp
    work_root.mkdir(parents=True, exist_ok=True)

    if "b2" in refs:
        state = kaggle_status(refs["b2"])
        out = REPO / "output" / refs["b2"].split("/", 1)[1]
        if state == "complete":
            output_to(refs["b2"], out)
            print(f"B2_COMPLETE={refs['b2']}")
        elif state in {"error", "cancel_acknowledged", "cancel_requested"}:
            handle_kaggle_error("B2", refs["b2"], state, out)
        else:
            output_to(refs["b2"], out, diagnostics_only=True)
            print(f"B2 still {state}; watcher exits")
        return

    if "b1" in refs:
        state = kaggle_status(refs["b1"])
        out = REPO / "output" / refs["b1"].split("/", 1)[1]
        if state == "complete":
            output_to(refs["b1"], out, diagnostics_only=True)
            try:
                require_b1_bundle_ready(out)
            except RuntimeError as e:
                if fail_on_kaggle_error():
                    raise
                print(f"B1 complete but not promotable: {e}; watcher exits")
                return
            folder = prepare_b(work_root, "b2", username, f"sdxl-{runtime}-b2-{model}-{soc}-{real}-{stamp}", refs["b1"], inputs)
            push_kernel(folder)
            print(f"STARTED_B2={username}/sdxl-{runtime}-b2-{model}-{soc}-{real}-{stamp}")
        elif state in {"error", "cancel_acknowledged", "cancel_requested"}:
            handle_kaggle_error("B1", refs["b1"], state, out)
        else:
            output_to(refs["b1"], out, diagnostics_only=True)
            print(f"B1 still {state}; watcher exits")
        return

    if "a" in refs:
        state = kaggle_status(refs["a"])
        if state == "complete":
            folder = prepare_b(work_root, "b1", username, f"sdxl-{runtime}-b1-{model}-{soc}-{real}-{stamp}", refs["a"], inputs)
            push_kernel(folder)
            print(f"STARTED_B1={username}/sdxl-{runtime}-b1-{model}-{soc}-{real}-{stamp}")
        elif state in {"error", "cancel_acknowledged", "cancel_requested"}:
            out = REPO / "output" / refs["a"].split("/", 1)[1]
            handle_kaggle_error("A", refs["a"], state, out)
        else:
            print(f"A still {state}; watcher exits")
        return

    print("Pipeline has no actionable stage")


def main():
    username = os.environ["KAGGLE_USERNAME"]
    action = os.environ.get("PIPELINE_ACTION", "watch")
    if action == "start":
        inputs = {
            "civitai_version_id": os.environ.get("CIVITAI_VERSION_ID", "2883731"),
            "runtime": os.environ.get("RUNTIME_MODE", "tpu"),
            "model_name": os.environ.get("MODEL_NAME", "customxl"),
            "realistic": os.environ.get("REALISTIC", "false"),
            "min_soc": os.environ.get("MIN_SOC", "8gen3"),
            "skip_phase1": os.environ.get("SKIP_PHASE1", "true"),
            "phase1_slug": os.environ.get("PHASE1_SLUG", ""),
        }
        if inputs["skip_phase1"] == "true" and not inputs["phase1_slug"]:
            raise SystemExit("PHASE1_SLUG is required when SKIP_PHASE1=true")
        start(username, inputs)
    elif action == "watch":
        watch(username)
    else:
        raise SystemExit(f"unknown PIPELINE_ACTION={action}")


if __name__ == "__main__":
    main()
