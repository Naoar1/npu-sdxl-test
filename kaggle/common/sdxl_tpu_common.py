import glob
import json
import os
import re
import shutil
import signal
import select
import subprocess
import time


LOG_DIR = "/kaggle/working/logs"


def now_hms():
    return time.strftime("%H:%M:%S")


def run(cmd, log=None, check=True, cwd=None, env=None):
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        wrapped = f"set -eo pipefail; {cmd} 2>&1 | tee {log}"
    else:
        wrapped = f"set -eo pipefail; {cmd}"
    rc = subprocess.call(["bash", "-lc", wrapped], cwd=cwd, env=env)
    if check and rc != 0:
        raise RuntimeError(f"shell failed rc={rc}: {cmd[:240]} log={log}")
    return rc


def file_mib(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def mem_avail_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def disk_avail_mb(path="/kaggle/working"):
    try:
        st = os.statvfs(path)
        return int(st.f_bavail * st.f_frsize / (1024 * 1024))
    except Exception:
        return -1


def tsv_escape(value, limit=1400):
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")[:limit]


def configure_env(model_name, min_soc, realistic, civitai_version_id=None):
    os.environ["MODEL_NAME"] = model_name
    os.environ["MIN_SOC"] = min_soc
    os.environ["REALISTIC"] = str(realistic).lower()
    if civitai_version_id is not None:
        os.environ["CIVITAI_VERSION_ID"] = str(civitai_version_id)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(
        f"[*] {now_hms()} config model={model_name} soc={min_soc} "
        f"realistic={os.environ['REALISTIC']}"
    )


def install_tools_and_convert_package():
    print(f"[*] {now_hms()} installing system tools and convertsdxl")
    run("apt-get update -y > /dev/null")
    run("apt-get install -y unzip zip curl time libc++1-19 libc++abi1-19 libunwind-19 > /dev/null")
    run("ldconfig")
    run("curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1")
    os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"
    run("rm -rf /tmp/convertsdxl.zip /tmp/convertsdxl_unzip")
    run(
        "curl -sL --fail --retry 5 --retry-delay 5 "
        "-o /tmp/convertsdxl.zip 'https://chino.icu/local-dream/convertsdxl.zip'"
    )
    run("mkdir -p /tmp/convertsdxl_unzip")
    run("unzip -q /tmp/convertsdxl.zip -d /tmp/convertsdxl_unzip")
    root = subprocess.check_output(
        "find /tmp/convertsdxl_unzip -maxdepth 3 -name 'export_sdxl.sh' -printf '%h\n' | head -1",
        shell=True,
        text=True,
    ).strip()
    assert root, "export_sdxl.sh not found in convertsdxl.zip"
    os.environ["NPUCONVERT_DIR"] = os.path.abspath(root)
    print(f"[*] {now_hms()} NPUCONVERT_DIR={os.environ['NPUCONVERT_DIR']}")


def install_qnn_sdk():
    print(f"[*] {now_hms()} installing QNN SDK")
    run("mkdir -p /tmp/qnn_sdk")
    run(
        "curl -sL --fail --retry 5 --retry-delay 5 -A 'Mozilla/5.0' "
        "-o /tmp/qnn_sdk/qnn.zip "
        "'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/"
        "qualcomm_neural_processing_sdk/v2.28.0.241029.zip'"
    )
    run("cd /tmp/qnn_sdk && unzip -q qnn.zip")
    envsetup = subprocess.check_output(
        "find /tmp/qnn_sdk -type f -name envsetup.sh -print -quit",
        shell=True,
        text=True,
    ).strip()
    assert envsetup, "envsetup.sh not found in QNN SDK"
    os.environ["QNN_SDK_BIN"] = os.path.dirname(envsetup)
    os.environ["QNN_SDK_ROOT_DIR"] = os.path.dirname(os.environ["QNN_SDK_BIN"])
    print(f"[*] {now_hms()} QNN_SDK_ROOT={os.environ['QNN_SDK_ROOT_DIR']}")


def setup_python_env():
    os.chdir(os.environ["NPUCONVERT_DIR"])
    run("uv venv -p 3.10.17 --clear")
    run(". .venv/bin/activate && uv sync")


def locate_phase1_output():
    pkl = glob.glob("/kaggle/input/**/data_sdxl.pkl", recursive=True)
    assert pkl, "Phase 1 data_sdxl.pkl not found in /kaggle/input"
    phase1_dir = os.path.dirname(pkl[0])
    model = glob.glob("/kaggle/input/**/model.safetensors", recursive=True)
    assert model, "model.safetensors not found in /kaggle/input"
    size = os.path.getsize(model[0])
    assert size > int(1e9), f"model.safetensors truncated: {size}"
    os.environ["PHASE1_DIR"] = phase1_dir
    os.environ["MODEL_PATH"] = model[0]
    print(f"[*] {now_hms()} Phase1 dir={phase1_dir}")
    print(f"[*] {now_hms()} MODEL_PATH size={size / 1e9:.2f}GB")


def copy_phase1_calibration_data():
    npu = os.environ["NPUCONVERT_DIR"]
    pkl = os.path.join(os.environ["PHASE1_DIR"], "data_sdxl.pkl")
    shutil.copy(pkl, f"{npu}/data_sdxl.pkl")
    src_img = f"{os.environ['PHASE1_DIR']}/images_sdxl"
    assert os.path.isdir(src_img), f"{src_img} missing"
    dst_img = f"{npu}/images_sdxl"
    if os.path.exists(dst_img):
        shutil.rmtree(dst_img)
    shutil.copytree(src_img, dst_img)
    assert os.listdir(dst_img), f"{dst_img} empty"


def start_background_services():
    os.makedirs(LOG_DIR, exist_ok=True)
    watcher_sh = """#!/bin/bash
echo "epoch,datetime,MemAvail_MB,TopRSS_MB,TopProc"
while true; do
  epoch=$(date +%s); dt=$(date -Iseconds)
  mem=$(awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  line=$(ps -eo rss,comm --sort=-rss --no-headers 2>/dev/null | head -1)
  rss=$(echo "$line" | awk '{print int($1/1024)}')
  proc=$(echo "$line" | awk '{print $2}')
  echo "$epoch,$dt,$mem,$rss,$proc"
  sleep 10
done
"""
    open("/tmp/mem_watch.sh", "w").write(watcher_sh)
    os.chmod("/tmp/mem_watch.sh", 0o755)
    p = subprocess.Popen(
        ["/tmp/mem_watch.sh"],
        stdout=open(f"{LOG_DIR}/mem-trace.csv", "w"),
        stderr=subprocess.DEVNULL,
    )
    open("/tmp/watcher.pid", "w").write(str(p.pid))
    print(f"[*] {now_hms()} mem-watcher pid={p.pid}")

    print(f"[*] {now_hms()} installing tensorflow-tpu runtime")
    run('export PATH="${HOME}/.local/bin:${PATH}" && uv pip uninstall --system jax 2>&1 | tail -3', check=False)
    run(
        'export PATH="${HOME}/.local/bin:${PATH}" && uv pip install --system --quiet '
        "tensorflow-tpu==2.18.0 --find-links https://storage.googleapis.com/libtpu-tf-releases/index.html"
    )

    burner_py = r'''
import json, os, signal, sys, time, traceback
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
LOG_DIR = "/kaggle/working/logs"
OK_FILE = f"{LOG_DIR}/tpu_burner.ok"
JSONL = f"{LOG_DIR}/tpu_burner.jsonl"
os.makedirs(LOG_DIR, exist_ok=True)
stopping = False
def log(msg):
    print("[tpu-burner " + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)
def write_ok(record):
    tmp = OK_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, sort_keys=True)
    os.replace(tmp, OK_FILE)
    with open(JSONL, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
def term(signum, frame):
    global stopping
    stopping = True
    log("signal received")
signal.signal(signal.SIGTERM, term)
signal.signal(signal.SIGINT, term)
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    log("initializing TPU")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    replicas = strategy.num_replicas_in_sync
    log("TPU ready replicas=" + str(replicas))
    dim, hidden, batch, steps_per_chunk = 1024, 2048, 1024, 8
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(dim,)),
            tf.keras.layers.Dense(hidden, activation="gelu"),
            tf.keras.layers.Dense(hidden, activation="gelu"),
            tf.keras.layers.Dense(dim),
        ])
        opt = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-3)
        x = tf.ones((batch, dim), dtype=tf.bfloat16)
        target = tf.zeros((batch, dim), dtype=tf.bfloat16)
        _ = model(x, training=True)
        try:
            opt.build(model.trainable_variables)
        except AttributeError:
            pass
    @tf.function
    def train_chunk():
        total = tf.constant(0.0, tf.float32)
        for _ in range(steps_per_chunk):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                diff = tf.cast(pred - target, tf.float32)
                loss = tf.reduce_mean(diff * diff)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            total += loss
        return total / tf.cast(steps_per_chunk, tf.float32)
    chunk = 0
    last_log = 0
    while not stopping:
        t0 = time.time()
        per_replica = strategy.run(train_chunk)
        reduced = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)
        elapsed = time.time() - t0
        chunk += 1
        record = {
            "epoch": int(time.time()),
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunk": chunk,
            "replicas": int(replicas),
            "steps_per_chunk": steps_per_chunk,
            "elapsed_sec": round(elapsed, 3),
            "loss": float(reduced.numpy()),
        }
        write_ok(record)
        if time.time() - last_log >= 30:
            log("chunk={chunk} loss={loss:.6f} elapsed={elapsed_sec:.2f}s replicas={replicas}".format(**record))
            last_log = time.time()
        time.sleep(0.2)
    log("clean exit")
except Exception as e:
    log("FATAL " + repr(e))
    traceback.print_exc()
    sys.exit(2)
'''
    open(f"{LOG_DIR}/tpu_burner.py", "w").write(burner_py)

    watchdog_py = r'''
import os, signal, subprocess, time
LOG_DIR = "/kaggle/working/logs"
BURNER = f"{LOG_DIR}/tpu_burner.py"
OK_FILE = f"{LOG_DIR}/tpu_burner.ok"
PID_FILE = "/tmp/tpu_burner.pid"
BURNER_LOG = f"{LOG_DIR}/tpu_burner.log"
WATCHDOG_LOG = f"{LOG_DIR}/tpu_watchdog.log"
PY = "/usr/local/bin/python3"
STALE_SEC = 180
FIRST_OK_STALE_SEC = 600
CHECK_SEC = 60
def log(msg):
    line = "[tpu-watchdog " + time.strftime("%H:%M:%S") + "] " + str(msg)
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")
def read_pid():
    try:
        return int(open(PID_FILE).read().strip())
    except Exception:
        return None
def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
def stop_pid(pid):
    if not pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(10)
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
def start_burner(reason):
    old = read_pid()
    if old:
        stop_pid(old)
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    logf = open(BURNER_LOG, "ab", buffering=0)
    proc = subprocess.Popen([PY, BURNER], stdout=logf, stderr=subprocess.STDOUT, env=env, cwd="/kaggle/working")
    open(PID_FILE, "w").write(str(proc.pid))
    log("started burner pid={} reason={}".format(proc.pid, reason))
start_burner("initial")
while True:
    time.sleep(CHECK_SEC)
    pid = read_pid()
    age = None
    if os.path.exists(OK_FILE):
        age = time.time() - os.path.getmtime(OK_FILE)
    if not pid_alive(pid):
        start_burner("pid_dead")
    elif age is None:
        log("waiting for first ok pid={}".format(pid))
        if os.path.exists(BURNER_LOG) and os.path.getsize(BURNER_LOG) > 0 and time.time() - os.path.getmtime(BURNER_LOG) > FIRST_OK_STALE_SEC:
            start_burner("no_ok_stale_log")
    elif age > STALE_SEC:
        start_burner("ok_stale_{:.0f}s".format(age))
    else:
        log("healthy pid={} ok_age={:.0f}s".format(pid, age))
'''
    open(f"{LOG_DIR}/tpu_watchdog.py", "w").write(watchdog_py)
    wd_log = open(f"{LOG_DIR}/tpu_watchdog.stdout", "ab", buffering=0)
    wd = subprocess.Popen(
        ["/usr/local/bin/python3", f"{LOG_DIR}/tpu_watchdog.py"],
        stdout=wd_log,
        stderr=subprocess.STDOUT,
        cwd="/kaggle/working",
    )
    open("/tmp/tpu_watchdog.pid", "w").write(str(wd.pid))
    print(f"[*] {now_hms()} tpu-watchdog pid={wd.pid}")
    wait_for_tpu_burner(wd)


def wait_for_tpu_burner(wd_proc):
    deadline = time.time() + 600
    seen = set()
    while time.time() < deadline:
        if wd_proc.poll() is not None:
            raise RuntimeError("TPU watchdog died early")
        ok = f"{LOG_DIR}/tpu_burner.ok"
        if os.path.exists(ok):
            try:
                rec = json.load(open(ok))
                seen.add(int(rec.get("chunk", 0)))
                if len(seen) >= 2:
                    print(
                        f"[*] {now_hms()} TPU burner verified chunks={sorted(seen)[-2:]} "
                        f"elapsed={rec.get('elapsed_sec')}s loss={rec.get('loss')}"
                    )
                    run(f"tail -20 {LOG_DIR}/tpu_burner.log", check=False)
                    run(f"tail -10 {LOG_DIR}/tpu_watchdog.log", check=False)
                    return
            except Exception as e:
                print(f"[*] {now_hms()} waiting for TPU burner ok parse: {e}")
        time.sleep(10)
    run(f"tail -80 {LOG_DIR}/tpu_burner.log", check=False)
    run(f"tail -80 {LOG_DIR}/tpu_watchdog.log", check=False)
    raise RuntimeError("TPU burner did not produce two chunks within 10 minutes")


def run_phase2():
    t0 = time.time()
    os.chdir(os.environ["NPUCONVERT_DIR"])
    run(
        ". .venv/bin/activate && /usr/bin/time -v python gen_quant_data_sdxl.py",
        log=f"{LOG_DIR}/phase2.log",
    )
    print(f"[*] {now_hms()} Phase2 elapsed={int(time.time() - t0)}s")


def run_phase3():
    t0 = time.time()
    os.chdir(os.environ["NPUCONVERT_DIR"])
    run(
        '. .venv/bin/activate && /usr/bin/time -v python export_onnx_sdxl.py --model_path "$MODEL_PATH"',
        log=f"{LOG_DIR}/phase3.log",
    )
    print(f"[*] {now_hms()} Phase3 elapsed={int(time.time() - t0)}s")
    run('find . -name "*.onnx" -exec ls -lh {} \\;', check=False)
    run('find . -name "weights.pb" -exec ls -lh {} \\;', check=False)


RUN_STAGE = r'''
stage_mark() {
  local event="$1"; shift
  local stage="$1"; shift
  local cmd="${1:-}"
  local ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[[QNN_STAGE]] ts=${ts} event=${event} stage=${stage} cmd=${cmd} shell_pid=$$"
  if [ -n "${QNN_STAGE_TSV:-}" ]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$ts" "$event" "$stage" "$cmd" "$$" >> "$QNN_STAGE_TSV"
  fi
  if [ -n "${QNN_CURRENT_STAGE:-}" ]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$ts" "$event" "$stage" "$cmd" "$$" > "$QNN_CURRENT_STAGE"
  fi
}
run_stage() {
  local stage="$1"; shift
  local cmd
  printf -v cmd '%q ' "$@"
  stage_mark start "$stage" "$cmd"
  set +e
  "$@"
  local rc=$?
  set -e
  stage_mark end "$stage" "$cmd rc=${rc}"
  return "$rc"
}
'''


def patch_convert_scripts():
    npu = os.environ["NPUCONVERT_DIR"]
    qnn = os.environ["QNN_SDK_ROOT_DIR"]
    soc = os.environ["MIN_SOC"]
    for path in glob.glob(f"{npu}/scripts/convert_*.sh"):
        orig = open(path).read()
        s = re.sub(r"QNN_SDK_ROOT=/data/qairt/[0-9.]+", f"QNN_SDK_ROOT={qnn}", orig)
        if "stage_mark()" not in s:
            s = s.replace("set -e\n", f"set -e\ncd \"{npu}\"\n{RUN_STAGE}\n", 1)
        name = os.path.basename(path).removeprefix("convert_").removesuffix("_sdxl.sh")
        if not path.endswith("convert_all_sdxl.sh"):
            out = []
            cp_i = 0
            for raw in s.splitlines():
                stripped = raw.strip()
                if stripped.startswith("qairt-converter "):
                    raw = f'run_stage "{name}/qairt-converter" {stripped}'
                elif stripped.startswith("qairt-quantizer "):
                    raw = f'run_stage "{name}/qairt-quantizer" {stripped}'
                elif stripped.startswith("qnn-context-binary-generator "):
                    raw = f'run_stage "{name}/qnn-context-binary-generator" {stripped}'
                elif stripped.startswith("./MNNConvert "):
                    raw = f'run_stage "{name}/MNNConvert" {stripped}'
                elif stripped.startswith("cp "):
                    cp_i += 1
                    raw = f'run_stage "{name}/cp-{cp_i}" {stripped}'
                out.append(raw)
            s = "\n".join(out) + "\n"
        open(path, "w").write(s)
        print(f"[*] {now_hms()} patched {os.path.basename(path)}")
    hbp = f"{npu}/htp_backend_{soc}.json"
    assert os.path.exists(hbp), f"{hbp} missing"
    d = json.load(open(hbp))
    d["backend_extensions"]["config_file_path"] = f"{npu}/htp_config_{soc}.json"
    json.dump(d, open(hbp, "w"), indent=2)


class QnnSupervisor:
    drop_keywords = ["Failed to set thread affinity for cpuset"]
    print_keywords = [
        "[[QNN_STAGE]]",
        "INFO_CONVERSION_SUCCESS",
        "Quantized Model saved at",
        "INFO_WRITE_SUCCESS",
        "Converted Success!",
        "Model larger than 2GB",
        "Elapsed (wall clock)",
        "Maximum resident set size",
        "ERROR",
        "Error",
        "error:",
        "Traceback",
        "Cannot open",
        "not found",
        "No such file",
        "Killed",
        "Aborted",
        "Segmentation fault",
        "context-binary",
        "qnn-context-binary-generator",
        "qairt-quantizer",
        "qairt-converter",
    ]

    def __init__(self, label):
        self.label = label
        self.npu = os.environ["NPUCONVERT_DIR"]
        self.soc = os.environ["MIN_SOC"]
        self.phase_log = f"{LOG_DIR}/{label}.log"
        self.affinity_log = f"{LOG_DIR}/{label}.affinity.sample.log"
        self.stage_tsv = f"{LOG_DIR}/{label}.qnn_stage.tsv"
        self.current_stage_file = f"{LOG_DIR}/{label}.qnn_current_stage.tsv"
        self.status_tsv = f"{LOG_DIR}/{label}.qnn_status.tsv"
        self.artifact_tsv = f"{LOG_DIR}/{label}.qnn_artifacts.tsv"
        self.progress_tsv = f"{LOG_DIR}/{label}.qnn_progress.tsv"
        self.last_milestone = "starting"
        self.suppressed_affinity = 0
        self.printed_lines = 0
        self.exec_start = 0
        self.exec_end = 0
        self.last_status_exec_end = 0
        self.stage_input_cache = {}
        for p in [self.phase_log, self.affinity_log, self.stage_tsv, self.current_stage_file]:
            open(p, "w").close()
        open(self.status_tsv, "w").write(
            "epoch\tts\telapsed_s\tmem_avail_mb\tdisk_avail_mb\tphase_log_mib\tstage\tproc\ttpu\t"
            "suppressed_affinity\taffinity_rate_per_min\twarnings\tlast_signal\n"
        )
        open(self.artifact_tsv, "w").write("epoch\tts\tstage\tout_files\tout_mib\tkey_files\n")
        open(self.progress_tsv, "w").write(
            "epoch\tts\tstage\tstage_age_s\tinput_list\tinput_rows\texec_start\texec_end\t"
            "exec_delta_per_min\tpass_est\tlast_qnn_ms\n"
        )

    def dropped_count(self, line):
        total = 0
        for keyword in self.drop_keywords:
            if keyword in line:
                total += max(1, line.count(keyword))
        if total == 0 and re.fullmatch(r"(WARNING:\s*)+", line.strip()):
            total = 1
        return total

    def handle_line(self, line, logf, afff):
        clean = line.rstrip()
        if "[QNN_CPU] QnnGraph execute start" in clean:
            self.exec_start += 1
        if "[QNN_CPU] QnnGraph execute end" in clean:
            self.exec_end += 1
        drop_n = self.dropped_count(clean)
        if drop_n:
            before = self.suppressed_affinity
            self.suppressed_affinity += drop_n
            if before < 20 or self.suppressed_affinity // 100000 != before // 100000:
                afff.write(f"{now_hms()} count={self.suppressed_affinity} delta={drop_n} {clean[:1000]}\n")
                afff.flush()
            return
        logf.write(line)
        logf.flush()
        if clean:
            self.last_milestone = clean[-240:]
        if any(k in clean for k in self.print_keywords):
            print(clean[:1200], flush=True)
            self.printed_lines += 1

    def tpu_status(self):
        ok = f"{LOG_DIR}/tpu_burner.ok"
        if not os.path.exists(ok):
            return "missing"
        age = time.time() - os.path.getmtime(ok)
        try:
            rec = json.load(open(ok))
            return f"age={int(age)}s chunk={rec.get('chunk')} elapsed={rec.get('elapsed_sec')}"
        except Exception:
            return f"age={int(age)}s parse_error"

    def tail_line(self, path):
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 4096))
                lines = f.read().decode("utf-8", "replace").splitlines()
                return lines[-1] if lines else ""
        except Exception:
            return ""

    def current_stage(self):
        try:
            raw = open(self.current_stage_file).read().strip()
            if not raw:
                return "missing", None, None, None, ""
            parts = raw.split("\t")
            ts, event, stage, cmd = (parts + ["?", "?", "?", "?"])[:4]
            age = time.time() - os.path.getmtime(self.current_stage_file)
            return f"{event}:{stage} age={int(age)}s cmd={cmd}", event, stage, age, cmd
        except Exception as e:
            return f"error:{e}", None, None, None, ""

    def qnn_proc_stats(self):
        keys = ("qnn-context-binary-generator", "qairt-quantizer", "qairt-converter", "MNNConvert")
        try:
            out = subprocess.check_output(
                ["ps", "-eo", "pid,ppid,etimes,pcpu,rss,vsz,comm,args"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            return f"ps_error:{e}"
        rows = []
        for line in out.splitlines()[1:]:
            if not any(k in line for k in keys):
                continue
            parts = line.split(None, 7)
            if len(parts) < 8:
                continue
            pid, _ppid, etimes, pcpu, rss, vsz, comm, _args = parts
            try:
                rss_mib = int(rss) // 1024
                vsz_mib = int(vsz) // 1024
            except ValueError:
                rss_mib = vsz_mib = -1
            rows.append(f"{comm} pid={pid} age={etimes}s cpu={pcpu}% rss={rss_mib}MiB vsz={vsz_mib}MiB")
        return "; ".join(rows[:4]) if rows else "none"

    def burner_proc_stats(self):
        try:
            out = subprocess.check_output(["ps", "-eo", "pid,etimes,pcpu,rss,comm,args"], text=True)
        except Exception:
            return "none"
        rows = []
        for line in out.splitlines()[1:]:
            if "tpu_burner.py" not in line and "tpu_watchdog.py" not in line:
                continue
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            pid, etimes, pcpu, rss, comm, _args = parts
            rows.append(f"{comm} pid={pid} age={etimes}s cpu={pcpu}% rss={int(rss)//1024}MiB")
        return "; ".join(rows) if rows else "none"

    def artifact_summary(self):
        out_dir = f"{self.npu}/output/qnn_models_sdxl_{self.soc}"
        files, total = 0, 0
        if os.path.isdir(out_dir):
            for root, _, names in os.walk(out_dir):
                for name in names:
                    p = os.path.join(root, name)
                    try:
                        files += 1
                        total += os.path.getsize(p)
                    except OSError:
                        pass
        keys = []
        for pat in [f"{self.npu}/*_sdxl/model.dlc", f"{self.npu}/*_sdxl/model_quantized.dlc", f"{out_dir}/*"]:
            for p in sorted(glob.glob(pat)):
                if os.path.isfile(p):
                    keys.append(f"{os.path.relpath(p, self.npu)}:{file_mib(p):.1f}MiB")
        return files, total / (1024 * 1024), ";".join(keys[-30:])

    def input_list_summary(self, stage, cmd):
        m = re.search(r"--input_list\s+(\S+)", cmd or "")
        if not m:
            return "", 0
        raw = m.group(1).strip("'\"")
        path = raw if os.path.isabs(raw) else os.path.join(self.npu, raw)
        if path in self.stage_input_cache:
            return self.stage_input_cache[path]
        rows = 0
        try:
            for line in open(path):
                s = line.strip()
                if s and not s.startswith("#"):
                    rows += 1
        except Exception:
            rows = 0
        self.stage_input_cache[path] = (raw, rows)
        return raw, rows

    def progress_record(self, stage_status, stage_age, cmd, now, last_affinity_time):
        input_list, rows = self.input_list_summary(stage_status, cmd)
        exec_delta = self.exec_end - self.last_status_exec_end
        self.last_status_exec_end = self.exec_end
        exec_rate = 60.0 * exec_delta / max(1.0, now - last_affinity_time)
        pass_est = (self.exec_end / rows) if rows else 0.0
        last_ms = ""
        m = re.search(r"([0-9.]+)ms", self.last_milestone)
        if m:
            last_ms = m.group(1)
        with open(self.progress_tsv, "a") as pf:
            pf.write(
                "\t".join(
                    map(
                        tsv_escape,
                        [
                            int(now),
                            now_hms(),
                            stage_status,
                            int(stage_age or 0),
                            input_list,
                            rows,
                            self.exec_start,
                            self.exec_end,
                            f"{exec_rate:.2f}",
                            f"{pass_est:.3f}",
                            last_ms,
                        ],
                    )
                )
                + "\n"
            )
        return f"input_rows={rows} exec={self.exec_end}/{self.exec_start} pass_est={pass_est:.2f} exec_rate={exec_rate:.1f}/min"

    def run(self, command, required_outputs=None):
        os.environ["QNN_STAGE_TSV"] = self.stage_tsv
        os.environ["QNN_CURRENT_STAGE"] = self.current_stage_file
        env = os.environ.copy()
        for k in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"]:
            env.pop(k, None)
        print(f"[*] {now_hms()} QNN supervisor start label={self.label}")
        proc = subprocess.Popen(
            ["bash", "-lc", "set -o pipefail; " + command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=self.npu,
            env=env,
        )
        t0 = time.time()
        last_status = 0.0
        last_affinity_count = 0
        last_affinity_time = time.time()
        with open(self.phase_log, "a", buffering=1) as logf, open(self.affinity_log, "a", buffering=1) as afff:
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 1)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        self.handle_line(line, logf, afff)
                    elif proc.poll() is not None:
                        break
                if proc.poll() is not None:
                    for line in proc.stdout:
                        self.handle_line(line, logf, afff)
                    break
                now = time.time()
                if now - last_status >= 60:
                    stage_status, _stage_event, _stage_name, stage_age, cmd = self.current_stage()
                    proc_status = self.qnn_proc_stats()
                    phase_mib = file_mib(self.phase_log)
                    affinity_rate = 60.0 * (self.suppressed_affinity - last_affinity_count) / max(
                        1.0, now - last_affinity_time
                    )
                    progress = self.progress_record(stage_status, stage_age, cmd, now, last_affinity_time)
                    last_affinity_count = self.suppressed_affinity
                    last_affinity_time = now
                    files, out_mib, key_files = self.artifact_summary()
                    warn = []
                    if disk_avail_mb() < 4096:
                        warn.append("low_disk_free")
                    warn_text = ",".join(warn) if warn else "none"
                    wd_tail = self.tail_line(f"{LOG_DIR}/tpu_watchdog.log")[-180:]
                    msg = (
                        f"[*] {now_hms()} QNN running label={self.label} elapsed={int(now - t0)}s "
                        f"log={phase_mib:.1f}MiB disk_free={disk_avail_mb()}MiB mem_avail={mem_avail_mb()}MiB "
                        f"tpu_ok={self.tpu_status()} stage=\"{stage_status}\" proc=\"{proc_status}\" "
                        f"burner=\"{self.burner_proc_stats()}\" progress=\"{progress}\" "
                        f"artifacts=\"files={files} mib={out_mib:.1f} keys={key_files[-700:]}\" "
                        f"suppressed_affinity={self.suppressed_affinity} affinity_rate={affinity_rate:.0f}/min "
                        f"warnings={warn_text} last_signal=\"{self.last_milestone[-200:]}\" watchdog=\"{wd_tail}\""
                    )
                    print(msg.replace('"', "'"), flush=True)
                    with open(self.status_tsv, "a") as sf:
                        sf.write(
                            "\t".join(
                                map(
                                    tsv_escape,
                                    [
                                        int(now),
                                        now_hms(),
                                        int(now - t0),
                                        mem_avail_mb(),
                                        disk_avail_mb(),
                                        f"{phase_mib:.1f}",
                                        stage_status,
                                        proc_status,
                                        self.tpu_status(),
                                        self.suppressed_affinity,
                                        f"{affinity_rate:.1f}",
                                        warn_text,
                                        self.last_milestone,
                                    ],
                                )
                            )
                            + "\n"
                        )
                    with open(self.artifact_tsv, "a") as af:
                        af.write(
                            "\t".join(
                                map(
                                    tsv_escape,
                                    [int(now), now_hms(), stage_status, files, f"{out_mib:.1f}", key_files],
                                )
                            )
                            + "\n"
                        )
                    last_status = now
        rc = proc.wait()
        print(
            f"[*] {now_hms()} QNN supervisor exit label={self.label} rc={rc} "
            f"printed={self.printed_lines} suppressed_affinity={self.suppressed_affinity} "
            f"exec_start={self.exec_start} exec_end={self.exec_end}"
        )
        if rc != 0:
            run(f"tail -120 {self.phase_log}", check=False)
            raise RuntimeError(f"QNN command failed rc={rc}: {command[:240]}")
        if required_outputs:
            missing = [p for p in required_outputs if not os.path.exists(p)]
            if missing:
                raise RuntimeError("QNN command completed but required outputs missing:\n" + "\n".join(missing))


def prepare_qnn_runtime():
    npu = os.environ["NPUCONVERT_DIR"]
    pylib = subprocess.check_output(
        [".venv/bin/python", "-c", 'import os,sys; print(os.path.join(sys.base_prefix, "lib"))'],
        cwd=npu,
        text=True,
    ).strip()
    assert os.path.exists(os.path.join(pylib, "libpython3.10.so.1.0")), f"libpython missing in {pylib}"
    os.environ["LD_LIBRARY_PATH"] = f"{pylib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    qnn_lib = f"{os.environ['QNN_SDK_ROOT_DIR']}/lib/x86_64-linux-clang"
    os.environ["LD_LIBRARY_PATH"] = f"{qnn_lib}:{os.environ['LD_LIBRARY_PATH']}"
    print(f"[*] {now_hms()} LD_LIBRARY_PATH ready")


def qnn_driver_command(scripts):
    npu = os.environ["NPUCONVERT_DIR"]
    qnn = os.environ["QNN_SDK_ROOT_DIR"]
    qnn_bin = os.environ["QNN_SDK_BIN"]
    lines = [
        f'cd "{npu}"',
        f'export QNN_SDK_ROOT="{qnn}"',
        f'source "{qnn_bin}/envsetup.sh"',
    ]
    for script in scripts:
        lines.append(f'bash "scripts/{script}" --min_soc "$MIN_SOC"')
    return " && ".join(lines)


def referenced_input_files(input_list_path, base_dir):
    deps = []
    rows = 0
    missing = []
    for line in open(input_list_path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows += 1
        for tok in re.split(r"\s+", s):
            val = tok
            if ":=" in val:
                val = val.split(":=", 1)[1]
            elif "=" in val:
                val = val.split("=", 1)[1]
            val = val.strip("'\"")
            if not val or val.startswith("[") or val.startswith("{"):
                continue
            path = val if os.path.isabs(val) else os.path.join(base_dir, val)
            if os.path.isfile(path):
                deps.append(os.path.abspath(path))
            elif "/" in val or "." in os.path.basename(val):
                missing.append(val)
    return rows, sorted(set(deps)), missing


def copy_preserve(src, dst_root, base_dir):
    src_abs = os.path.abspath(src)
    try:
        rel = os.path.relpath(src_abs, base_dir)
        if rel.startswith(".."):
            rel = os.path.join("_external", os.path.basename(src_abs))
    except ValueError:
        rel = os.path.join("_external", os.path.basename(src_abs))
    dst = os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src_abs, dst)
    return rel, os.path.getsize(src_abs)


def create_b1_bundle():
    npu = os.environ["NPUCONVERT_DIR"]
    soc = os.environ["MIN_SOC"]
    bundle = "/kaggle/working/b1_bundle"
    if os.path.exists(bundle):
        shutil.rmtree(bundle)
    os.makedirs(bundle)
    output_dir = f"{npu}/output/qnn_models_sdxl_{soc}"
    assert os.path.isdir(output_dir), f"{output_dir} missing"
    shutil.copytree(output_dir, f"{bundle}/output/qnn_models_sdxl_{soc}")
    for rel in [
        "unet_sdxl/model.onnx",
        "input_list_unet_sdxl.txt",
        f"htp_backend_{soc}.json",
        f"htp_config_{soc}.json",
        "tokenizer.json",
    ]:
        src = f"{npu}/{rel}"
        assert os.path.exists(src), f"required B1 bundle file missing: {src}"
        os.makedirs(os.path.dirname(f"{bundle}/{rel}"), exist_ok=True)
        shutil.copy2(src, f"{bundle}/{rel}")
    rows, deps, missing = referenced_input_files(f"{npu}/input_list_unet_sdxl.txt", npu)
    dep_bytes = 0
    dep_rels = []
    for dep in deps:
        rel, size = copy_preserve(dep, bundle, npu)
        dep_bytes += size
        dep_rels.append(rel)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": os.environ["MODEL_NAME"],
        "min_soc": soc,
        "input_list_rows": rows,
        "dependency_count": len(dep_rels),
        "dependency_mib": round(dep_bytes / (1024 * 1024), 1),
        "dependencies": dep_rels[:2000],
        "missing_tokens": missing[:200],
    }
    json.dump(manifest, open(f"{bundle}/manifest.json", "w"), indent=2)
    print(f"[*] {now_hms()} B1 bundle manifest: {json.dumps(manifest)[:1200]}")
    run(f"du -sh {bundle}/*", check=False)


def locate_b1_bundle():
    hits = glob.glob("/kaggle/input/**/b1_bundle/manifest.json", recursive=True)
    assert hits, "b1_bundle/manifest.json not found in /kaggle/input"
    bundle = os.path.dirname(hits[0])
    manifest = json.load(open(hits[0]))
    print(f"[*] {now_hms()} B1 bundle={bundle} manifest={json.dumps(manifest)[:1200]}")
    return bundle, manifest


def restore_b1_bundle_to_npu():
    npu = os.environ["NPUCONVERT_DIR"]
    bundle, manifest = locate_b1_bundle()
    for name in os.listdir(bundle):
        src = os.path.join(bundle, name)
        dst = os.path.join(npu, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    soc = os.environ["MIN_SOC"]
    required = [
        f"{npu}/unet_sdxl/model.onnx",
        f"{npu}/input_list_unet_sdxl.txt",
        f"{npu}/htp_backend_{soc}.json",
        f"{npu}/htp_config_{soc}.json",
        f"{npu}/output/qnn_models_sdxl_{soc}/vae_encoder.bin",
        f"{npu}/output/qnn_models_sdxl_{soc}/vae_decoder.bin",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    assert not missing, "B1 restore missing files:\n" + "\n".join(missing)
    rows, deps, missing_tokens = referenced_input_files(f"{npu}/input_list_unet_sdxl.txt", npu)
    if missing_tokens:
        print(f"[*] {now_hms()} input_list unresolved tokens sample={missing_tokens[:20]}")
    print(f"[*] {now_hms()} restored B1 bundle rows={rows} deps={len(deps)}")
    return manifest


def package_final_zip():
    npu = os.environ["NPUCONVERT_DIR"]
    soc = os.environ["MIN_SOC"]
    out_dir = f"{npu}/output/qnn_models_sdxl_{soc}"
    assert os.path.isdir(out_dir), f"{out_dir} missing"
    assert glob.glob(f"{out_dir}/unet*"), "UNet output not found in final output dir"
    run(f'touch "{out_dir}/SDXL"')
    zip_name = f"{os.environ['MODEL_NAME']}_qnn2.28_{soc}.zip"
    run("df -h /kaggle/working", check=False)
    run(f'zip -r "/kaggle/working/{zip_name}" "{out_dir}" > "{LOG_DIR}/package_zip.log"')
    run(f'ls -lh "/kaggle/working/{zip_name}"')
    run("df -h /kaggle/working", check=False)


def stop_background_services():
    for pidfile, name in [
        ("/tmp/tpu_watchdog.pid", "tpu-watchdog"),
        ("/tmp/tpu_burner.pid", "tpu-burner"),
        ("/tmp/watcher.pid", "mem-watcher"),
    ]:
        try:
            pid = int(open(pidfile).read().strip())
        except Exception:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[*] {now_hms()} SIGTERM {name} pid={pid}")
        except ProcessLookupError:
            continue


def common_bootstrap(model_name, min_soc, realistic, civitai_version_id=None):
    configure_env(model_name, min_soc, realistic, civitai_version_id)
    run("free -h", check=False)
    install_tools_and_convert_package()
    install_qnn_sdk()
    setup_python_env()
    start_background_services()


def run_b1(model_name, min_soc, realistic, civitai_version_id=None):
    common_bootstrap(model_name, min_soc, realistic, civitai_version_id)
    locate_phase1_output()
    copy_phase1_calibration_data()
    run_phase2()
    run_phase3()
    patch_convert_scripts()
    prepare_qnn_runtime()
    npu = os.environ["NPUCONVERT_DIR"]
    soc = os.environ["MIN_SOC"]
    required = [
        f"{npu}/output/qnn_models_sdxl_{soc}/clip.mnn",
        f"{npu}/output/qnn_models_sdxl_{soc}/clip_2.mnn",
        f"{npu}/output/qnn_models_sdxl_{soc}/clip_2.mnn.weight",
        f"{npu}/output/qnn_models_sdxl_{soc}/vae_encoder.bin",
        f"{npu}/output/qnn_models_sdxl_{soc}/vae_decoder.bin",
    ]
    scripts = [
        "convert_clip_sdxl.sh",
        "convert_clip2_sdxl.sh",
        "convert_vae_encoder_sdxl.sh",
        "convert_vae_decoder_sdxl.sh",
    ]
    QnnSupervisor("b1_phase45").run(qnn_driver_command(scripts), required_outputs=required)
    create_b1_bundle()
    stop_background_services()
    print(f"[*] {now_hms()} B1 done")


def run_b2(model_name, min_soc, realistic, civitai_version_id=None):
    common_bootstrap(model_name, min_soc, realistic, civitai_version_id)
    restore_b1_bundle_to_npu()
    patch_convert_scripts()
    prepare_qnn_runtime()
    npu = os.environ["NPUCONVERT_DIR"]
    soc = os.environ["MIN_SOC"]
    required = [f"{npu}/output/qnn_models_sdxl_{soc}/unet.bin"]
    QnnSupervisor("b2_unet").run(qnn_driver_command(["convert_unet_sdxl.sh"]), required_outputs=required)
    package_final_zip()
    stop_background_services()
    print(f"[*] {now_hms()} B2 done")
