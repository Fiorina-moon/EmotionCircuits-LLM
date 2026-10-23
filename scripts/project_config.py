# scripts/project_config.py
# -*- coding: utf-8 -*-
"""
统一的运行配置入口
Unified runtime configuration

配置来源（优先级从高到低）Priority (high → low):
    环境变量 > 仓库根目录 .env > config.yaml > 内置默认值

导出 Exports:
    MODEL              解析后的模型来源（本地目录或 HuggingFace ID）
    MODEL_NAME         输出子目录名，如 llama32_3b
    HF_TOKEN           HuggingFace token（本地模型时为 None）
    DEVICE / DTYPE     默认设备与精度字符串
    OUTPUT_ROOT        个人输出根目录（默认 outputs_local/）
    REPO_OUTPUT_ROOT   仓库自带产物根目录（默认 outputs/，只读回退）

工具函数 Helpers:
    out(*parts)        个人输出目录下的路径（写入用）
    src(*parts)        读取用：个人目录存在则用个人目录，否则回退到仓库目录
    ensure_out(*parts) 创建并返回个人输出目录
    maybe_login(model) 仅在远程模型 + 有 token 时登录
    hf_token(model)    本地模型返回 None
"""

import os
import sys
from pathlib import Path

import yaml

# Windows 控制台默认 GBK，统一改为 UTF-8，避免中文/符号输出报错或乱码
# Windows consoles default to GBK; switch to UTF-8 so Chinese/symbol output is safe
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config.yaml"
ENV_FILE = REPO_ROOT / ".env"


# ---------------- 配置加载 Configuration loading ----------------
def _load_env(path: Path):
    """最小的 .env 解析器（KEY=VALUE），不覆盖已存在的环境变量。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_config(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_load_env(ENV_FILE)
CFG = _load_config(CONFIG_FILE)

_model_cfg = CFG.get("model") or {}
_path_cfg = CFG.get("paths") or {}
_hf_cfg = CFG.get("huggingface") or {}

MODEL_NAME = str(_path_cfg.get("model_name") or "llama32_3b")


# ---------------- 模型 Model ----------------
def _abs(path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def is_local_model(model) -> bool:
    if not model:
        return False
    try:
        return _abs(model).is_dir()
    except OSError:
        return False


def resolve_model(model=None) -> str:
    """按 显式参数 > MODEL_PATH > config.model.local_path > MODEL_ID > hf_id 解析。"""
    for cand in (model, os.environ.get("MODEL_PATH"), _model_cfg.get("local_path")):
        if is_local_model(cand):
            return str(_abs(cand))
    return (
        os.environ.get("MODEL_ID")
        or _model_cfg.get("hf_id")
        or "meta-llama/Llama-3.2-3B-Instruct"
    )


MODEL = resolve_model()
DEVICE = str(os.environ.get("DEVICE") or _model_cfg.get("device") or "auto")
DTYPE = str(_model_cfg.get("dtype") or "float32")

HF_TOKEN = os.environ.get("HF_TOKEN") or None


def resolve_device() -> str:
    """把配置里的 auto 解析成具体设备：有 CUDA 用 cuda:0，否则 cpu。"""
    if DEVICE and DEVICE != "auto":
        return DEVICE
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch 不可用时退回 CPU
        return "cpu"

# 镜像地址需要在导入 transformers 之前设置
_hf_endpoint = os.environ.get("HF_ENDPOINT") or _hf_cfg.get("endpoint") or ""
if _hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", _hf_endpoint)


def hf_token(model=None):
    """本地模型返回 None，避免任何 HuggingFace 网络请求。"""
    if is_local_model(model if model is not None else MODEL):
        return None
    return HF_TOKEN


def maybe_login(model=None):
    """仅在远程模型 + 提供了 token + 配置允许时登录，失败不中断。"""
    model = model if model is not None else MODEL
    if is_local_model(model) or not _hf_cfg.get("login", True):
        return None
    token = HF_TOKEN
    if not token:
        return None
    try:
        from huggingface_hub import login

        login(token=token)
    except Exception as exc:  # noqa: BLE001 - 登录失败不应阻塞本地缓存推理
        print(f"[WARN] HuggingFace 登录失败，继续使用本地缓存 Login failed: {exc}")
    return token


# ---------------- 路径 Paths ----------------
OUTPUT_ROOT = _abs(os.environ.get("OUTPUT_DIR") or _path_cfg.get("output_dir") or "outputs_local")
REPO_OUTPUT_ROOT = _abs(_path_cfg.get("repo_output_dir") or "outputs")
QUICK_START_OUTPUT_DIR = _abs(
    os.environ.get("QUICK_START_OUTPUT_DIR")
    or _path_cfg.get("quick_start_output_dir")
    or "quick_start_output_local"
)


def out(*parts) -> Path:
    """个人输出目录下的路径（写入用）。"""
    return OUTPUT_ROOT.joinpath(*[str(p) for p in parts])


def src(*parts) -> Path:
    """读取用：个人目录存在则用个人目录，否则回退到仓库目录。"""
    personal = out(*parts)
    if personal.exists():
        return personal
    repo = REPO_OUTPUT_ROOT.joinpath(*[str(p) for p in parts])
    return repo if repo.exists() else personal


def ensure_out(*parts) -> Path:
    """创建并返回个人输出目录。"""
    path = out(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe() -> str:
    """打印当前配置（调试用）。"""
    return (
        f"MODEL={MODEL}\n"
        f"OUTPUT_ROOT={OUTPUT_ROOT}\n"
        f"REPO_OUTPUT_ROOT={REPO_OUTPUT_ROOT}\n"
        f"DEVICE={DEVICE} DTYPE={DTYPE}"
    )
