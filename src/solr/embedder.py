"""OrQA's embedding server: a user-level Ollama, on CPU, serving the model
the Solr cores' vectors come from (``bge-m3``).

The questions OrQA's gate ranks and the tables in the cores must be
embedded by the same model, and LakeGen embeds its questions through Ollama
too (``lakegen.llm.embeddings.OllamaMultilingualEmbedding``), so one server
gives both projects the same vectors. It is not the machine's shared Ollama
service (port 11434), whose models folder is on the full root disk: this one
keeps its models under ``ORQA_OLLAMA_MODELS``.

It never touches the GPU, which other users' jobs fill: CUDA, ROCm and
Vulkan devices are all hidden (with CUDA alone hidden, Ollama 0.35 found the
RTX 4090 through Vulkan and loaded the model there).

Settings (environment or ``.env``):
    ORQA_OLLAMA_MODELS  the models folder; required.
    ORQA_OLLAMA_PORT    its port, default 11435.
    OLLAMA_BIN          the ollama binary, default /usr/local/bin/ollama.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

DEFAULT_PORT = 11435
DEFAULT_BIN = Path("/usr/local/bin/ollama")
# Every GPU runtime Ollama's runners can use.
_NO_GPU = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "GGML_VK_VISIBLE_DEVICES": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "ROCR_VISIBLE_DEVICES": "-1",
}


class EmbedderError(RuntimeError):
    """OrQA's embedding server is not set up, not running, or not OrQA's."""


def default_url() -> str:
    port = int(os.environ.get("ORQA_OLLAMA_PORT") or DEFAULT_PORT)
    return f"http://127.0.0.1:{port}"


class OllamaServer:
    """OrQA's Ollama: its process and its models.

    Args:
        models: The models folder (``ORQA_OLLAMA_MODELS``).
        port: The HTTP port (``ORQA_OLLAMA_PORT``).
        binary: The ollama binary.
    """

    def __init__(self, models: Path, port: int = DEFAULT_PORT, binary: Path = DEFAULT_BIN) -> None:
        self.models = Path(models).expanduser().resolve()
        self.port = int(port)
        self.binary = Path(binary)

    @classmethod
    def from_env(cls) -> "OllamaServer":
        models = os.environ.get("ORQA_OLLAMA_MODELS", "").strip()
        if not models:
            raise EmbedderError(
                "ORQA_OLLAMA_MODELS is not set: point it (e.g. in .env) at the folder "
                "OrQA's Ollama keeps its models in, on a disk with room for them."
            )
        return cls(
            Path(models),
            port=int(os.environ.get("ORQA_OLLAMA_PORT") or DEFAULT_PORT),
            binary=Path(os.environ.get("OLLAMA_BIN") or DEFAULT_BIN),
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def _host(self) -> str:
        return f"127.0.0.1:{self.port}"

    @property
    def pid_file(self) -> Path:
        return self.models.parent / f"orqa-ollama-{self.port}.pid"

    @property
    def log_file(self) -> Path:
        return self.models.parent / f"orqa-ollama-{self.port}.log"

    def _env(self) -> dict[str, str]:
        return {
            **os.environ,
            "OLLAMA_HOST": self._host,
            "OLLAMA_MODELS": str(self.models),
            # The model stays loaded: the gate embeds one question at a time.
            "OLLAMA_KEEP_ALIVE": "-1",
            # Other models in the folder are not this server's to delete.
            "OLLAMA_NOPRUNE": "1",
            **_NO_GPU,
        }

    def _pid(self) -> Optional[int]:
        """The pid of this server's process, when it is alive and is ours
        (an ``ollama serve`` with this models folder and host)."""
        try:
            pid = int(self.pid_file.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            environ = dict(
                item.split("=", 1)
                for item in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0")
                if "=" in item
            )
        except (OSError, ValueError):
            return None
        if b"serve" not in cmdline or not cmdline[0].endswith(b"ollama"):
            return None
        if environ.get("OLLAMA_MODELS") != str(self.models) or environ.get("OLLAMA_HOST") != self._host:
            return None
        return pid

    def _answers(self) -> bool:
        try:
            return requests.get(f"{self.url}/api/version", timeout=3).status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def running(self) -> bool:
        """Whether OrQA's server answers. Another server on the port is an
        error, not "not running"."""
        if not self._answers():
            return False
        if self._pid() is None:
            raise EmbedderError(
                f"an Ollama answers at {self.url} that OrQA did not start "
                f"(no live pid in {self.pid_file}): stop it or change ORQA_OLLAMA_PORT."
            )
        return True

    def start(self, wait: float = 60.0) -> bool:
        """Start the server unless it already runs; True when started."""
        if self.running():
            return False
        self.models.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "ab") as log:
            process = subprocess.Popen(
                [str(self.binary), "serve"], env=self._env(), stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        self.pid_file.write_text(f"{process.pid}\n")
        deadline = time.monotonic() + wait
        while not self._answers():
            if process.poll() is not None:
                raise EmbedderError(f"ollama serve exited ({process.returncode}): see {self.log_file}")
            if time.monotonic() > deadline:
                raise EmbedderError(f"ollama serve did not answer within {wait:.0f} s: see {self.log_file}")
            time.sleep(0.5)
        return True

    def stop(self) -> bool:
        """Stop the server; True when it was running."""
        pid = self._pid()
        if pid is None:
            return False
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.5)
        self.pid_file.unlink(missing_ok=True)
        return True

    def models_available(self) -> list[str]:
        response = requests.get(f"{self.url}/api/tags", timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def has_model(self, model: str) -> bool:
        names = set(self.models_available())
        return model in names or f"{model}:latest" in names

    def ensure(self, model: str) -> None:
        """Start the server and download ``model`` into it when missing."""
        self.start()
        if self.has_model(model):
            return
        if self._pid() is None:
            raise EmbedderError(f"refusing to pull {model!r}: {self.url} is not OrQA's Ollama")
        subprocess.run(
            [str(self.binary), "pull", model],
            env={**os.environ, "OLLAMA_HOST": self._host}, check=True,
        )
