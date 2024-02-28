from dataclasses import dataclass
from typing import Optional
import importlib.metadata
import importlib.util
from ..config import BackendConfig

DEVICE_MAPS = ["auto", "sequential"]
AMP_DTYPES = ["bfloat16", "float16"]
TORCH_DTYPES = ["bfloat16", "float16", "float32", "auto"]

QUANTIZATION_CONFIGS = {"squeezellm", "gptq", "awq"}


def vllm_version():
    if importlib.util.find_spec("vllm") is not None:
        return importlib.metadata.version("vllm")


@dataclass
class vLLMConfig(BackendConfig):
    name: str = "vllm"
    version: Optional[str] = vllm_version()
    _target_: str = "optimum_benchmark.backends.vllm.backend.vLLMBackend"

    # load options
    no_weights: bool = False
    torch_dtype: Optional[str] = None

    # quantization options
    quantization_scheme: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.torch_dtype is not None and self.torch_dtype not in TORCH_DTYPES:
            raise ValueError(f"`torch_dtype` must be one of {TORCH_DTYPES}. Got {self.torch_dtype} instead.")

        if self.quantization_scheme is not None:
            if self.quantization_scheme not in QUANTIZATION_CONFIGS:
                raise ValueError(
                    f"`quantization_scheme` must be one of {list(QUANTIZATION_CONFIGS)}. Got {self.quantization_scheme} instead."
                )
