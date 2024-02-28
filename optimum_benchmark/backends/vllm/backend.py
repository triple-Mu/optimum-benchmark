import gc
from collections import OrderedDict
from logging import getLogger
from typing import Any, Dict

import torch
from ..base import Backend
from .config import vLLMConfig

# bachend logger
LOGGER = getLogger("vllm")


class vLLMBackend(Backend[vLLMConfig]):
    NAME = "vllm"

    def __init__(self, config: vLLMConfig):
        super().__init__(config)
        self.config.library = 'vllm'
        self.validate_library()
        self.dtype = getattr(torch, config.torch_dtype, torch.float32)
        # Thread settings
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch inter_op_num_threads({self.config.inter_op_num_threads}))")
            torch.set_num_threads(self.config.inter_op_num_threads)
        if self.config.intra_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch intra_op_num_threads({self.config.intra_op_num_threads}))")
            torch.set_num_interop_threads(self.config.intra_op_num_threads)

        assert not self.config.no_weights
        LOGGER.info("\t+ Loading model with pretrained weights")
        self.load_model_from_pretrained()

    def validate_library(self) -> None:
        if self.config.library == "vllm":
            from vllm import LLM, SamplingParams
            class WarpLLM(LLM):
                sampling_params: SamplingParams = SamplingParams(temperature=0.8, top_p=0.95)

                def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
                    self.sampling_params.max_tokens = kwargs.get('max_new_tokens', 2)
                    # kwargs.get('min_new_tokens', 2)
                    outputs = super().generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=input_ids.tolist(),
                        prefix_pos=None,
                        use_tqdm=False,
                        lora_request=None)
                    outputs = torch.tensor([o.outputs[0].token_ids for o in outputs]).to(input_ids)
                    return torch.cat([input_ids, outputs], dim=1)

            WarpLLM.sampling_params.seed = self.config.seed

            self.automodel_class: type = WarpLLM
            LOGGER.info(f"\t+ Using vLLM method {self.automodel_class.__name__}")
        else:
            raise ValueError(f"Library {self.config.library} not supported")

    def load_model_from_pretrained(self) -> None:
        LOGGER.info(f"\t+ Loading model directly on device: {self.config.device}")
        with torch.device(self.config.device):
            self.pretrained_model = self.automodel_class(model=self.config.model, **self.automodel_kwargs)

    @property
    def is_quantized(self) -> bool:
        return self.config.quantization_scheme is not None

    @property
    def is_awq_quantized(self) -> bool:
        return self.config.quantization_scheme == "awq"

    @property
    def is_gptq_quantized(self) -> bool:
        return self.config.quantization_scheme == "gptq"

    @property
    def is_squeezellm_quantized(self) -> bool:
        return self.config.quantization_scheme == "squeezellm"

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {'trust_remote_code': True}

        if self.is_quantized:
            kwargs["quantization"] = self.config.quantization_scheme

        if self.config.torch_dtype is not None:
            kwargs["dtype"] = getattr(torch, self.config.torch_dtype)

        return kwargs

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super().prepare_inputs(inputs)
        for key, value in inputs.items():
            inputs[key] = value.to(self.config.device)
        return inputs

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.dtype, enabled=True):
            return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.dtype, enabled=True):
            return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def call(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def seed(self):
        super().seed()
        torch.manual_seed(self.config.seed)

        if self.config.device == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

    def clean(self) -> None:
        super().clean()
        gc.collect()
