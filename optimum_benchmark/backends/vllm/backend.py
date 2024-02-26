import gc
import os
from collections import OrderedDict
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import (
    AwqConfig,
    BitsAndBytesConfig,
    GPTQConfig,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainingArguments,
)


from ..base import Backend
from ..transformers_utils import random_init_weights
from .config import vLLMConfig


# bachend logger
LOGGER = getLogger("vllm")


class vLLMBackend(Backend[vLLMConfig]):
    NAME = "vllm"

    def __init__(self, config: vLLMConfig):
        super().__init__(config)
        self.validate_library()

        # Thread settings
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch inter_op_num_threads({self.config.inter_op_num_threads}))")
            torch.set_num_threads(self.config.inter_op_num_threads)
        if self.config.intra_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch intra_op_num_threads({self.config.intra_op_num_threads}))")
            torch.set_num_interop_threads(self.config.intra_op_num_threads)


        LOGGER.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            LOGGER.info("\t+ Loading model with random weights")
            self.load_model_with_no_weights()
        else:
            LOGGER.info("\t+ Loading model with pretrained weights")
            self.load_model_from_pretrained()

        # Eval mode
        LOGGER.info("\t+ Turning on model's eval mode")
        self.pretrained_model.eval()

        self.tmpdir.cleanup()

    def validate_library(self) -> None:
        if self.config.library == "vllm":
            LOGGER.info(f"\t+ Using vLLM method {self.automodel_class.__name__}")
        else:
            raise ValueError(f"Library {self.config.library} not supported")

    def load_model_from_pretrained(self) -> None:
        LOGGER.info(f"\t+ Loading model directly on device: {self.config.device}")
        with torch.device(self.config.device):
            self.pretrained_model = self.automodel_class(
                model=self.config.model, **self.automodel_kwargs
            )

    def create_no_weights_model(self) -> None:
        if self.pretrained_config is None:
            raise ValueError("Can't create no weights model without a pretrained config")

        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        LOGGER.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        LOGGER.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        LOGGER.info("\t+ Saving no weights model safetensors")
        safetensors = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensors, metadata={"format": "pt"})

        if self.is_quantized:
            LOGGER.info("\t+ Adding quantization config to no weights model's pretrained config")
            self.pretrained_config.quantization_config = self.quantization_config.to_dict()
            # tricking from_pretrained to load the model as if it was quantized

        LOGGER.info("\t+ Saving no weights model pretrained config")
        if self.config.library == "transformers":
            self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

    def load_model_with_no_weights(self) -> None:
        LOGGER.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        with random_init_weights():
            original_model, self.config.model = self.config.model, self.no_weights_model
            LOGGER.info("\t+ Loading no weights AutoModel")
            self.load_model_from_pretrained()
            self.config.model = original_model

        # dunno how necessary this is
        LOGGER.info("\t+ Tying model weights")
        self.pretrained_model.tie_weights()

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
        kwargs = {}
        from vllm import LLM
        if self.is_quantized:
            kwargs["quantization"] = self.config.quantization_scheme

        if self.config.torch_dtype is not None:
            kwargs["dtype"] = getattr(torch, self.config.torch_dtype)

        if self.config.no_weights:
            # we use our own context manager to load the model with random weights
            kwargs["_fast_init"] = False

        return kwargs

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super().prepare_inputs(inputs)

        if self.config.library == "diffusers":
            return {"prompt": inputs["prompt"]}
        elif self.config.library == "timm":
            return {"x": inputs["pixel_values"].to(self.config.device)}
        else:
            for key, value in inputs.items():
                inputs[key] = value.to(self.config.device)
            return inputs

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
            return self.pretrained_model.forward(**inputs, **kwargs)

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
            return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def call(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model(**inputs, **kwargs)

    def train(
        self,
        training_dataset: Dataset,
        training_arguments: Dict[str, Any],
        training_callbacks: List[TrainerCallback],
        training_data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    ) -> TrainerState:
        LOGGER.info(f"\t+ Wrapping training arguments with {TrainingArguments.__name__}")
        training_arguments = TrainingArguments(**training_arguments)
        LOGGER.info(f"\t+ Wrapping model with {Trainer.__name__}")
        trainer = Trainer(
            args=training_arguments,
            model=self.pretrained_model,
            callbacks=training_callbacks,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
        )
        LOGGER.info("\t+ Starting training")
        trainer.train()
        LOGGER.info("\t+ Finished training")

    def seed(self):
        super().seed()
        torch.manual_seed(self.config.seed)

        if self.config.device == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            LOGGER.info("\t+ Cleaning backend temporary directory")
            self.tmpdir.cleanup()

        gc.collect()
