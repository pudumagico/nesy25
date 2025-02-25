import torch
from typing import Any, Dict
from transformers import pipeline

from prompt_tools import LLMConfig
from .genericLLM import GenericLLM


class HuggingFaceLLM(GenericLLM):

    def __init__(
        self,
        config: LLMConfig,
        version: str = "HuggingFaceH4/zephyr-7b-alpha",
        name: str = None,
        inference_kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__(config)
        self.pipe = pipeline(
            "text-generation",
            model=version,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.inference_kwargs = inference_kwargs
        self.inference_kwargs["pad_token_id"] = self.pipe.tokenizer.eos_token_id
        self.inference_kwargs["max_new_tokens"] = self._max_tokens
        self._name = name if name else version

    def _generate(self, question: str) -> tuple[str, int]:
        prompt, nr_examples = self.prompt_creator.get_prompt(question)
        messages = self.pipe.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        return self.pipe(messages, **self.inference_kwargs)[0]["generated_text"], nr_examples

    def post_process(self, output: str) -> str:
        output = output.split("<|assistant|>\n")[1]
        return self.cutoff_output(output)
