import openai

from prompt_tools import LLMConfig
from .genericLLM import GenericLLM
from constants import API_KEY


class OpenAILLM(GenericLLM):

    def __init__(
        self,
        config: LLMConfig,
        version: str = "gpt-35-turbo",
        name: str = None,
        inference_kwargs: dict = {},
    ) -> None:
        super().__init__(config)
        self.version = version
        self.inference_kwargs = inference_kwargs
        self.inference_kwargs["max_tokens"] = self._max_tokens
        self.inference_kwargs["temperature"] = 0.0
        self._name = name if name else version
        self.client = openai.OpenAI(api_key=API_KEY)

    
    def _generate(self, question: str) -> tuple[str, int]:
        prompt, nr_examples = self.prompt_creator.get_prompt(question)
        
        completion = self.client.chat.completions.create(
            model=self.version,
            messages=prompt,
            **self.inference_kwargs
        )
        return completion.choices[0].message.content, nr_examples
    
    def post_process(self, output: str) -> str:
        return super().cutoff_output(output)
