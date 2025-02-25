import re
from prompt_tools import PromptCreator, LLMConfig

class GenericLLM:

    def __init__(self, config: LLMConfig, name: str = "GenericLLM") -> None:
        self.prompt_creator = PromptCreator(config=config)
        self.config = config
        self.target_repr = config.target_repr
        self._name = name
        self._max_tokens = {
            "asp_flat": 50,
            "asp_code": 75,
            "asp_nested": 100,
            "gqa": 150,
        }[self.target_repr]

    @property
    def name(self):
        return self._name
    
    @property
    def ablation_name(self):
        return self.name + "_" + "_".join(map(str, self.config.get_ablation_values()))
  
    def post_process(self, output: str) -> str:
        return output

    def _generate(self, question: str) -> tuple[str, int]:
        pass

    def generate(self, question: str) -> tuple[str, int]:
        output, nr_examples = self._generate(question)
        return self.post_process(output), nr_examples
    
    def cutoff_output(self, output):
        def cutoff_at_closing_symbol(instr, opensym="(", closingsym=")"):
            open_pars = 0
            first_par = instr.find(opensym)
            for i, c in enumerate(instr[first_par:]):
                if c == opensym:
                    open_pars += 1
                elif c == closingsym:
                    open_pars -= 1
                if open_pars == 0:
                    return instr[:first_par+i+1]
            return instr + open_pars * closingsym
        # especially gpt loves to use code block
        if "```" in output:
            output = output.split("```", maxsplit=1)[1]
            output = output.split("\n", maxsplit=1)[1]
        
        if self.target_repr == "asp_flat":
            output = re.split(r"end\(\d+\)\.", output)[0]
            last_index = len(output.split(".\n"))-2
            return output + f"end({last_index}).\n"
        elif self.target_repr == "asp_nested":
            return cutoff_at_closing_symbol(output)
        if self.target_repr == "asp_code":
            out = ""
            for line in output.split("\n"):
                if "=" in line:
                    # still at variable stage
                    out += line + "\n"
                else:
                    return out + cutoff_at_closing_symbol(line)
        elif self.target_repr == "gqa":
            return cutoff_at_closing_symbol(output, "[", "]")
        return output

    def reinit(self, new_config: LLMConfig):
        super(self.__class__, self).__init__(new_config, self.name)