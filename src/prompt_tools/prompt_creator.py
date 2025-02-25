import random
import torch
import os
import json
import warnings
from transformers import BertTokenizer, BertModel
from utils import question_iterator, flat_to_nested, flat_to_code
from constants import GQA_DATAPATH, CACHE_PATH
from asp_encoding import encode_question
from .prompt_templates import gqa_templates, asp_templates


class LLMConfig:
    def __init__(
        self,
        target_repr: str,
        sampling_strategy: str = "random",
        n_similar_examples: int = 5,
        cover_operators: bool = False,
        prompt_version: str = "v1",
        instruction_based: bool = False,
    ) -> None:
        self.target_repr = target_repr
        self.n_similar_examples = n_similar_examples
        self.cover_operators = cover_operators
        self.sampling_strategy = sampling_strategy
        self.prompt_version = prompt_version
        self.instruction_based = instruction_based

    @staticmethod
    def ablation_categories():
        return [
            "target_repr",
            "sampling_strategy",
            "n_similar_examples",
            "cover_operators",
            "prompt_version",
        ]

    @staticmethod
    def cat_to_human_readable(cat):
        return {
            "target_repr": "Target Representation",
            "sampling_strategy": "Sampling Strategy",
            "n_similar_examples": "Number of Similar Examples",
            "cover_operators": "Cover Operators",
            "prompt_version": "Prompt Version",
        }[cat]
    
    @staticmethod
    def repr_to_human_readable(repr):
        return {
            "asp_flat": "Flat ASP",
            "asp_nested": "Nested ASP",
            "asp_code": "Code-like ASP",
            "gqa": "GQA",
        }[repr]

    def get_ablation_values(self):
        return [self.__dict__[key] for key in self.ablation_categories()]


class PromptCreator:

    def __init__(self, config: LLMConfig, precache_examples: int = 500) -> None:
        """_summary_

        Args:
            target_repr (str):
                - asp_flat
                - asp_nested
                - gqa
            n_similar_examples (int, optional): Number of examples to pick by similarity. More may be added to cover all operators.
            sampling_strategy (str, optional):
                - random: sample randomly from the in-context examples
                - bert: sample the closest in-context examples using BERT
                - jaccard: sample the closest in-context examples using number of shared words
            instruction_based (bool, optional): whether to return separate system prompts

        """
        prompt_version = config.prompt_version
        self.target_repr = config.target_repr
        self.strategy = config.sampling_strategy
        self.n_examples = config.n_similar_examples
        self.instruction_based = config.instruction_based
        self.cover_operators = config.cover_operators

        self.in_context_examples = {}
        translators = {
            "asp_flat": lambda x: encode_question(x),
            "asp_nested": lambda x: flat_to_nested(encode_question(x)),
            "asp_code": lambda x: flat_to_code(encode_question(x)),
            "gqa": lambda x: str(x["semantic"]),
        }
        if self.target_repr == "gqa":
            self.preprompt = gqa_templates["preprompt"][prompt_version]
            self.template = gqa_templates["template"][prompt_version]
        elif self.target_repr.startswith("asp"):
            self.preprompt = asp_templates["preprompt"][prompt_version]
            self.template = asp_templates["template"][prompt_version]
        else:
            raise ValueError(f"Unknown target representation: {self.target_repr}")

        with open(os.path.join(GQA_DATAPATH, "in_context.json"), "r") as f:
            self.in_context_examples = {
                example["question"]: translators[self.target_repr](example)
                for example in json.load(f)
            }

        if self.strategy == "bert":
            self.example_embeddings = self.load_or_embed(
                list(self.in_context_examples.keys())
            )

            precache_questions = [question["question"] for (_, _, question) in question_iterator(precache_examples)]
            precache_values =  self.load_or_embed(precache_questions, targetset="val")
            self.target_embeddings = dict(zip(precache_questions, precache_values))
    

    def load_or_embed(self, examples, targetset: str = "train"):
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        embed = lambda x: bert_model(**bert_tokenizer(x, return_tensors="pt"))[0].mean(dim=1)
        bert_cache = os.path.join(CACHE_PATH, f"bert_{targetset}_{len(set(examples))}.pt")
        if not os.path.exists(bert_cache):
            with torch.no_grad():
                embeddings = torch.stack(
                    [
                        embed(example)
                        for example in examples
                    ]
                ).squeeze()
            torch.save(embeddings, bert_cache)
        else:
            embeddings = torch.load(bert_cache)
        return embeddings
        

    def get_embedding(self, question):
        if question not in self.target_embeddings:
            warnings.warn(f"Embedding for {question} not found, not precaching gets expensive.")
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            bert_model = BertModel.from_pretrained("bert-base-uncased")
            with torch.no_grad():
                return bert_model(**bert_tokenizer(question, return_tensors="pt"))[0].mean(dim=1)
        return self.target_embeddings[question]


    def get_prompt(self, question: str):
        examples, nr_examples = self.sample_examples(question)
        content = self.template.format(examples=examples, question=question)
        prompt = [
            {
                "role": "system",
                "content": self.preprompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        if self.instruction_based:
            prompt = [
                {
                    "role": "system",
                    "content": self.preprompt,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
        else:
            prompt = self.preprompt + "\n" + content
        return prompt, nr_examples

    def sample_examples(self, question: str = None) -> tuple[str, int]:
        "Gather- and return the number of examples."
        if self.strategy == "random":
            selected_examples = random.sample(
                sorted(self.in_context_examples), self.n_examples+1
            )
        elif self.strategy == "bert":
            question_embedding = self.get_embedding(question)
            similarities = torch.cosine_similarity(
                question_embedding, self.example_embeddings
            )
            order = torch.argsort(similarities, descending=True)
            selected_examples = [
                list(self.in_context_examples.keys())[i]
                for i in order[: self.n_examples+1]
            ]
        elif self.strategy == "jaccard":
            words = set(question.split())
            selected_examples = sorted(
                self.in_context_examples,
                key=lambda x: len(words.intersection(set(x.split()))),
                reverse=True,
            )[: self.n_examples+1]
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
        if question in selected_examples:
            selected_examples.remove(question)
        else:
            selected_examples.pop()
        if self.cover_operators:
            selected_examples = self.cover_all_operators(selected_examples)
        if self.target_repr == "asp_code":
            # replace one example with one that includes a variable
            for example, program in self.in_context_examples.items():
                if "=" in program:
                    selected_examples[-1] = example
                    break
                

        return "\n\n".join(
            [
                example + "\n" + self.in_context_examples[example]
                for example in selected_examples
            ]
        ), len(selected_examples)

    def cover_all_operators(self, selected_examples):
        if self.target_repr.startswith("asp"):
            operators = [
                "scene",
                # "end", # always there for flat, never there for nested
                "select",
                "relate",
                "query",
                "verify_rel",
                "choose_attr",
                "choose_rel",
                "exist",
                "all_same",
                "two_same",
                "and",
                "or",
                "unique",
                "negate",
            ]
        elif self.target_repr == "gqa":
            operators = [
                "select",
                "relate",
                "common",
                "verify",
                "choose",
                "filter",
                "query",
                "same",
                "different",
                "and",
                "or",
                "exist",
            ]
        selected_example_content = "\n\n".join(
            [self.in_context_examples[example] for example in selected_examples]
        )
        for operator in operators:

            def _get_missing_operator():
                for key, candidate in self.in_context_examples.items():
                    if operator in candidate:
                        return key
                raise Exception(f"Could not find example with operator {operator}")

            if operator not in selected_example_content:
                new_example = _get_missing_operator()
                selected_examples.append(new_example)
                selected_example_content += (
                    "\n\n" + self.in_context_examples[new_example]
                )
        return selected_examples
