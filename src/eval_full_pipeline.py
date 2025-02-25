import os
import torch  
from tqdm import tqdm


from utils import answer_is_correct, question_iterator, code_to_flat, nested_to_flat
from evaluation import CSVLogger
from prompt_tools import LLMConfig
from gs_vqa.model.clip_model import CLIPModel
from gs_vqa.object_detection.owl_vit_object_detector import OWLViTObjectDetector


from asp_encoding import encode_question, run_clingo, encode_scene
from llms import OpenAILLM

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

concept_model = CLIPModel(device, model="openai/clip-vit-base-patch32")

object_detector = OWLViTObjectDetector(device, usev2=True)

def count_operators(question):
    operations = ["select", "query", "filter", "relate", "verify", "choose", "exist", "or", "different", "and", "same", "common"]
    op_counts = {f"op_{op}": 0 for op in operations}
    for op in question["semantic"]:
        operator = op["operation"].split(" ")[0]
        op_counts[f"op_{operator}"] += 1
    return op_counts

def is_scene_question(question):
    return question["semantic"][0]["operation"] == "select" and question["semantic"][0]["argument"] == "scene"

def evaluate_question(question): 
    op_counts = count_operators(question)
    result = {
        "question_id": question["qid"], 
        "semantic_str": question["semanticStr"], 
        "image_id": question["imageId"],
        "answer": question["answer"],
        **op_counts
    }
    
    if is_scene_question(question):
        return {**result, "skipped": True, "model_response": None, "correct": False, "timeout": False, "runtime_sec": 0.0}
    else:
        result["skipped"] = False

        scene_encoding = encode_scene(question, concept_model, object_detector)
        question_encoding = encode_question(question)
        answer = run_clingo(scene_encoding, question_encoding)


        if len(answer) > 0:
            return {**result, "model_response": answer, "correct": answer_is_correct(answer, question["answer"])}
        else: 
            return {**result, "model_response": "UNSAT", "correct": False}
        

translators = {
    "asp_flat": lambda x: x,
    "asp_nested": lambda x: nested_to_flat(x),
    "asp_code": lambda x: code_to_flat(x),
    "gqa": lambda x: encode_question(x)
}

COMPARE_BLIND = True
TOPK = 5
LLM_MODEL = "gpt-4o"
config = LLMConfig(target_repr="asp_nested", sampling_strategy="bert", n_similar_examples=10, cover_operators=True, prompt_version="v6", instruction_based=True)
llm_model = OpenAILLM(version=LLM_MODEL, config=config)

if __name__ == "__main__":
    logger = CSVLogger("./logs/new_eval.csv")
    models = [LLM_MODEL]
    for scene_graph, qid, question in tqdm(question_iterator(500), total=500):
        if logger.is_answered(qid, "pipeline_base"):
            continue
        scene_encoding = encode_scene(question, concept_model, object_detector)
        gt_answer = run_clingo(scene_encoding, encode_question(question), topk=TOPK)
        logger.log_safe(qid, "pipeline_base", gt_answer, config=None)
        logger.log_safe(qid, "answer", question["answer"], config=None)
        for model in models:
            
            full_name = f"pipeline_{model}"
            try:
                if not logger.is_answered(qid, model+"_raw", config):
                    llm_output, n_actual_examples = llm_model.generate(
                        question["question"]
                    )
                    logger.log(qid, llm_model.name, llm_output, n_actual_examples, raw=True, config=llm_model.config)
                pred_question_enc = translators[config.target_repr](logger.get_answer(qid, model+"_raw").values[0])
                pred_answer = run_clingo(scene_encoding, pred_question_enc, topk=TOPK)
                logger.log_safe(qid, full_name, pred_answer, config=config)
            except Exception as e:
                print(e)
                logger.log_safe(qid, full_name, "error", config=config)
        if COMPARE_BLIND:
            try:
                pred_question_enc = translators[config.target_repr](logger.get_answer(qid, model+"_raw").values[0])
                pred_answer = run_clingo(encode_scene(question, concept_model, object_detector, blind=True, question_enc=pred_question_enc), pred_question_enc, topk=TOPK)
                logger.log_safe(qid, full_name+"_blind", pred_answer, config=config)
            except Exception as e:
                print(e)
                logger.log_safe(qid, full_name+"_blind", "error", config=config)
        logger.save()

            



    
