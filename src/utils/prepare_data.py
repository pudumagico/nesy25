import json
import pickle
import os
from tqdm import tqdm
from utils import question_iterator
from constants import GQA_DATAPATH
from semantic_interpreter.gqa_object import GQAObject



def populate_objects(target: str = "val") -> dict[str, GQAObject]:

    with open(os.path.join(GQA_DATAPATH, f"{target}_balanced_questions.json")) as f:
        gqa_questions = json.load(f)

    with open(os.path.join(GQA_DATAPATH, f"{target}_sceneGraphs.json")) as f:
        gqa_scene_graphs = json.load(f)
    images = {}

    for image_id in tqdm(gqa_scene_graphs, desc="gathering images"):
        image = GQAObject(image_id, gqa_scene_graphs[image_id])
        images[image.image_id] = image

    for question_id in tqdm(gqa_questions, desc="adding questions"):
        image_id = gqa_questions[question_id]["imageId"]
        if image_id not in images:
            continue
        images[image_id].add_question(question_id, gqa_questions[question_id])
    return images

def write_incontext_examples(n):
    examples = []
    for _, _, question in question_iterator(n):
        examples.append({"question": question['question'], "semantic": question['semantic']})
    with open(os.path.join(GQA_DATAPATH, "in_context.json"), "w") as f:
        json.dump(examples, f)


    


if __name__ == "__main__":
    # target = "val"
    # pickle.dump(populate_objects(target=target), open(os.path.join(GQA_DATAPATH, f"{target}_objects.pickle"), "wb"))
    # gqa_objects = pickle.load(open(os.path.join(GQA_DATAPATH, f"{target}_objects.pickle"), "rb"))
    write_incontext_examples(n=1000)
    pass