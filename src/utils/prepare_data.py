import json
import pickle
import os
from tqdm import tqdm

# sys add path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import question_iterator
from constants import GQA_DATAPATH
from semantic_interpreter.gqa_object import GQAObject, FakeGQAObject


def populate_objects(target: str = "val") -> dict[str, GQAObject]:

    with open(os.path.join(GQA_DATAPATH, f"{target}_balanced_questions.json")) as f:
        gqa_questions = json.load(f)

    images = {}
    # no scene graphs for testdev, therefore spoof gqa objects
    if target == "testdev":
        for question_id in tqdm(gqa_questions, desc="gathering questions"):
            image_id = gqa_questions[question_id]["imageId"]
            if image_id not in images:
                images[image_id] = FakeGQAObject(None, {"image_id": image_id, "questions": {question_id: gqa_questions[question_id]}, "scene_graph": {}})
            else:
                images[image_id]["questions"][question_id] = gqa_questions[question_id]
        return images

    with open(os.path.join(GQA_DATAPATH, f"{target}_sceneGraphs.json")) as f:
        gqa_scene_graphs = json.load(f)

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
    target = "testdev"
    # pickle.dump(populate_objects(target=target), open(os.path.join(GQA_DATAPATH, f"{target}_objects.pickle"), "wb"))
    # gqa_objects = pickle.load(open(os.path.join(GQA_DATAPATH, f"{target}_objects.pickle"), "rb"))
    # write_incontext_examples(n=1000)
    pass