import os
import json

API_KEY = None
MODELPATH = os.path.join("/home/bileam", "models")
DATAPATH = os.path.join("/home/bileam", "data")
GQA_DATAPATH = os.path.join(DATAPATH, "gqa")
VISUAL_GENOME_DATAPATH = os.path.join(DATAPATH, "visual_genome")
CACHE_PATH = os.path.join(DATAPATH, "cache")

def read_concepts():
    with open(os.path.join(GQA_DATAPATH, "metadata", 'gqa_all_class.json')) as f:
        categories = json.load(f)
    class_to_category = {}
    for category, classes in categories.items():
        for c in classes:
            if c not in class_to_category:
                class_to_category[c] = [category]
            else:
                class_to_category[c].append(category)

    with open(os.path.join(GQA_DATAPATH, "metadata", 'gqa_all_attribute.json')) as f:
        attributes = json.load(f)
    value_to_attribute = {}
    for attribute, values in attributes.items():
        for v in values:
            if v not in value_to_attribute:
                value_to_attribute[v] = [attribute]
            else:
                value_to_attribute[v].append(attribute)
    return class_to_category, value_to_attribute

def read_synsets():
    with open(os.path.join(VISUAL_GENOME_DATAPATH, "object_synsets.json")) as f:
        obj_synsets = json.load(f)

    with open(os.path.join(VISUAL_GENOME_DATAPATH, "attribute_synsets.json")) as f:
        attr_synsets = json.load(f)
    return obj_synsets, attr_synsets
