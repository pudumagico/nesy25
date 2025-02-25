import json 
from gs_vqa.gs_vqa_utils import sanitize, sanitize_asp
import os
from constants import GQA_DATAPATH

with open(os.path.join(GQA_DATAPATH, "metadata", "gqa_all_attribute.json")) as f:
    all_attributes = json.load(f)

with open(os.path.join(GQA_DATAPATH, "metadata", "gqa_all_class.json")) as f:
    all_categories = json.load(f)

# scene, end, select, relate, query, verify_rel, 
# choose_attr, choose_rel, exist, all_same, all_different, 
# two_same, two_same, and, or, unique, negate.

def extract_attributes_blind(question_enc):
    attributes = set()
    standalone_values = set()
    for predicate in question_enc.split("\n"):
        if predicate in ["", "\r"]:
            continue
        op, args = predicate.replace(" ", "").split("(", maxsplit=1)
        args = args.split(")")[0].split(",")
        if op == 'relate_attr':
            attributes.add(args[-1])
        elif op == 'query':
            attributes.add(args[-1])
        elif op == 'common':
            attributes.update(all_attributes.keys())
        elif 'same' in op or 'different' in op:
            if args[-1] != "class":
                attributes.add(args[-1])
        elif 'filter' in op:
            if len(args) == 4:
                attributes.add(args[-2])
            else:
                standalone_values.add(args[-1])
        elif op == 'choose':
            standalone_values.add(args[-1])
            standalone_values.add(args[-2])
        elif op == "verify_attr":
            if args[-2] != "any":
                attributes.add(args[-2])
            else:
                standalone_values.add(args[-1])
        elif op == 'choose_attr':
            if args[-3] != "any":
                attributes.add(args[-3])
            else:
                standalone_values.add(args[-1])
                standalone_values.add(args[-2])
        elif op == 'compare':
            standalone_values.add(args[-2])
    return {sanitize(a).replace("_", " ") for a in attributes if a != 'name' and a != 'vposition' and a != 'hposition'}, \
           {sanitize(v).replace("_", " ") for v in standalone_values}


def extract_classes_blind(question_enc):
    classes = {
        "categories": set(),
        "classes": set(),
        "all": False
    }

    def add_class_or_category(c):
        c = sanitize_asp(c)
        if c in all_categories.keys():
            classes["categories"].add(c)
        else: 
            classes["classes"].add(c)
            
    for predicate in question_enc.split("\n"):
        if predicate in ["", "\r"]:
            continue
        op, args = predicate.split("(", maxsplit=1)
        args = args.split(")")[0].split(",")
        if op == 'select':
            add_class_or_category(args[-1])
        elif op == 'relate':
            add_class_or_category(args[-3])
        elif op == 'relate_any':
            classes["all"] = True
        elif op == 'relate_attr':
            add_class_or_category(args[-2])
        elif op == 'choose_rel':
            add_class_or_category(args[-4])
        elif op == 'verify_rel':
            add_class_or_category(args[-3])
    return classes


def extract_relations_blind(question_enc):
    relations = set()
    for predicate in question_enc.split("\n"):
        if predicate in ["", "\r"]:
            continue
        op, args = predicate.split("(", maxsplit=1)
        args = args.split(")")[0].split(",")
        if op in ['relate', 'relate_any']:
            relations.add(args[-2])
        elif op == 'choose_rel':
            relations.add(args[-2])
            relations.add(args[-3])
        elif op == 'verify_rel':
            relations.add(args[-2])
    return {sanitize(r).replace("_", " ") for r in relations}