from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset

import os
import pickle
from itertools import chain
import pandas as pd

from constants import GQA_DATAPATH, read_synsets

from gs_vqa.gs_vqa_utils import sanitize_asp
from asp_encoding import encode_question
from .transform_representation import flat_to_nested, flat_to_code

# import nltk
# nltk.download("wordnet")

obj_synsets, attr_synsets = read_synsets()

def question_iterator(n_questions=10, target_set="val", only_converging=False):
    gqa_objects = pickle.load(open(os.path.join(GQA_DATAPATH, f"{target_set}_objects.pickle"), "rb"))
    i = 0
    for obj in gqa_objects.values():
        scene_graph = obj.scene_graph

        for qid, question in obj.questions.items():
            if only_converging:
                q_enc = encode_question(question)
                if flat_to_code(q_enc) == flat_to_nested(q_enc):
                    continue
            yield scene_graph, qid, question
            i += 1
            if i == n_questions:
                return

def is_hyponym(a: Synset, b: Synset):
    if a == b:
        return True
    return b in set(chain(*a._iter_hypernym_lists()))
            
def valid_interpretation(pred, actual, hypernym_levels=0):
    if actual not in obj_synsets:
        return False
    actual_synset = wordnet.synset(obj_synsets[actual])
    pred_synsets = wordnet.synsets(pred)
    # synonym
    if actual_synset in pred_synsets:
        return True
    # pred was more specific
    for syn_candidate in pred_synsets:
        if is_hyponym(syn_candidate, actual_synset):
            return True
    # pred was slightly more general
    def _limited_hypernym(current_syn, n):
        if n == 0:
            return False
        if syn_candidate == current_syn:
            return True
        for parent in current_syn.hypernyms():
            return _limited_hypernym(parent, n-1)
        
    for syn_candidate in pred_synsets:
        if _limited_hypernym(actual_synset, hypernym_levels):
            return True
    return False

def answer_is_correct(answer, correct_answer, use_wordnet=False, hypernym_levels=0, top_k=1):
    if answer in ["", None]:
        return False
    if isinstance(answer, list) or isinstance(answer, str) and answer.startswith("["):
        answer = eval(answer) if isinstance(answer, str) else answer
        return any([answer_is_correct(ans, correct_answer, use_wordnet=use_wordnet, hypernym_levels=hypernym_levels, top_k=top_k) for ans in answer[:top_k]])
    if isinstance(answer, dict):
        # probably trying to untangle a gqa raw answer
        return False
    if pd.isna(answer):
        return False
    if sanitize_asp(answer) == sanitize_asp(correct_answer): 
        return True
    if (answer == "to_the_right_of" and correct_answer == "right") or \
        (answer == "to_the_left_of" and correct_answer == "left") or \
        (answer == "in_front_of" and correct_answer == "front"):
        return True
    if use_wordnet and valid_interpretation(answer, correct_answer, hypernym_levels=hypernym_levels):
        return True
    return False
