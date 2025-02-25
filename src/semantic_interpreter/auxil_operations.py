from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset

from itertools import chain

from constants import read_concepts, read_synsets

from asp_encoding.asp_utils import sanitize_asp
from semantic_interpreter.known_exceptions import (
    NotInWordnetException,
    ManyAttrCandidatesException,
    IncompleteMetadataException,
)


class_to_category, value_to_attribute = read_concepts()
obj_synsets, attr_synsets = read_synsets()

WORDNET_EXCEPTIONS = {
    "white.n.01": "white.n.02",
    "grey.n.01": "grey.n.05",
    "dark.n.01": "black.n.01",
    "size.n.01": "size.n.02",
}


def treat_wordnet_exceptions(word):
    return (
        wordnet.synset(WORDNET_EXCEPTIONS[word])
        if word in WORDNET_EXCEPTIONS
        else wordnet.synset(word)
    )


def is_hyponym(a: Synset, b: Synset):
    if a == b:
        return True
    return b in set(chain(*a._iter_hypernym_lists()))


def equal_or_hyponym(a, b, use_metadata=False):
    if a == b or "_" in [a, b]:
        return True

    if use_metadata:
        a,b = sanitize_asp(a), sanitize_asp(b)
        def resolve_match(a, b):
            if a in class_to_category:
                categories = class_to_category[a]
                if b in categories:
                    return True
            return a == b
        return resolve_match(a, b) or resolve_match(b, a)

    if a not in obj_synsets or b not in obj_synsets:
        return False
    syn_a = wordnet.synset(obj_synsets[a])
    syn_b = wordnet.synset(obj_synsets[b])
    return is_hyponym(syn_a, syn_b)


def pick_attribute(category, attributes, use_metadata=False):
    """Given a category and a list of attributes, returns the one that fits the category. Either through wordnet or metadata."""
    if use_metadata:
        for attr in attributes:
            if attr not in value_to_attribute:
                raise IncompleteMetadataException()
            if value_to_attribute[attr] == category:
                return attr

    def _try_nounify(syn):
        try:
            nounified_name = syn.name().split(".")[0] + ".n.01"
            return treat_wordnet_exceptions(nounified_name)
        except:
            return syn

    if category in obj_synsets:
        syn_category = treat_wordnet_exceptions(obj_synsets[category])
    elif wordnet.synsets(category):
        syn_category = wordnet.synsets(category)[0]
    else:
        raise NotInWordnetException()

    for attr in attributes:
        if attr not in attr_synsets:
            continue
        syn_attr = _try_nounify(wordnet.synset(attr_synsets[attr]))
        if syn_attr and is_hyponym(syn_attr, syn_category):
            return attr


def get_category(attribute, use_metadata=False):
    known_categories = ["color", "material", "shape"]
    for category in known_categories:
        if pick_attribute(category, attribute, use_metadata=use_metadata):
            return category
    raise Exception("If this can happen it needs a custom Exception.")


def get_attr(scene_graph, node_id, attr="attributes"):
    if isinstance(node_id, list):
        if len(node_id) == 1:
            node_id = node_id[0]
        else:
            return [get_attr(scene_graph, x, attr) for x in node_id]
    if node_id != "scene":
        # sometimes weather is asked of sky even though the scene holds that information
        attr = attr.replace("location", "attributes").replace("weather", "attributes")
    return scene_graph.nodes[node_id][attr]