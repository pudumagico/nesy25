from constants import read_concepts

from .question_encoding import encode_question
from .asp_utils import sanitize_asp


class_to_category, value_to_attribute = read_concepts()


def encode_sample(question):
    return (encode_scene_perfect(question['sceneGraph']), encode_question(question))


def encode_scene_perfect(scene_graph):
    scene_encoding = ""
    for oid, nodeinfo in scene_graph.nodes(data=True):
        if oid  == 'scene':
            continue

        scene_encoding += f"object({oid}).\n"

        scene_encoding += f"has_attr({oid}, class, {sanitize_asp(nodeinfo['name'])}).\n"
        scene_encoding += f"has_attr({oid}, name, {sanitize_asp(nodeinfo['name'])}).\n"
       
        for category in class_to_category.get(sanitize_asp(nodeinfo['name']), []):
            scene_encoding += f"has_attr({oid}, class, {category}).\n"    

        for value in nodeinfo['attributes']:
            if value in value_to_attribute:
                for att in value_to_attribute[value]:
                    scene_encoding += f"has_attr({oid}, {sanitize_asp(att)}, {sanitize_asp(value)}).\n"
            else:
                scene_encoding += f"has_attr({oid}, any, {sanitize_asp(value)}).\n"
        
        scene_encoding += f"has_attr({oid}, hposition, {nodeinfo['hposition']}).\n"
        scene_encoding += f"has_attr({oid}, vposition, {nodeinfo['vposition']}).\n"

        for rel in scene_graph.out_edges(oid, data=True):
            scene_encoding += f"has_rel({oid}, {sanitize_asp(rel[2]['name'])}, {rel[1]}).\n"

        # for rel in scene_graph.in_edges(oid, data=True):
        #     scene_encoding += f"has_rel({oid}, {sanitize_asp(rel[2]['name'])}, {rel[0]}).\n"
    return scene_encoding
