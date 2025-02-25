import networkx as nx

class GQAObject:
    def __init__(self, image_id, scene_json):
        self.image_id = image_id
        self.scene_graph = nx.DiGraph()
        self.questions = {}
        self.scene_graph.add_node(
            "scene",
            name="scene",
            location=scene_json["location"] if "location" in scene_json else None,
            weather=scene_json["weather"] if "weather" in scene_json else None,
            image_size=(scene_json["width"], scene_json["height"]),
            attributes = [],
        )
        for nodeid, node in scene_json["objects"].items():
            # this would crash on out of bounds objects
            # hposition = ["left", "middle", "right"][int(3 * (node["x"] + 0.5 * node["w"]) / scene_json["width"])]
            # vposition = ["top", "middle", "bottom"][int(3 * (node["y"] + 0.5 * node["h"]) / scene_json["height"])]
            midpoint = (node["x"] + 0.5 * node["w"], node["y"] + 0.5 * node["h"])
            if midpoint[0] > 2 * scene_json["width"] / 3:
                hposition = "right"
            elif midpoint[0] > scene_json["width"] / 3:
                hposition = "middle"
            else:
                hposition = "left"
            if midpoint[1] > 2 * scene_json["height"] / 3:
                vposition = "bottom"
            elif midpoint[1] > scene_json["height"] / 3:
                vposition = "middle"
            else:
                vposition = "top"
            self.scene_graph.add_node(
                nodeid,
                name=node["name"],
                hposition=hposition,
                vposition=vposition,
                x=node["x"],
                y=node["y"],
                w=node["w"],
                h=node["h"],
                attributes=node["attributes"],
            )
            for rel in node["relations"]:
                self.scene_graph.add_edge(nodeid, rel["object"], name=rel["name"])

    def add_question(self, question_id, question_json):
        self.questions[question_id] = question_json
