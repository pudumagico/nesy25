from itertools import chain
from .auxil_operations import pick_attribute, equal_or_hyponym, get_category, get_attr
from .known_exceptions import *

AUXIL_ATTRIBUTES = ["name", "weather", "location", "hposition", "vposition"]


class Operation:
    def __init__(self, linked_exception=None, fail_on_none=False, use_metadata=False) -> None:
        self.linked_exception = linked_exception
        self.fail_on_none = fail_on_none
        self.use_metadata = use_metadata

    def run(self, argument, dependencies, extra_args, scene_graph):
        pass

    def __call__(self, argument, dependencies, extra_args, scene_graph):
        results = self.run(argument, dependencies, extra_args, scene_graph)
        if self.fail_on_none and len([x for x in results if x is not None]) == 0:
            raise self.linked_exception
        return results
    

class OpSelect(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        
        if argument == "scene":
            return ["scene"]
        node_type = argument.split(" (")[0]
        return [
            x
            for x in scene_graph.nodes
            if equal_or_hyponym(get_attr(scene_graph, node_id=x, attr="name"), node_type, use_metadata=self.use_metadata)
        ]


class OpRelate(Operation):
    def __init__(self, fail_on_none=False, use_metadata=False) -> None:
        super().__init__(MissingEdgeException, fail_on_none, use_metadata)

    def run(self, argument, dependencies, extra_args, scene_graph):
        node_type, rel_type, dir = argument.split(",")
        def matches(candidate_id, candidate_rel):
            candidate_name = get_attr(scene_graph, node_id=candidate_id, attr="name")
            class_matches = equal_or_hyponym(candidate_name, node_type, use_metadata=self.use_metadata)
            if rel_type.startswith("same"):
                category = rel_type.split(" ")[1]
                source_attr = pick_attribute(category, get_attr(scene_graph, node_id=dependencies[0]), use_metadata=self.use_metadata)
                target_attr = pick_attribute(category, get_attr(scene_graph, node_id=candidate_id), use_metadata=self.use_metadata)
                rel_matches = source_attr == target_attr
            else:
                rel_matches = candidate_rel == rel_type
            return rel_matches and class_matches
        
        results = []
        if dir.startswith("o") or dir.startswith("_"):
            for e in scene_graph.out_edges(dependencies[0], data=True):
                if matches(e[1], e[2]["name"]):
                    results.append(e[1])
        if dir.startswith("s") or dir.startswith("_"):
            for e in scene_graph.in_edges(dependencies[0], data=True):
                if matches(e[0], e[2]["name"]):
                    results.append(e[0])
        return results


class OpCommon(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        common_attr = set(get_attr(scene_graph, node_id=dependencies[0])).intersection(get_attr(scene_graph, node_id=dependencies[1]))
        return get_category(common_attr, use_metadata=self.use_metadata)


class OpVerify(Operation):
    def __init__(self, use_metadata=False) -> None:
        # Not sure if this is the right exception
        super().__init__(EmptyQueryException, use_metadata=use_metadata)

    def run(self, argument, dependencies, extra_args, scene_graph):
        if extra_args == "rel":
            return len(OpRelate(fail_on_none=False)(argument, dependencies, extra_args, scene_graph)) > 0
        # This would be the proper way but "verify: blue" seems identical to "verify color: blue" and is less error prone so we ignore it
        # if extra_args:
        #     return argument.strip() == pick_attribute(extra_args, get_attr(scene_graph, dependencies[0]))
        if len(dependencies[0]) > 1:
            raise AmbiguousAnswerException()
        return argument.strip() in get_attr(
            scene_graph,
            node_id=dependencies[0],
            attr=extra_args if extra_args in AUXIL_ATTRIBUTES else "attributes",
        )


class OpChoose(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        if argument == "":
            raise EmptyChoiceException()
        if extra_args == "rel":
            node_type, rel_type, dir = argument.split(",")
            option1, option2 = rel_type.split("|")
            if OpRelate(fail_on_none=False, use_metadata=self.use_metadata)(
                f"{node_type},{option1},{dir}",
                dependencies,
                extra_args,
                scene_graph,
            ):
                return option1
            if OpRelate(fail_on_none=True, use_metadata=self.use_metadata)(
                f"{node_type},{option2},{dir}",
                dependencies,
                extra_args,
                scene_graph,
            ):
                return option2
            
        relevant_attr = get_attr(scene_graph, node_id=dependencies[0], attr=extra_args if extra_args in AUXIL_ATTRIBUTES else "attributes")
        for option in argument.split("|"):
            if option in relevant_attr:
                return option


class OpFilter(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        if extra_args in ["hposition", "vposition"]:
            category = extra_args
        else:
            category = "attributes"
        is_negated = "not(" in argument
        argument = argument.replace("not(", "").replace(")", "")
        return list(
            filter(
                lambda x: (argument in get_attr(scene_graph, node_id=x, attr=category)) != is_negated,
                dependencies[0],
            )
        )


class OpQuery(Operation):
    def __init__(self, fail_on_none=True, use_metadata=False) -> None:
        super().__init__(EmptyQueryException, fail_on_none, use_metadata)

    def run(self, argument, dependencies, extra_args, scene_graph):
        results = []
        if argument == "place":
            raise QueryPlaceException()
        if argument in AUXIL_ATTRIBUTES:
            for node in dependencies[0]:
                results.append(get_attr(scene_graph, node_id=node, attr=argument))
        else:
            for node in dependencies[0]:
                results.append(pick_attribute(argument, get_attr(scene_graph, node_id=node), use_metadata=self.use_metadata))
        return results

class OpSame(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        candidates = list(chain(*dependencies))
        if extra_args:
            extracted_attributes = [pick_attribute(extra_args, get_attr(scene_graph, node_id=x), use_metadata=self.use_metadata) for x in candidates]
        else:
            # type-name equivalence is a little questionable
            if argument not in ["name", "type"]:
                raise NonTrivialCategoryException()
            extracted_attributes = [get_attr(scene_graph, node_id=x, attr="name") for x in candidates]
        return len(set(extracted_attributes)) == 1

class OpDifferent(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        return not OpSame()(argument, dependencies, extra_args, scene_graph)


class OpAnd(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        return dependencies[0] and dependencies[1]

class OpOr(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        return dependencies[0] or dependencies[1]

class OpExist(Operation):
    def run(self, argument, dependencies, extra_args, scene_graph):
        return len(dependencies[0]) > 0

ATOMIC_OPERATIONS = {
    "relate": OpRelate,
    "and": OpAnd,
    "common": OpCommon,
    "verify": OpVerify,
    "choose": OpChoose,
    "filter": OpFilter,
    "query": OpQuery,
    "select": OpSelect,
    "same": OpSame,
    "different": OpDifferent,
    "or": OpOr,
    "exist": OpExist,
}

