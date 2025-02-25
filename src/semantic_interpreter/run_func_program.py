import networkx as nx
from .known_exceptions import QueryPlaceException, KnownException, AmbiguousAnswerException
from .atomic_operations import ATOMIC_OPERATIONS

def sanitize_answer(answer):
    if isinstance(answer, bool):
        return ["no", "yes"][answer]
    if isinstance(answer, list):
        answers = set(sanitize_answer(x) for x in answer)
        if len(answers) > 1:
            raise AmbiguousAnswerException()
        return sanitize_answer(answer[0])
    if isinstance(answer, str):
        # spatial fixes
        if answer.startswith("to the "):
            return answer.split()[2]
        if answer == "in front of":
            return "front"
    return answer

def run_func_program(scene_graph: nx.DiGraph, semantic, use_metadata=False) -> tuple[str, str, type]:
    """Run the functional program. Returns (status, answer, exception) tuple."""
    results = []
    try:
        for step in semantic:
            split_op = step["operation"].split(" ", 1)
            operation = ATOMIC_OPERATIONS[split_op[0]](use_metadata=use_metadata)
            stepresult = operation(
                step["argument"],
                dependencies=[results[x] for x in step["dependencies"]],
                extra_args=split_op[1] if len(split_op) > 1 else None,
                scene_graph=scene_graph,
            )
            results.append(stepresult)
        return "success", sanitize_answer(results[-1]), None
    except QueryPlaceException:
        return "skipped", None, None
    except KnownException as e:
        return "known_error", None, type(e)
    except Exception as e:
        return "unknown_error", None, type(e)
