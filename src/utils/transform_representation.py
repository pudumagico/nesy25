import re
from gs_vqa.gs_vqa_utils import sanitize


def extract_predicates_detailed(flat_question):
    """
    Extracts predicates with their names, output step, input steps, and remaining arguments 
    from a flat ASP representation.

    Args:
    flat_question (str): A string representation of the question in flat ASP format.

    Returns:
    List[Tuple[str, str, List[str], List[str]]]: A list of tuples, each containing the predicate name, 
    output step, list of input steps, and a list of remaining arguments.

    CREDIT: Nelson Higuera
    """

    # Improved regular expression pattern to match predicates with all details
    pattern = r'(\w+)\((.*?)\)\.'  # Matches 'predicate_name(args).'

    # Find all matches in the flat_question string
    matches = re.findall(pattern, flat_question)

    # Process each match to format the output
    extracted_predicates = []
    for match in matches:
        predicate_name = match[0]
        args = match[1].split(',')
        args = [arg.strip() for arg in args if arg.strip()]  # Remove any extra spaces and empty strings

        # Splitting the arguments into output step, input steps, and remaining arguments
        output_step = args[0]
        input_steps = [arg for arg in args[1:] if arg.isdigit()]
        remaining_args = [arg for arg in args[1:] if not arg.isdigit()]

        extracted_predicates.append((predicate_name, output_step, input_steps, remaining_args))

    return extracted_predicates


def flat_to_nested(flat_question):
    """
    Transforms a question from the flat ASP representation to a nested representation, excluding the 'end' predicate.

    Args:
    flat_question (str): A string representation of the question in flat ASP format.

    Returns:
    str: The nested representation of the question.

    CREDIT: Nelson Higuera
    """
    if "end(" in flat_question:
        flat_question = flat_question[:flat_question.find("end(")]

    # Extract predicates and their detailed information
    predicates = extract_predicates_detailed(flat_question)

    # Dictionary to store the predicates with their output step number as key
    step_predicates = {output_step: (name, input_steps, remaining_args) for name, output_step, input_steps, remaining_args in predicates}

    # Recursive function to build nested representation
    def build_nested(output_step):
        if output_step not in step_predicates:
            # If it's a scene with no arguments, return 'scene()'
            return "scene()" if output_step == '0' else output_step

        name, input_steps, remaining_args = step_predicates[output_step]

        # Build nested structure for input steps
        nested_inputs = [build_nested(input_step) for input_step in input_steps]

        # Combine nested inputs with remaining arguments
        all_args = nested_inputs + remaining_args

        # Form the predicate string
        return f"{name}({', '.join(all_args)})"

    # Find the last step (which is not 'end')
    last_step = max(step_predicates.keys(), key=int)
    nested_question = build_nested(last_step)

    return nested_question

def nested_to_flat(nested_question):
    output = ""
    remaining = nested_question
    id_counter = 0
    while remaining and ")" in remaining:
        firstparen = remaining.find(')')
        predicate, args = remaining[:firstparen].split('(')[-2:]
        args = sanitize(args)
        prefix_args = []
        if "," in predicate:
            prefix_args = predicate.split(",")[:-1]
            predicate = predicate.split(",")[-1].strip()
        remaining = "(".join(remaining[:firstparen].split('(')[:-2]) + f"({', '.join(prefix_args+[str(id_counter)])}" +  remaining[firstparen+1:]
        args = f"{id_counter}, " + args if args else f"{id_counter}"
        output += f"{predicate}({args}).\n"
        id_counter += 1
    return output + f"end({id_counter-1})."

def flat_to_code(flat_question):
    if "end(" in flat_question:
        flat_question = flat_question[:flat_question.find("end(")]
    
    predicates = extract_predicates_detailed(flat_question)
    step_predicates = {output_step: (name, input_steps, remaining_args) for name, output_step, input_steps, remaining_args in predicates}
    
    def build_code(output_step, var: str = "var0"):
        if output_step not in step_predicates:
            return "scene()" if var == "var0" else var
        name, input_steps, remaining_args = step_predicates[output_step]
        nested_inputs = [build_code(input_step, var=var) for input_step in input_steps]
        all_args = nested_inputs + remaining_args
        return f"{name}({', '.join(all_args)})"

    def needs_var(partial_preds):
        all_deps = set()
        for (_, _, deps, _) in partial_preds:
            for dep in deps:
                if (dep in all_deps) and (dep not in handled_vars):
                    return dep
                all_deps.add(dep)
        return False

    partial_preds = predicates
    partial_programs = []
    handled_vars=[]
    
    while needs_var(partial_preds):
        var = needs_var(partial_preds)
        handled_vars.append(var)
        last_var = f"var{len(partial_programs)}"

        partial_program = f"var{len(partial_programs)+1} = {build_code(var, var=last_var)}"
        partial_programs.append(partial_program)

        partial_preds = predicates[int(var)+1:]
        step_predicates = {output_step: (name, input_steps, remaining_args) for name, output_step, input_steps, remaining_args in partial_preds}
    

    last_step = max(step_predicates.keys(), key=int)
    partial_programs.append(build_code(last_step, var=f"var{len(partial_programs)}"))
    return "\n".join(partial_programs)

def code_to_nested(code_question):
    vars = code_question.split("\n")
    while len(vars) > 1:
        firstvar = vars.pop(0)
        for i, var in enumerate(vars):
            vars[i] = var.replace(firstvar.split(" = ")[0], firstvar.split(" = ")[1])
    return vars[0]

def code_to_flat(code_question):
    nested_question = code_to_nested(code_question)
    return nested_to_flat(nested_question)




if __name__ == "__main__":
    # Testing with the revised example
    revised_flat_question = """scene(0).
    select(1, 0, airplane).
    relate_any(2, 1, in_front_of, object).
    unique(3, 2).
    query(4, 3, name).
    """
    
    # nested_question = flat_to_nested(revised_flat_question)
    # print(f"nested question: {nested_question}")
    # flat_question = nested_to_flat(nested_question)
    # print(f"flat question: {flat_question}")
    flat_question = """
    scene(0).
    select(1, 0, truck).
    relate(2, 1, tire, of, subject).
    unique(3, 2).
    verif y attr(4, 3, color, black).
    unique(5, 2).
    verif y attr(6, 5, shape, round).
    and(7, 4, 6).
    end(7)
    """
    flat_to_code(flat_question)
    code_to_flat(flat_to_code(flat_question))

    code_question = """
    truck = unique(relate(select(scene(), truck), tire, of, subject))
    and(verif y attr(truck, color, black), verif y attr(truck, shape, round))
    """
