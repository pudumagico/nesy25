from clingo.control import Control
from .asp_utils import sanitize
import os

def run_clingo(scene_encoding, question_encoding, topk=1, forced_answer=None):
    ctl = Control()
    # ctl.configuration.solver.opt_strategy = "usc,3"
    
    with open(os.path.join(os.path.dirname(__file__), "theory.lp")) as theory_file:
        lp = theory_file.read()

    lp += "\n% ------ scene encoding ------\n"
    lp += scene_encoding
    lp += "\n% ------ question encoding ------\n"
    lp += question_encoding

    if forced_answer is not None:
        lp += "\n% ------ forced answer ------\n"
        if forced_answer == "front":
            forced_answer = "in_front_of"
        elif forced_answer == "left":
            forced_answer = "to_the_left_of"
        elif forced_answer == "right":
            forced_answer = "to_the_right_of"

        lp += f":~ not ans({sanitize(forced_answer)}). [1@2]"

    
    # with open("full.lp", "w") as f:
    #     f.write(lp)
    ctl.add(lp)
    answers = []
    def on_model(model):
        for s in model.symbols(shown=True):
            answers.append(s.arguments[0].name)

    ctl.ground()
    handle = ctl.solve(on_model=on_model, async_ = True)
    handle.wait(timeout=10.0)
    return answers[-topk:]
