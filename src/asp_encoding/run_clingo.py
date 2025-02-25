from clingo.control import Control
import os

def run_clingo(scene_encoding, question_encoding, topk=1):
    ctl = Control()
    # ctl.configuration.solver.opt_strategy = "usc,3"
    
    with open(os.path.join(os.path.dirname(__file__), "theory.lp")) as theory_file:
        lp = theory_file.read()

    lp += "\n% ------ scene encoding ------\n"
    lp += scene_encoding
    lp += "\n% ------ question encoding ------\n"
    lp += question_encoding
    
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