from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
import numpy as np

from .execution_functions import evaluate_run


class MultiObjectiveMixedLearn(ElementwiseProblem):

    def __init__(self, **kwargs):
        self.category = kwargs.pop("category", None)
        print(f"Category: {self.category}")
        vars = {
            "inner_lr": Real(bounds=(1e-3, 2.5e-2)),
            "log2_hashmap_size": Choice(options=[14, 15, 16, 17, 18]),
            # "per_level_scale": Choice(options=[1.25992, 1.31951, 1.38191, 1.44727]),
            "per_level_scale": Real(bounds=(1.26, 2.00)),
            "n_neurons": Choice(options=[16, 32, 64, 128]),
            "n_hidden_layers": Choice(options=[1, 2, 3]),
            "density_lambda": Real(bounds=(0.00, 0.20)),
            "depth_lambda": Real(bounds=(0.00, 2.00)),
        }
        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # evaluate the run
        identifier = np.random.randint(10000000)
        out["F"] = list(evaluate_run(
            self.category,
            identifier,
            float(X["inner_lr"]),
            int(X["log2_hashmap_size"]),
            float(X["per_level_scale"]),
            int(X["n_neurons"]),
            int(X["n_hidden_layers"]),
            float(X["density_lambda"]),
            float(X["depth_lambda"]),
            1.0,
            load_meta_model = False,
            inner_loops_to_test = [250, 500, 750,],
            use_depths = [True, False],
            evaluate_folder = "validation"
        ))