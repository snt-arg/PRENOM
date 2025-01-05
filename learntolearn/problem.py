from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
from .execution_functions import run_single_meta_iteration, evaluate_run


class MultiObjectiveMixedMetaLearn(ElementwiseProblem):

    def __init__(self, **kwargs):
        self.category = kwargs.pop("category", None)
        print(f"Category: {self.category}")
        vars = {
            "num_meta_loops": Integer(bounds=(10, 50)),
            "num_inner_iterations": Integer(bounds=(100, 1000)),
            "meta_lr": Real(bounds=(1e-3, 1e-1)),
            "inner_lr": Real(bounds=(1e-3, 2.5e-2)),
            "log2_hashmap_size": Integer(bounds=(14, 18)),
            "per_level_scale": Choice(options=[1.25992, 1.31951, 1.38191, 1.44727]),
            "n_neurons": Choice(options=[16, 32, 64]),
            "n_hidden_layers": Integer(bounds=(1, 2))
        }
        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # run the single meta iteration
        identifier = run_single_meta_iteration(
            self.category,
            int(X["num_meta_loops"]),
            int(X["num_inner_iterations"]),
            float(X["meta_lr"]),
            float(X["inner_lr"]),
            int(X["log2_hashmap_size"]),
            float(X["per_level_scale"]),
            int(X["n_neurons"]),
            int(X["n_hidden_layers"])
        )
        
        # evaluate the run
        out["F"] = list(evaluate_run(
            self.category,
            identifier,
            float(X["inner_lr"]),
            int(X["log2_hashmap_size"]),
            float(X["per_level_scale"]),
            int(X["n_neurons"]),
            int(X["n_hidden_layers"])
        ))