from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
from .execution_functions import run_single_meta_iteration, evaluate_run
from .subsampled_states import SUBSAMPLED_STATES

class MultiObjectiveMixedMetaLearn(ElementwiseProblem):

    def __init__(self, **kwargs):
        self.category = kwargs.pop("category", None)
        print(f"Category: {self.category}")
        if not SUBSAMPLED_STATES.get(self.category):
            raise ValueError(f"Category not found in SUBSAMPLED_STATES")
        self.subsampled_states = SUBSAMPLED_STATES[self.category]
        
        vars = {
            # meta learning parameters
            "num_meta_loops": Integer(bounds=(25, 50)),
            "num_inner_iterations": Integer(bounds=(500, 1000)),
            "meta_lr": Real(bounds=(1e-3, 2e-1)), # most important parameter
            
            # inner learning parameters - ranges already optimised in the learning problem
            # fine-tuning parameters in the meta learning problem
            "inner_lr": Real(bounds=tuple(self.subsampled_states["inner_lr"])),
            "log2_hashmap_size": Choice(options=tuple(self.subsampled_states["log2_hashmap_size"])),
            "per_level_scale": Choice(options=list(self.subsampled_states["per_level_scale"])),
            "n_neurons": Choice(options=list(self.subsampled_states["n_neurons"])),
            "n_hidden_layers": Choice(options=list(self.subsampled_states["n_hidden_layers"]))
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
            int(X["n_hidden_layers"]),
            load_meta_model = True,
            inner_loops_to_test = [25, 60, 125, 250],
            use_depths = [True, False],
            evaluate_folder = "test"
        ))