import dill
import os
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from learntolearn import MultiObjectiveMixedMetaLearn


SAVE_EVERY = 5
TOTAL_EVALS = 100
LOAD_FROM = None

if __name__ == '__main__':

    # make the checkpoints directory if not already present
    os.makedirs("checkpoints", exist_ok=True)

    problem = MultiObjectiveMixedMetaLearn()
    algorithm = MixedVariableGA(pop_size=20, survival=RankAndCrowdingSurvival())

    checkpoint = algorithm
    termination = SAVE_EVERY
    if LOAD_FROM is not None:
        evals = int(LOAD_FROM.partition("/")[2].partition(".")[0])
        termination = evals + SAVE_EVERY
        with open(LOAD_FROM, "rb") as f:
            checkpoint = dill.load(f)    
    
    while termination < TOTAL_EVALS:
        checkpoint.termination = MaximumFunctionCallTermination(termination)
        res = minimize(problem,
                algorithm,
                seed=1,
                verbose=True,
                copy_algorithm=False)

        with open(f"checkpoints/{termination}.pkl", "wb") as f:
            dill.dump(res, f)
        termination += SAVE_EVERY
    print("Done")
    
    # plot the final result
    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()

