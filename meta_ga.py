# Description: This script is used to run the multi-objective mixed variable optimization for meta-learning objects
# Usage: python meta_ga.py <category>

import dill
import os
import sys
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from learntolearn import MultiObjectiveMixedMetaLearn


SAVE_EVERY = 1
TOTAL_EVALS = 50
POP_SIZE = 30
LOAD_FROM = None

if __name__ == '__main__':

    # Create the checkpoints directory if not already present
    os.makedirs("checkpoints/res", exist_ok=True)

    # Initialize the problem
    kwargs = {
        "category": sys.argv[1]
    }
    problem = MultiObjectiveMixedMetaLearn(**kwargs)

    # Initialize or load algorithm
    evals_done = 0  # Track the total number of evaluations
    if LOAD_FROM is not None:
        with open(LOAD_FROM, "rb") as f:
            checkpoint = dill.load(f)
        evals_done = int(LOAD_FROM.split("/")[-1].split(".")[0])
        print(f"Resuming from checkpoint: {LOAD_FROM} (evaluations done: {evals_done})")
    else:
        checkpoint = MixedVariableGA(pop_size=POP_SIZE, survival=RankAndCrowdingSurvival())
    
    # Optimization loop
    while evals_done < TOTAL_EVALS:
        # Calculate the next termination point
        next_termination = min(evals_done + SAVE_EVERY, TOTAL_EVALS)
        print(f"Next termination point: {next_termination}")

        termination = DefaultMultiObjectiveTermination(n_max_evals=next_termination)
        checkpoint.termination = termination

        # Perform optimization
        res = minimize(problem,
                       checkpoint,
                       termination,
                       seed=1,
                       verbose=True,
                       copy_algorithm=False)

        # Save algorithm state
        with open(f"checkpoints/{next_termination}.pkl", "wb") as f:
            dill.dump(checkpoint, f)
        print(f"Checkpoint saved: checkpoints/{next_termination}.pkl")

        # Save result state
        with open(f"checkpoints/res/{next_termination}.pkl", "wb") as f:
            dill.dump(res, f)
        print(f"Result saved: checkpoints/res/{next_termination}.pkl")

        # Update evaluations done
        evals_done = next_termination

    print("Optimization complete!")

    # Plot the final result
    plot = Scatter()
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
