import dill
import sys
import numpy as np
import matplotlib.pyplot as plt
from kneefinder import KneeFinder


if __name__ == "__main__":
    add_path = ""
    category = sys.argv[1]
    iteration = sys.argv[2]
    
    res = f"checkpoints/{category}/res/{iteration}.pkl"
    with open(res, "rb") as f:
        res = dill.load(f)
    algo = f"checkpoints/{category}/{iteration}.pkl"
    with open(algo, "rb") as f:
        algo = dill.load(f)
        
    print("Population size: ", len(algo.pop.get("F")))
    print("Solution size: ", len(res.F))

    # get the knee point
    found_knee = False
    if len(res.F) > 2:
        x = res.F[:, 0]
        y = res.F[:, 1]
        kf = KneeFinder(x, y)
        x_val, y_val = kf.find_knee()
    
        y = list(y)
        if y_val in y:
            chosen_part = res.X[y.index(y_val)]
            chosen_cost = res.F[y.index(y_val)]
            found_knee = True
            print(y.index(y_val))
            print(f"Chosen design for {category}: {chosen_part}")
    
    # print the results
    print(res.X)
    print(res.F)
    
    # plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(algo.pop.get("F")[:, 0], algo.pop.get("F")[:, 1], facecolor="none", edgecolor="blue", label="Non-dominant solutions")
    ax.scatter(res.F[:, 0], res.F[:, 1], facecolor="none", edgecolor="green", label="Pareto front")
    if found_knee:
        ax.scatter(x_val, y_val, facecolor="red", edgecolor="red", label="Knee point")
    for i, txt in enumerate(res.X):
        ax.annotate(f"{i}", (res.F[i, 0]-0.05, res.F[i, 1]-1), fontsize=10, va='top', ha='right')
    ax.set_xlabel("Reconstruction quality (Cost 1)")
    ax.set_ylabel("Training time (Cost 2)")
    ax.set_title(f"Pareto front for {category} at iteration {iteration}")
    ax.legend()
    plt.show()
    