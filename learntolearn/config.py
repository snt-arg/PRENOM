# paths - most need to be absolute paths because they are used in subprocess calls
REPO_DIR = "/home/saadejazz/PRENOM"
TASKS_DIR = f"{REPO_DIR}/learntolearn/task_generator/tasks"
SCRIPT_DIR = f"{REPO_DIR}/dependencies/Multi-Object-NeRF"

# output directory needs to be absolute and have a trailing slash
OUTPUT_DIR = f"{REPO_DIR}/learntolearn/output/"

# scaling constants for optimization objective of model complexity
# total objective is: chamfer_distance (in mm) + lambda_model_size * model_size + lambda_train_time * train_time
LAMBDA_MODEL_SIZE = 0.001
LAMBDA_TRAIN_TIME = 2.00

# for evaluation
NUM_SAMPLED_POINTS = 2**16


# Meta-learning configuration
SAVE_EVERY = 1          # save the algorithm state every SAVE_EVERY evaluations
TOTAL_EVALS = 20        # total number of evaluations to perform
POP_SIZE = 10           # population size for the meta-learning algorithm
