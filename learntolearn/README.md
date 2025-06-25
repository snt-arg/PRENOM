<div align="center">
    <h1>👩🏻‍🎓 Learn to Learn 👨🏻‍🎓</h1>
    <p><i>Meta learning + Genetic Algorithm to optimize category-level priors</i></p>
</div>


## Introduction
This part of the code is to train priors for new categories. We use [pymoo](https://pymoo.org/) for the genetic algorithm and [Reptile](https://openai.com/index/reptile/) for meta-learning. Refer to the paper and code for more details. 


## How to Train Your Priors  
1. Install the required python libraries
    ```bash
    cd learntolearn
    pip install -r requirements.txt
    ```

2. Download the data for the specific category that you want to add (for e.g., sofa, table, car, etc.). You can download models from [ShapeNet](https://shapenet.org/) or [Sapien](https://sapien.ucsd.edu/) repositories. Additionally, for ShapeNet models, you need to preprocess them as follows:  
    - Modify the configuration in the file `task_generator/process_shapenet_category.py` to specify the data directory and category. Additionaly, you can modify further configuration to limit the models to only a subset of the original models by removing outliers that are either too large or have too few points/vertices. Moreover, you can also specify the ratio of models to be kept for the testing tasks.
    - Once the configuration is done, run the following command to generate the CAD models
        ```bash
        python process_shapenet_category.py <category_name>
        ```
        This will generate a bunch of CAD models with the format expected by the task generator in the `task_generator/cads` directory. 

3. To generate tasks, the configuration and dimension info can be seen in `task_generator/constants.py`. This file can be extended to new categories by following similar pattern as in the file. Following that, you can generate tasks using:
    ```bash
    cd task_generator
    python generate_meta_dataset.py <category_name> <num_training_tasks> <num_test_tasks>
    ```
    This will generate tasks in the `task_generator/tasks` folder. Choose a suitably high number of training tasks to ensure variability, for example `python generate_meta_dataset.py table 1000 10`.

4. To finally train the prior, follow these steps:  
    - Modify the file `config.py`. Be sure to change the `REPO_PATH` to the *absolute* path of this repository. You can also change the parameters of the genetic algorithm to train a bigger population or for longer. 
    - The file `subsampled_states.py` is the subset of the total sample space of the decision variables for each category already trained for. You can intuitively select that (reading the Instant NGP paper and the tiny-cuda-nn documentation might be helpful to understand what each variables mean) or the default will be used if you don't specify any. The subsampled states included in the file were discovered by another genetic algorithm that did not have meta-learning included. This part of the code was later removed, but should be simple to re-implement :)))
    - Following this, run this command to start the optimization process  
        ```bash
        python meta_ga.py <category_name>
        ```
    - Once the command finishes, you can visualize the results to choose the best architecture by running:  
        ```bash
        python plot_res.py <category_name> <iteration_num>
        ```
        The `iteration_num` should normally correspond to `TOTAL_EVALS`, but can be set to a lower number to visualize an earlier checkpoint. 
    - From the resulting plot, you will be able to see the *individuals* that are Pareto-optimal (dominant solutions in this context) and also the knee point (if that computation was successful). These results are annotated and you can choose manually the index of the result that you want, depending on the cost functions.
    - The command line will print out all solutions and you can copy the desired one (by counting up to the index selected in the previous step) to the file `train_final_meta.py`. This file also already has optimal architectures for the categories that already have a prior. You can use that as a baseline if needed.
    - Finally, run the meta-learning on the chosen architecture to generate the prior:
        ```bash
        python train_final_meta.py
        ```
        The generated priors are in the folder `priors` under folder `<category_name>`. They can be copied to the folder `cookbook` at the base directory of the repository, and added to `cookbook/recipes.txt` respectively. You can follow the structure of the already trained priors provided in this repository.

That's it, I guess. Good Luck!! :)))

