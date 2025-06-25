from execution_functions import run_single_meta_iteration


ARCHITECTURES = {
    # "mug":      {'num_meta_loops': 20, 'num_inner_iterations': 1537, 'meta_lr': 0.09016509459007134, 'meta_lr_decay': 0.9690245386471475, 'density_lambda': 0.16628791618788216, 'depth_lambda': 0.6721754363392872, 'depth_prescale': 4.8053579562065085, 'inner_lr': 0.00903998567977555, 'per_level_scale': 1.377257212392083, 'log2_hashmap_size': 15, 'n_neurons': 16, 'n_hidden_layers': 1},
    # "ball":     {'num_meta_loops': 64, 'num_inner_iterations': 513, 'meta_lr': 0.046653214901279105, 'meta_lr_decay': 0.9520956964838974, 'density_lambda': 0.10956631023611813, 'depth_lambda': 0.7252489318144489, 'depth_prescale': 1.9943387633786571, 'inner_lr': 0.0014540261556016069, 'per_level_scale': 1.9997413344571477, 'log2_hashmap_size': 15, 'n_neurons': 32, 'n_hidden_layers': 1},
    # "laptop":   {'num_meta_loops': 95, 'num_inner_iterations': 1000, 'meta_lr': 0.033519904794919033, 'meta_lr_decay': 0.9925863895374577, 'density_lambda': 0.04239419390402817, 'depth_lambda': 1.9463793090888988, 'depth_prescale': 8.9650381975497910, 'inner_lr': 0.0246478801409673, 'per_level_scale': 1.4157668686062883, 'log2_hashmap_size': 14, 'n_neurons': 16, 'n_hidden_layers': 1},
    # "book":     {'num_meta_loops': 72, 'num_inner_iterations': 1272, 'meta_lr': 0.017944506796766808, 'meta_lr_decay': 0.9519594162537623, 'density_lambda': 0.10129410804650187, 'depth_lambda': 1.7629282548511624, 'depth_prescale': 13.81050767012106, 'inner_lr': 0.0011370492742948152, 'per_level_scale': 1.5433294452923831, 'log2_hashmap_size': 14, 'n_neurons': 16, 'n_hidden_layers': 1},
    # "display":  {'num_meta_loops': 87, 'num_inner_iterations': 892, 'meta_lr': 0.17739313111227162, 'meta_lr_decay': 0.9970071643795654, 'density_lambda': 0.00429705532711985, 'depth_lambda': 1.5661073960491976, 'depth_prescale': 6.958383156194673, 'inner_lr': 0.022796669368370354, 'per_level_scale': 1.3678934286042799, 'log2_hashmap_size': 14, 'n_neurons': 32, 'n_hidden_layers': 1},
    # "chair":    {'num_meta_loops': 100, 'num_inner_iterations': 1302, 'meta_lr': 0.07241830993075528, 'meta_lr_decay': 0.9886212516966383, 'density_lambda': 0.0060526093030672775, 'depth_lambda': 1.4389532772266946, 'depth_prescale': 4.153522968330267, 'inner_lr': 0.015751578148259063, 'per_level_scale': 1.5462192131433055, 'log2_hashmap_size': 15, 'n_neurons': 16, 'n_hidden_layers': 1},
    # "keyboard": {'num_meta_loops': 93, 'num_inner_iterations': 765, 'meta_lr': 0.13937898804779347, 'meta_lr_decay': 0.9993791561928042, 'density_lambda': 0.0006175540102148963, 'depth_lambda': 1.0304594028867418, 'depth_prescale': 14.581160829056236, 'inner_lr': 0.02409278959742059, 'per_level_scale': 1.7226395121510072, 'log2_hashmap_size': 15, 'n_neurons': 16, 'n_hidden_layers': 1},
    # "plant":    {'num_meta_loops': 40, 'num_inner_iterations': 538, 'meta_lr': 0.17289469857414308, 'meta_lr_decay': 0.9319940595484021, 'density_lambda': 5.146781777229611e-05, 'depth_lambda': 1.3891975891260886, 'depth_prescale': 7.831161575222925, 'inner_lr': 0.02091278572080373, 'per_level_scale': 1.4337710328882156, 'log2_hashmap_size': 14, 'n_neurons': 16, 'n_hidden_layers': 1},
    # "mouse":    {'num_meta_loops': 47, 'num_inner_iterations': 905, 'meta_lr': 0.03438585311019424, 'meta_lr_decay': 0.9843666566621579, 'density_lambda': 0.0563100843902192, 'depth_lambda': 0.6668097018769862, 'depth_prescale': 14.001918189938738, 'inner_lr': 0.0012099475358555603, 'per_level_scale': 1.411160538450563, 'log2_hashmap_size': 15, 'n_neurons': 16, 'n_hidden_layers': 1},
    # "car":      {'num_meta_loops': 62, 'num_inner_iterations': 1546, 'meta_lr': 0.07606911273549567, 'meta_lr_decay': 0.9328813993199957, 'density_lambda': 0.023319683408760564, 'depth_lambda': 0.9733728744011605, 'depth_prescale': 2.0965648695935934, 'inner_lr': 0.012878292671570373, 'per_level_scale': 1.4235588644682873, 'log2_hashmap_size': 14, 'n_neurons': 32, 'n_hidden_layers': 2}
}


if __name__ == "__main__":
    for category in ARCHITECTURES.keys():
        run_single_meta_iteration(
            category,
            **ARCHITECTURES[category],
            save_prior=True
        )