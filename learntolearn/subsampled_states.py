####### FORMAT #######
# "task_name": {
#    "inner_lr": (min, max),
#    "log2_hashmap_size": (choices,),
#    "per_level_scale": (choices,),
#    "n_neurons": (choices,),
#    "n_hidden_layers": (choices,)
# }


SUBSAMPLED_STATES = {
    "laptop": { # final
        "inner_lr": (0.0190, 0.0249),
        "log2_hashmap_size": (14, 15,),
        "per_level_scale": (1.25992,),
        "n_neurons": (64,),
        "n_hidden_layers": (1, 2,)
    },
    "mug": { # final
        "inner_lr": (0.00525, 0.0118),
        "log2_hashmap_size": (14, 15, 16, 17),
        "per_level_scale": (1.25992, 1.31951, 1.44727),
        "n_neurons": (16, 64, 128),
        "n_hidden_layers": (1, 2)
    },
    "ball": {
        "inner_lr": (0.0140, 0.0150),
        "log2_hashmap_size": (14,),
        "per_level_scale": (1.44727,),
        "n_neurons": (16,),
        "n_hidden_layers": (1,)
    },
    "display": { # final
        "inner_lr": (0.0193, 0.0235),
        "log2_hashmap_size": (14, 15,),
        "per_level_scale": (1.25992, 1.44727,),
        "n_neurons": (64, 128,),
        "n_hidden_layers": (1, 2,)
    },
    "book": { # final
        "inner_lr": (0.0188, 0.0238),
        "log2_hashmap_size": (14, 15, 16,),
        "per_level_scale": (1.25992, 1.31951,),
        "n_neurons": (16, 64,),
        "n_hidden_layers": (2, 3, )
    },
    "keyboard": { # final
        "inner_lr": (0.0226, 0.0248),
        "log2_hashmap_size": (14, ),
        "per_level_scale": (1.25992, 1.31951, 1.38191, 1.44727,),
        "n_neurons": (16, 64, 128,),
        "n_hidden_layers": (1, 2,)
    },
    "mouse": { # final
        "inner_lr": (0.0163, 0.225),
        "log2_hashmap_size": (14, 15, 17,),
        "per_level_scale": (1.25992, 1.31951, 1.38191, 1.44727,),
        "n_neurons": (16, 64,),
        "n_hidden_layers": (1, 2, 3,)
    },
    "plant": { # final
        "inner_lr": (0.0140, 0.0150),
        "log2_hashmap_size": (14,),
        "per_level_scale": (1.25992,),
        "n_neurons": (16,),
        "n_hidden_layers": (1,)
    },
}