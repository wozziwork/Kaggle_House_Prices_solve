from pathlib import Path
from omegaconf import OmegaConf


config = {
    'general' : {
        'experiment_name' : 'FINAL',
        'seed' : 67,
        'train' : True,
        'inference' : False, # модели для инференса указываются в разделе inference
        'train_dl_on_all_data' : True, # если нужно получить и сохранить модель, иначе для dl моделей просто проводится поиск или cv
        'save_models' : False,
        'save_logs' : True,
        'show_logs' : True,
        'boosting_verbose': 0,
        'device' : 'cpu',
        'show_graphs' : False,
        'save_oof' : True,
        'tqdm_disable' : True,
    },
    'paths' : {
        'TRAIN_PATH' : Path(__file__).resolve().parent / "data" / "train.csv",
        'TEST_PATH' : Path(__file__).resolve().parent / "data" / "test.csv",
        'MODELS_DIR' : Path(__file__).resolve().parent / "saved_models",
        'OOF_PREDS' : Path(__file__).resolve().parent / "oof_preds" / "oof_preds_FINAL.csv",
        'INFERENCE' : Path(__file__).resolve().parent / "inference_predicts",
    },
    'dataset' : {
        'dropped_columns' : [],
        'additional_columns' : ['OverallQualCond'],
        'categorical_overrides' : ['MSSubClass', 'MoSold'], #меняет тип на category
    },
    'preprocessing' : {
        'numeric' : {
            'strategy' : 'median',
        },
        'categorical' : {
            'strategy' : 'constant',
            'fill_value' : 'Unknown',
        },
        'ordinal_categories' : {
         },
        'embedding_cols' : ['Neighborhood', 'MSSubClass']
    },
    'validation' : {
        'n_splits' : 5,
        'shuffle' : True,
    },
    'models' : {
        'elastic_net' : {
            'alpha' : 0.0005,
            'l1_ratio': 0.775,
            'selection' : 'random',
            'random_state' : '${general.seed}',
            'max_iter' : 10000,
        },
        'ridge' : {
            'solver' : 'auto',
            'alpha' : 6
        },
        'knn' : {
            'n_neighbors' : 10,
            'metric' : 'manhattan',
            'weights' : 'distance',
            'algorithm' : 'brute'
        },
        'decision_tree' : {
            'max_depth' : 12,
            'random_state' : '${general.seed}',
            'min_samples_split' : 28,
            'min_samples_leaf' : 6,
            'criterion' : 'squared_error',
            'splitter' : 'best',
            'max_features' : 0.7,
        },
        'random_forest' : {
            'max_depth' : 11,
            'max_features' : 0.5,
            'min_samples_leaf' : 2,
            'min_samples_split' : 3,
            'n_estimators' : 140,
            'n_jobs' : 1,
            'random_state' : '${general.seed}',
        },
        'catboost' : {
            'random_seed' : '${general.seed}',
            'verbose' : '${general.boosting_verbose}',
            'thread_count': 2,
            'early_stopping_rounds' : 0,
            'use_best_model' : False,
            'iterations' : 600,
            'max_depth' : 5,
            'learning_rate' : 0.08,
            'l2_leaf_reg' : 0.2,
            'random_strength' : 0.3,
            'subsample' : 0.8,
            'bootstrap_type': 'MVS',
        },
        'lgbm' : {
            'random_state': '${general.seed}',
            'verbosity': -1,
            'n_jobs': 4,
            'early_stopping_round' : 0,
            'learning_rate':  0.013664,
            'reg_alpha' : 0.005566,
            'num_leaves' : 26,
            'n_estimators': 1100,
            'reg_lambda' : 0.0199,
            'min_child_samples' : 15,
            'colsample_bytree' : 0.561659,
            'subsample' :  0.69448,
            'subsample_freq' : 1,
        },
        'mlp' : {
            'hidden_dims' : [32, 64, 32, 256],
            'batch_norm' : True,
            'dropout' : 0.2557301094436889,
            'activation_func' : 'silu'
        },
        'mlp_emb' : {
            'hidden_dims' : [64, 256],
            'embedding_dims' : [12, 2],
            'batch_norm': True,
            'dropout': 0.4279249228516782,
            'activation_func': 'silu'
        },
        'stacking': {
            'alpha' : 1,
            'catboost_iterations' : 600,
            'lgbm_n_estimators' : 1100,
            'cv' : True # флаг проведения кросс-валидации
        }
    },
    'to_train': {
        'elastic_net': True,
        'ridge' : False,
        'knn': False,
        'decision_tree': False,
        'random_forest': False,
        'catboost': True,
        'lgbm': True,
        'mlp': True,
        'mlp_emb': True,
        'averaging': False,
        'stacking': False,
    },
    'dl_train' : {
        'mlp' : {
            'num_workers' : 0,
            'epochs' : 151,
            'batch_size' : 32,
            'lr' : 0.0010500797171677308,
            'optimizer_name': 'adam',
            'weight_decay': 0.050216560009322016,
            'early_stopping_patience' : 0,
        },
        'mlp_emb' : {
            'num_workers': 0,
            'epochs': 247,
            'batch_size': 32,
            'lr': 0.0011794174096597123,
            'optimizer_name': 'adam',
            'weight_decay': 0.002877948856855312,
            'early_stopping_patience': 0,
        }
    },
    'search' : {
        'summarize_results': False,
        'n_jobs' : -1,
        'verbose' : 0,
    },
    'param_grids' : {
        'elastic_net' : {
            'model__alpha' : [0.0001, 0.001, 0.003, 0.01],
            'model__l1_ratio': [0.05, 0.1, 0.2],
        },
        'ridge' : {
            'model__alpha' : [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        },
        'knn' : {
            'model__n_neighbors': [9,10,11,12,13,14,15],
            'model__weights': ['uniform', 'distance'],
        },
        'decision_tree' : {
            'model__max_depth' : [i for i in range(1,15)],
            'model__min_samples_leaf' : [i for i in range(1, 10)],
            'model__max_features' : [None, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        'random_forest' : {
            'model__n_estimators' : [140],
            'model__max_depth' : [11],
            'model__min_samples_leaf': [2],
            'model__min_samples_split' : [3],
            'model__max_features' : [0.5]
        },
        'catboost': {
            'model__max_depth': [5],
            'model__learning_rate': [0.08],
            'model__l2_leaf_reg': [0.2],
            'random_strength': [0.2, 0.3],

        },
        'lgbm': {
            'model__num_leaves': [16],
            'model__learning_rate': [0.05],
            'model__min_child_samples': [22],
            'model__reg_lambda': [1],
            'model__subsample': [1.0],
            'model__subsample_freq': [1],
            'model__colsample_bytree': [1.0],
        },
        'mlp': {
            'hidden_dims': [[32,32,32]],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'lr': [0.002],
            'activation_func': ['leacky_relu'],
            'batch_norm': [True],
            'batch_size': [24],
            'optimizer_name': ['adam'],
            'weight_decay': [0.0, 0.1, 0.01],
        },
        'mlp_emb': {
            'dropout': [0.1, 0.2, 0.3],
            'lr': [0.0015, 0.002, 0.0025],
            'hidden_dims': [[128,128]],
            'batch_norm': [True],
            'batch_size': [24],
            'activation_func': ['leacky_relu'],
            'optimizer_name': ['adam'],
            'weight_decay': [0.01],
        },
    },

    'to_grid_search' : {
        'elastic_net' : False,
        'ridge' : False,
        'knn' : False,
        'decision_tree' : False,
        'random_forest' : False,
        'catboost' : False,
        'lgbm' : False,
        'mlp' : False,
        'mlp_emb' : False,
    },
    'inference' : {
        'elastic_net' : 'FINAL_elastic_net',
        'ridge' : 'FINAL_ridge',
        'knn' : 'FINAL_knn',
        'decision_tree' : 'FINAL_decision_tree',
        'random_forest' : 'FINAL_random_forest',
        'catboost' : 'FINAL_catboost',
        'lgbm' : 'FINAL_lgbm',
        'mlp' : 'FINAL_mlp',
        'mlp_emb' : 'FINAL_mlp_emb',
        'averaging' : {
            'catboost' : 'FINAL_catboost',
            'lgbm' : 'FINAL_lgbm',
        },
        'stacking' : {
            'ridge' : 'FINAL_meta_model_ridge', # кетбуст и лгбм берутся по пути выше
        },
    },
    'to_inference' : {
        'elastic_net' : False,
        'ridge' : False,
        'knn' : False,
        'decision_tree' : False,
        'random_forest' : False,
        'catboost' : False,
        'lgbm' : False,
        'mlp' : False,
        'mlp_emb' : False,
        'averaging': False,
        'stacking': False,
    }
}

config = OmegaConf.create(config)

