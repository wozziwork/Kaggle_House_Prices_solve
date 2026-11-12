import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from artifacts import load_model, save_predictions, load_dl_checkpoint
from dataset import Dataset
from dl_models.mlp import MLP
from dl_models.mlp_emb import MLPEmb
from preprocessing import prepare_embedding_inputs
from config import config as cfg




def run_inference(dataset : Dataset) -> None:
    """
    Выполняет инференс моделей и ансамблей, включенных в cfg.to_inference.
    Загружает модели и препроцессоры согласно конфигу, получает предсказания
        для dataset.X_test и сохраняет CSV с Id и SalePrice
    """
    X_test = dataset.X_test
    house_ids = dataset.test_house_ids
    to_inference = OmegaConf.to_container(cfg.to_inference, resolve=True)

    classic_models = ['elastic_net', 'ridge', 'knn', 'decision_tree', 'random_forest', 'catboost', 'xgboost', 'lgbm']
    mlp_models = ['mlp', 'mlp_emb']
    ensembles = ['averaging', 'stacking']
    device = cfg.general.device

    for model_name, flag in to_inference.items():
        if flag and model_name in classic_models:
            model : Pipeline = load_model(cfg.inference[model_name])
            test_preds = np.exp(model.predict(X_test))
            save_predictions(house_ids, test_preds, model_name)

        if flag and model_name in mlp_models:
            file_name = cfg.inference[model_name]
            checkpoint, preprocessor = load_dl_checkpoint(file_name)
            X_processed = preprocessor.transform(X_test)
            if model_name == 'mlp':
                X = (torch.from_numpy(X_processed.astype('float32')).to(device),)
                mlp_model = MLP(**checkpoint['model_config'])
            else:
                n_num = checkpoint['model_config']['num_numeric']
                X = prepare_embedding_inputs(X_processed, n_num, device)
                mlp_model = MLPEmb(**checkpoint['model_config'])

            mlp_model.load_state_dict(checkpoint['model_state_dict'])
            mlp_model.to(device)
            mlp_model.eval()

            with torch.no_grad():
                preds = np.exp(mlp_model(*X).to('cpu').numpy().ravel())
            save_predictions(house_ids, preds, model_name)

        if flag and model_name in ensembles:
            catboost = (load_model(cfg.inference.averaging.catboost))
            catboost_preds = catboost.predict(X_test).ravel()
            lgbm = load_model(cfg.inference.averaging.lgbm)
            lgbm_preds = lgbm.predict(X_test).ravel()

            if model_name == 'averaging':
                average_preds = np.exp((catboost_preds + lgbm_preds) / 2)
                save_predictions(house_ids, average_preds, 'averaging')

            if model_name == 'stacking':
                meta_model = load_model(cfg.inference.stacking.ridge)
                X_meta = pd.DataFrame({
                    'catboost': catboost_preds,
                    'lgbm': lgbm_preds,
                })
                stacking_preds = np.exp(meta_model.predict(X_meta))
                save_predictions(house_ids, stacking_preds, 'stacking')

