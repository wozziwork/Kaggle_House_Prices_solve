import numpy as np
import torch
import pandas as pd

from train.train_mlp import train_mlp_fold
from train.train_mlp_emb import train_mlp_emb_fold

from validation.general import build_cv

from typing import Any

from sklearn.model_selection import ParameterGrid

from statistics import mean

from preprocessing import prepare_embedding_inputs


def mlp_cv(
        X : pd.DataFrame,
        y : pd.Series,
        model_name : str,
        **params : Any,
        ) -> tuple[dict, list, float, np.ndarray]:
    """
    Кросс-валидации MLP- или MLPEmb-модели с переданными параметрами архитектуры и обучения.
    Создаваемая модель определяется аргументом model_name (ожидается mlp_emb или mlp)
    Args:
        X : данные для кросс-валидации
        y : ответы к данным
        model_name : наименование модели
        params : словарь, содержащий настройки обучения и архитектуры модели
    Returns:
        experiment_results : словарь с результатами кросс-валидации, содержит:
            выбранную лучшую эпоху по validation rmse
            значение loss на выбранной эпохе
            значение validation rmse на выбранной эпохе
            значения loss на трейне, а также loss и rmse на валидации за последние 5 эпох
        full_history : список словарей историй, по одному на фолд
        mean_rmse: среднее val_rmse по фолдам (для оптимальных состояний моделей если early_stopping включен)
        oof_preds : numpy массив out of fold предсказаний для всех объектов из X
    """
    cv = build_cv()
    experiment_results = {}
    fold_n = 0
    full_history = []
    oof_preds = np.full(len(y), np.nan)

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_name == 'mlp_emb':
            train_func = train_mlp_emb_fold

        elif model_name == 'mlp':
            train_func = train_mlp_fold
        else:
            raise ValueError(f'Неизвестное имя модели: {model_name}')

        model, preprocessor, history, checkpoint , best_epoch = train_func(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, **params)

        device = next(model.parameters()).device

        if model_name == 'mlp_emb':
            model.eval()
            with torch.no_grad():
                X_val_processed = preprocessor.transform(X_val)
                n_num = preprocessor.output_indices_['categorical'].start
                X_num, X_cat = prepare_embedding_inputs(X_val_processed, n_num, device)
                preds = model(X_num, X_cat)
                oof_preds[val_idx] = preds.cpu().numpy().ravel()
        elif model_name == 'mlp':
            model.eval()
            with torch.no_grad():
                X_val_processed = preprocessor.transform(X_val)
                X_val_processed = torch.from_numpy(X_val_processed.astype('float32')).to(device)
                preds = model(X_val_processed)
                oof_preds[val_idx] = preds.cpu().numpy().ravel()


        full_history.append(history)

        fold_n += 1
        experiment_results[fold_n] = {
            'best_epoch': best_epoch + 1 if best_epoch >= 0 else len(history['train_loss']),
            'val_loss_at_best_epoch': history['val_loss'][best_epoch],
            'val_rmse_at_best_epoch': history['val_rmse'][best_epoch],
            'last_5_val_rmse' : history['val_rmse'][-5:],
            'last_5_train_loss': history['train_loss'][-5:],
            'last_5_val_loss': history['val_loss'][-5:],
        }
    mean_rmse = mean(fold_res['val_rmse_at_best_epoch'] for fold_res in experiment_results.values())
    return experiment_results, full_history, mean_rmse, oof_preds

def search_mlp_cv(
        X_train : pd.DataFrame,
        y_train : pd.Series,
        model_name : str,
        param_grid : dict,
        base_params : dict
    ) -> tuple[float, dict, dict]:
    """
    Поиск по сетке для MLP- или MLPEmb-модели по переданным параметрам.
    Args:
        X_train : данные для тренировки
        y_train : ответы
        model_name : наименование модели
        param_grid : словарь со списками значений перебираемых параметров
        base_params : базовые параметры; настройки param_grid их переопределяют
    Returns:
        best_rmse : лучший средний rmse на кросс-валидации
        best_params : настройки, при которых был достигнут best_rmse
        search_results: словарь с результатами поиска по сетке.
            Содержит параметры и их значения и средний rmse на кросс-валидации для этих параметров
    """

    best_rmse= float('inf')
    best_params = None
    search_results = {}
    iteration_cnt = 0
    for params in ParameterGrid(param_grid):
        candidate_params = {**base_params, **params}
        experiment_results, _, mean_rmse,  _ = mlp_cv(X_train, y_train, model_name, **candidate_params)

        search_results[iteration_cnt] = {
            'params' : candidate_params,
            'mean_rmse' : mean_rmse,
        }
        iteration_cnt+=1

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = candidate_params

    return best_rmse, best_params, search_results