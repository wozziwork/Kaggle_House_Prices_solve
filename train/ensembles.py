import pandas as pd

from omegaconf import OmegaConf
from config import config as cfg

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

from dataset import Dataset
from validation.stacking import evaluate_stacking_cv
from artifacts import save_oof_predictions, save_experiment_log, save_model


def averaging(oof_res : pd.DataFrame) -> float:
    """
    Оценка ансамбля усреднения catboost, lgbm на их out of fold предсказаниях.
    Возвращает rmse для ансамбля усреднением.
    Args:
        oof_res :датафрейм, содержащий oof предсказания всех моделей ансамбля и правильные ответы
    Return:
        rmse_averaging : rmse для ответов, полученных усреднением предсказаний
    """
    averaging_preds = (oof_res['lgbm'] + oof_res['catboost'])/2
    rmse_averaging = root_mean_squared_error(oof_res['y_true_log'], averaging_preds)
    return rmse_averaging

def stacking_cv(meta_model : Ridge, dataset : Dataset) -> float:
    """
    Оценка стекинга catboost, lgbm.
    Возвращает RMSE по всем OOF-прогнозам внешней кросс-валидации
    """
    # создаем модели, которые будут обучаться для стекинга
    lgbm = (LGBMRegressor(**OmegaConf.to_container(cfg.models.lgbm, resolve=True))
            .set_params(n_estimators=cfg.models.stacking.lgbm_n_estimators ,early_stopping_round=0))
    catboost = (CatBoostRegressor(**OmegaConf.to_container(cfg.models.catboost, resolve=True))
            .set_params(iterations=cfg.models.stacking.catboost_iterations ,early_stopping_rounds=None, use_best_model=False))
    stacking_preds, rmse = evaluate_stacking_cv(
        dataset.X_train, dataset.y_train_log,
        catboost, lgbm, meta_model
    )
    return rmse

def stacking_train(meta_model : Ridge, oof_results : pd.DataFrame) -> Ridge:
    """
    Обучает переданный Ridge на OOF-прогнозах CatBoost и LightGBM с таргетом y_true_log и возвращает обученную модель
    """
    y_true = oof_results['y_true_log']
    X = oof_results[['catboost', 'lgbm']]
    meta_model.fit(X, y_true)
    return meta_model


def ensemble_experiment(dataset : Dataset, oof_preds : dict) -> dict[str, float]:
    """
    Эксперимент-оценка ансамблей(stacking с Ridge, averaging) на OOF-прогнозах логарифма цены базовых моделей.
    Оценка стекинга выполняется отдельно через вложенную кросс-валидацию (при stacking.cv=True).
    Также обучает Ridge (метамодель) на всей OOF-таблице и сохраняет его согласно конфигу.
    OOF-таблица формируется из словаря oof_preds или читается из файла, путь к которому указан в конфиге.
    Выводит и сохраняет результаты оценок согласно настройкам конфига.
    Args:
        dataset : объект Dataset для получения столбцов Id, y_true_log и CV стекинга
        oof_preds :словарь, содержащий OOF-прогнозы логарифма цены базовых моделей (если они обучались)
    Return:
        res : словарь с rmse ансамблей
    """
    res = {}
    if cfg.to_train.stacking or cfg.to_train.averaging:
        # Если 2 базовые модели ансамбля обучаются, то
        # согласно конфигу происходит формирование и сохранение таблицы или только формирование (без сохранения)
        if all([cfg.to_train.catboost, cfg.to_train.lgbm]):
            if cfg.general.save_oof:
                oof_res = save_oof_predictions(dataset, oof_preds, cfg.paths.OOF_PREDS)
            else:
                oof_res = save_oof_predictions(dataset, oof_preds, None)
        else:
            # Если 2 базовых модели в данном эксперименте не обучаются, то читаем oof предсказания по пути из конфига
            oof_res = pd.read_csv(cfg.paths.OOF_PREDS)

        if cfg.to_train.averaging:
            rmse_averaging = averaging(oof_res)
            results = pd.DataFrame([{'rmse_averaging': rmse_averaging}])
            if cfg.general.save_logs:
                save_experiment_log(model=None, results=results, model_name='averaging')
            res['averaging'] = rmse_averaging

        if cfg.to_train.stacking:
            alpha = cfg.models.stacking.alpha
            meta_model = Ridge(alpha=alpha)
            if cfg.models.stacking.cv:
                rmse_stacking = stacking_cv(meta_model, dataset)
                results = pd.DataFrame([{'rmse_stacking': rmse_stacking}])
                if cfg.general.save_logs:
                    save_experiment_log(model=meta_model, results=results, model_name='stacking')
                res['stacking'] = rmse_stacking
            meta_model = stacking_train(meta_model, oof_res)

            if cfg.general.save_models:
                save_model(meta_model, 'meta_model_ridge')
    return res