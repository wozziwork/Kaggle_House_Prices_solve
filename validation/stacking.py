from functools import partial

import numpy as np
import pandas as pd
import copy

from sklearn.metrics import root_mean_squared_error

from validation.general import build_cv
from validation.boosting import get_boosting_oof

from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from preprocessing import build_catboost_preprocessor, build_category_caster_preprocessor


def evaluate_stacking_cv(
        X : pd.DataFrame,
        y : pd.Series,
        catboost : CatBoostRegressor,
        lgbm : LGBMRegressor,
        meta_model : Ridge,
    ) -> tuple[np.ndarray, float]:
    """
    Обучение и валидация стекинга из catboost, lgbm.
    Для каждого внешнего фолда:
        1) На его обучающей части проводится внутренняя кросс-валидация и
            полученные oof предсказания используются для обучения мета-модели.
        2) Новые базовые модели обучаются на всей обучающей части внешнего фолда
            и получают предсказания для внешней валидационной части.
        3) Обученная метамодель преобразует их в итоговые предсказания.
    Предсказания всех внешних фолдов собираются в массив в исходном порядке, после чего считается rmse
    Args:
        X : данные для оценки стекинга
        y : логарифмы правильных цен домов
        catboost : настроенная необученная catboost-модель (early_stopping отключен, количество деревьев фиксировано)
        lgbm : настроенная lgbm-модель (early_stopping отключен, количество деревьев фиксировано)
        meta_model : необученная мета-модель с настроенной архитектурой (не изменяется в ходе cv)
    Returns:
        stacking_preds : массив oof предсказаний логарифмов цен(порядок сохранен)
        rmse : подсчитанный rmse для всех oof-предсказаний стекинга
    """
    outer_cv = build_cv()
    stacking_preds = np.full(len(y), np.nan)
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    catboost = copy.deepcopy(catboost)
    catboost.set_params(cat_features=tuple(cat_cols))
    build_lgbm_preprocessor = partial(build_category_caster_preprocessor, cat_cols)

    for train_outer_idx, val_outer_idx in outer_cv.split(X, y):
        # для обучения базовых моделей, получения их oof предсказаний и обучения стекинга
        X_train, y_train = X.iloc[train_outer_idx], y.iloc[train_outer_idx]
        X_val, y_val = X.iloc[val_outer_idx], y.iloc[val_outer_idx] # для валидации стекинга

        #обучение стекинга на ооф предсказаниях
        catboost_oof = get_boosting_oof(catboost, X_train, y_train, build_catboost_preprocessor)
        lgbm_oof = get_boosting_oof(lgbm, X_train, y_train, build_lgbm_preprocessor)
        # записываем ооф предсказания
        X_meta = pd.DataFrame({
            'catboost' : catboost_oof,
            'lgbm' : lgbm_oof
        })

        ridge = copy.deepcopy(meta_model)
        ridge.fit(X_meta, y_train)

        # обучение всех моделей и получение по их предиктам предсказания стекинга
        catboost_val_preprocessor = build_catboost_preprocessor()
        catboost_val = copy.deepcopy(catboost)

        lgbm_val_preprocessor = build_lgbm_preprocessor()
        lgbm_val = copy.deepcopy(lgbm)

        X_train_lgbm = lgbm_val_preprocessor.fit_transform(X_train, y_train)
        X_train_catboost = catboost_val_preprocessor.fit_transform(X_train, y_train)

        lgbm_val.fit(X_train_lgbm, y_train)
        catboost_val.fit(X_train_catboost, y_train)


        X_val_lgbm = lgbm_val_preprocessor.transform(X_val)
        X_val_catboost = catboost_val_preprocessor.transform(X_val)

        X_val_meta = pd.DataFrame({
            'catboost' : catboost_val.predict(X_val_catboost),
            'lgbm' : lgbm_val.predict(X_val_lgbm)
        })
        stacking_preds[val_outer_idx] = ridge.predict(X_val_meta)

    return stacking_preds, root_mean_squared_error(y, stacking_preds)



