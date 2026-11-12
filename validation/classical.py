import pandas as pd

from omegaconf import OmegaConf
from config import config as cfg

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.base import BaseEstimator

from validation.general import mape_on_prices, build_grid_search, build_cv


def evaluate_model_cv(model : BaseEstimator, X_train : pd.DataFrame, y_train : pd.Series, cv : KFold) -> dict:
    """
    Валидация переданной модели по переданным аргументам.
    Исходная модель остается неизменной, на каждом фолде обучается ее копия
    Args:
         model: модель для валидации
         X_train : тренировочные данные
         y_train : ответы к тренировочным данным
         cv: схема разбиения данных для кросс-валидации
    Returns:
        cv_results: словарь с результатами кросс-валидации
    """
    cv_results = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring={
            'rmse' : 'neg_root_mean_squared_error',
            'mape' : make_scorer(mape_on_prices, greater_is_better=False)
        },
        cv=cv,
        return_train_score=True)
    return cv_results


def grid_search_summary(result : dict, best_index : int) -> pd.DataFrame:
    """
    Фильтрует таблицу с результатами GridSearchCV, оставляет только средние валидационные метрики из конфига по индексу лучшего эксперимента
    Args:
        result: словарь с результатами
        best_index: индекс лучшего сочетания гиперпараметров
    Returns:
        summary: Датафрейм, где строки - средние валидационные метрики и среднее время обучения для лучшего сочетания гиперпараметров,
                                         столбец - название эксперимента
    """
    cols =['mean_fit_time', 'mean_test_rmse', 'mean_test_mape']
    summary = pd.DataFrame(result).loc[[best_index], cols]
    summary.index = [cfg.general.experiment_name]
    return summary.T


def fit_and_evaluate(model : Pipeline, X_train : pd.DataFrame, y_train : pd.Series) -> tuple[Pipeline, pd.DataFrame]:
    """
    Оценка модели и ее обучение. Модель обучается на логарифмах цен. RMSE рассчитывается на той же шкале.
    Args:
        model: исходная модель
        X_train : тренировочные данные
        y_train : ответы к тренировочным данным
    Returns:
        model: обученная модель
        cv_results: DataFrame с результатами по фолдам и строкой средних значений
    """
    cv = build_cv()
    cv_results = pd.DataFrame(evaluate_model_cv(model, X_train, y_train, cv))
    cv_results.loc['mean'] = cv_results.mean()
    model.fit(X_train, y_train)
    return model, cv_results


def search_and_fit(model : Pipeline, X_train : pd.DataFrame, y_train : pd.Series, model_name : str) -> tuple[Pipeline, dict, int]:
    """
    Перебор гиперпараметров модели, заданных в конфиге.
    Возвращает модель с лучшим результатом, а также результаты поиска
    Args:
        model: исходная модель
        X_train : тренировочные данные
        y_train : ответы к тренировочным данным
        model_name: имя модели
    Returns:
        best_model: обученная на всех исходных данных модель с лучшим, среди проверенных по выбранной метрике сочетанием, гиперпараметров
        search_results: словарь с результатами GridSearch
        best_index: индекс лучшего сочетания гиперпараметров
    """
    cv = build_cv()
    # достаю параметры для GridSearchCV из конфига вида ИмяШагаПайплайна__параметр : [значения]
    param_grid = OmegaConf.to_container(cfg.param_grids.get(model_name,{}))
    grid_search = build_grid_search(model, param_grid, cv)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    search_results = grid_search.cv_results_
    best_index = grid_search.best_index_

    return best_model, search_results, best_index