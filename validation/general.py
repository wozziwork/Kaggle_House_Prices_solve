import numpy as np

from config import config as cfg

from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import BaseEstimator


def mape_on_prices(y_true_log, y_pred_log):
    return mean_absolute_percentage_error(np.exp(y_true_log), np.exp(y_pred_log))

def build_cv() -> KFold:
    """
    Создает и возвращает объект KFold с настройками из конфига
    """
    cv = KFold(
        n_splits=cfg.validation.n_splits,
        shuffle=cfg.validation.shuffle,
        random_state=cfg.general.seed if cfg.validation.shuffle else None
    )
    return cv

def build_grid_search(
        model : BaseEstimator,
        param_grid : dict[str,list[int | float | str | bool]],
        cv
    ) -> GridSearchCV:
    """
    Создает объект GridSearchCV, настроенный по переданным аргументам и конфигу. Поиск не происходит
    Args:
         model: модель или пайплайн для валидации
         param_grid: словарь с названиями гиперпараметров и списками значений для перебора
         cv: схема разбиения данных для кросс-валидации
    Returns:
        Настроенный объект GridSearchCV
    """
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring={
            'rmse': 'neg_root_mean_squared_error',
            'mape': make_scorer(mape_on_prices, greater_is_better=False)
        },
        cv=cv,
        refit='rmse',
        n_jobs=cfg.search.n_jobs,
        verbose=cfg.search.verbose,
        return_train_score=True
    )
    return search