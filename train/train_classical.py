from functools import partial
import pandas as pd

from omegaconf import OmegaConf
from config import config as cfg

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Ridge

from catboost import  CatBoostRegressor
from lightgbm import  LGBMRegressor

from dataset import Dataset
from preprocessing import build_preprocessor, build_catboost_preprocessor, build_category_caster_preprocessor, \
    build_log_gr_liv_area_transformer
from artifacts import save_model, save_experiment_log
from validation.boosting import get_boosting_oof, run_boosting_experiment
from validation.classical import search_and_fit, grid_search_summary, fit_and_evaluate


def train(data : Dataset) -> tuple[dict[str, Pipeline], dict, dict]:
    """
    Создает, оценивает и обучает модели, выбранные в конфиге
    Args:
        data: датасет с X_train и y_train_log
    Returns:
        trained_models: словарь {имя_модели : обученный пайплайн}
        oof_preds : словарь с oof предсказаниями (только для catboost и lgbm)
        cv_scores : словарь {имя_модели: средний rmse на кросс-валидации}.
            Заполняется только для моделей, обучаемых без grid search
    """
    trained_models = {}
    oof_preds = {}
    results = {}
    cv_scores = {}
    cat_cols = data.X_train.select_dtypes(exclude='number').columns.tolist()
    X, y_log = data.X_train, data.y_train_log

    need_oof = ((cfg.to_train.averaging or cfg.to_train.stacking)
            and cfg.to_train.catboost and cfg.to_train.lgbm)

    if cfg.to_train.elastic_net:
        model_name  = 'elastic_net'
        base_preprocessor = build_preprocessor(X=X, use_ordinal=True)
        model = ElasticNet(**cfg.models[model_name])
        elastic_net = Pipeline([
            ('log_area', build_log_gr_liv_area_transformer()),
            ('preprocessor', base_preprocessor),
            ('model', model),
        ])

        if cfg.to_grid_search[model_name]:
            elastic_net, experiment_results, best_index = search_and_fit(elastic_net, X, y_log, model_name)
            # саммари отображает mean по каждой метрике из конфига (для лучшего эксперимента)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            elastic_net, experiment_results = fit_and_evaluate(elastic_net, X, y_log)
            cv_scores[model_name] = -experiment_results.loc['mean', 'test_rmse']

        trained_models[model_name] = elastic_net
        results[model_name] = experiment_results

    if cfg.to_train.ridge:
        model_name  = 'ridge'
        base_preprocessor = build_preprocessor(X=X, use_ordinal=True)
        model = Ridge(**cfg.models[model_name])
        ridge = Pipeline([
            ('log_area', build_log_gr_liv_area_transformer()),
            ('preprocessor', base_preprocessor),
            ('model', model),
        ])

        if cfg.to_grid_search[model_name]:
            ridge, experiment_results, best_index = search_and_fit(ridge, X, y_log, model_name)
            # саммари отображает mean по каждой метрике из конфига (для лучшего эксперимента)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            ridge, experiment_results = fit_and_evaluate(ridge, X, y_log)
            cv_scores[model_name] = -experiment_results.loc['mean', 'test_rmse']

        trained_models[model_name] = ridge
        results[model_name] = experiment_results

    if cfg.to_train.knn:
        base_preprocessor = build_preprocessor()
        model_name = 'knn'
        model = KNeighborsRegressor(**cfg.models[model_name])
        knn = Pipeline([
            ('preprocessor', base_preprocessor),
            ('model', model),
        ])

        if cfg.to_grid_search[model_name]:
            knn, experiment_results, best_index = search_and_fit(knn, X, y_log, model_name)
            # саммари отображает mean по каждой метрике из конфига (для лучшего эксперимента)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            knn, experiment_results = fit_and_evaluate(knn, X, y_log)
            cv_scores[model_name] = -experiment_results.loc['mean', 'test_rmse']

        trained_models[model_name] = knn
        results[model_name] = experiment_results

    if cfg.to_train.decision_tree:
        base_preprocessor = build_preprocessor(scale_numeric=False)
        model_name = 'decision_tree'
        model = DecisionTreeRegressor(**cfg.models[model_name])
        decision_tree = Pipeline([
            ('preprocessor', base_preprocessor),
            ('model', model),
        ])

        if cfg.to_grid_search[model_name]:
            decision_tree, experiment_results, best_index = search_and_fit(decision_tree, X, y_log, model_name)
            # саммари отображает mean по каждой метрике из конфига (для лучшего эксперимента)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            decision_tree, experiment_results = fit_and_evaluate(decision_tree, X, y_log)
            cv_scores[model_name] = -experiment_results.loc['mean', 'test_rmse']
        trained_models[model_name] = decision_tree
        results[model_name] = experiment_results

    if cfg.to_train.random_forest:
        base_preprocessor = build_preprocessor(scale_numeric=False)
        model_name = 'random_forest'
        model = RandomForestRegressor(**cfg.models[model_name])
        random_forest = Pipeline([
            ('preprocessor', base_preprocessor),
            ('model', model),
        ])
        if cfg.to_grid_search[model_name]:
            random_forest, experiment_results, best_index = search_and_fit(random_forest, X, y_log, model_name)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            random_forest, experiment_results = fit_and_evaluate(random_forest, X, y_log)
            cv_scores[model_name] = -experiment_results.loc['mean', 'test_rmse']
        trained_models[model_name] = random_forest
        results[model_name] = experiment_results

    if cfg.to_train.catboost:
        model_name = 'catboost'
        # преобразуем cat_features в кортеж для совместимости клонирования модели
        catboost_params = OmegaConf.to_container(cfg.models[model_name], resolve=True)

        cat_features = data.X_train.select_dtypes(exclude='number').columns.tolist()
        catboost_params['cat_features'] = tuple(cat_features)

        if catboost_params['early_stopping_rounds'] <= 0:
            catboost_params.pop('early_stopping_rounds')
            catboost_params['use_best_model'] = False

        catboost = CatBoostRegressor(**catboost_params)
        catboost_preprocessor = build_catboost_preprocessor()

        catboost, experiment_results, chosen_iterations = run_boosting_experiment(model_name, catboost, X, y_log, build_catboost_preprocessor)
        if not cfg.to_grid_search[model_name]:
            cv_scores[model_name] = experiment_results.loc['mean', 'val_rmse']

        # Обучаем на всех тренировочных данных без eval_set
        # число итераций выбрано по CV, поэтому early stopping отключаем.
        catboost.set_params(early_stopping_rounds=None, use_best_model=False, iterations=chosen_iterations)
        if need_oof:
            oof_preds[model_name] = get_boosting_oof(catboost, X, y_log, build_catboost_preprocessor)

        X_train = catboost_preprocessor.fit_transform(X)

        catboost.fit(X_train, y_log)
        catboost = Pipeline([
            ('preprocessor', catboost_preprocessor),
            ('model', catboost),
        ])

        trained_models[model_name] = catboost
        results[model_name] = experiment_results

    if cfg.to_train.lgbm:
        model_name = 'lgbm'
        lgbm_params = OmegaConf.to_container(cfg.models[model_name], resolve=True)
        lgbm = LGBMRegressor(**lgbm_params)
        build_lgbm_preprocessor = partial(build_category_caster_preprocessor, cat_cols)
        lgbm_preprocessor = build_lgbm_preprocessor()

        lgbm, experiment_results, chosen_iterations = run_boosting_experiment(model_name, lgbm, X, y_log, build_lgbm_preprocessor)
        if not cfg.to_grid_search[model_name]:
            cv_scores[model_name] = experiment_results.loc['mean', 'val_rmse']

        # Обучаем на всех тренировочных данных без eval_set
        # число итераций выбрано по CV, поэтому early stopping отключаем.
        lgbm.set_params(early_stopping_round=0, n_estimators=chosen_iterations)
        if need_oof:
            oof_preds[model_name] = get_boosting_oof(lgbm, X, y_log, build_lgbm_preprocessor)

        X_train = lgbm_preprocessor.fit_transform(X)
        lgbm.fit(X_train, y_log)
        lgbm = Pipeline([
           ('preprocessor', lgbm_preprocessor),
           ('model', lgbm),
        ])

        trained_models[model_name] = lgbm
        results[model_name] = experiment_results

    if cfg.general.save_models:
        for model_name, model in trained_models.items():
            save_model(model, model_name)

    if cfg.general.save_logs:
        for model_name, model in trained_models.items():
            save_experiment_log(model, results[model_name], model_name)

    if cfg.general.show_logs:
        for model_name, experiment_results in results.items():
            print(f'Result for {model_name}')
            print(pd.DataFrame(experiment_results).to_string() + '\n')


    return trained_models, oof_preds, cv_scores
