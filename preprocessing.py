import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import torch

from config import config as cfg

def log_gr_liv_area(X : pd.DataFrame) -> pd.DataFrame:
    return X.assign(GrLivArea=np.log1p(X['GrLivArea']))
def build_log_gr_liv_area_transformer():
    return FunctionTransformer(log_gr_liv_area)


def build_preprocessor(
        scale_numeric : bool = True,
        sparse_output : bool = True,
        use_ordinal : bool = False,
        X : pd.DataFrame = None,
    ) -> ColumnTransformer:
    """
    Создает препроцессор для числовых и категориальных признаков
    Args:
        scale_numeric : флаг, отвечающий за наличие стандартизации
        sparse_output : флаг, отвечающий за формат выхода OneHotEncoder.
            True - разреженная матрица, False - обычный numpy массив
        use_ordinal : флаг, отвечающий за ordinal обработку некоторых категориальных признаков из конфига
        X : Таблица, обязательная при use_ordinal=True. Нужна для выбора столбцов, которые пойдут в one-hot
    Returns: необученный настроенный ColumnTransformer
    """
    if use_ordinal and X is None:
        raise ValueError('При use_ordinal=True нужно передать X')

    scaler = StandardScaler() if scale_numeric else 'passthrough'

    base_numeric_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.numeric.strategy)),
        ('scaler', scaler),
    ])

    base_categorical_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.categorical.strategy,
                                  fill_value=cfg.preprocessing.categorical.fill_value)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output)),
    ])

    transformers = [
        ('numeric', base_numeric_processing, make_column_selector(dtype_include='number')),
    ]

    onehot_cols = make_column_selector(dtype_exclude='number')
    if use_ordinal and X is not None:
        cat_cols = X.select_dtypes(exclude='number').columns
        orders = cfg.preprocessing.ordinal_categories
        ordinal_cols = list(orders)
        categories = [list(orders[col]) for col in ordinal_cols]
        onehot_cols = [col for col in cat_cols if col not in ordinal_cols]

        ordinal_processing = Pipeline([
            ('encoder', OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        transformers.append(('ordinal', ordinal_processing, ordinal_cols))
    transformers.append(('categorical', base_categorical_processing, onehot_cols))
    return ColumnTransformer(transformers)


def build_catboost_preprocessor() -> ColumnTransformer:
    """
    Создает препроцессор для catboost, заполняющий пропуски в категориальных признаках.
    Остальные столбцы передаются без преобразований.
    Returns: необученный настроенный ColumnTransformer, возвращающий DataFrame после трансформации
    """

    categorical_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.categorical.strategy,
                                  fill_value=cfg.preprocessing.categorical.fill_value))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_processing, make_column_selector(dtype_exclude='number')),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return preprocessor


class CategoryCaster(TransformerMixin, BaseEstimator):
    """
    Преобразует указанные категориальные столбцы DataFrame в тип 'category'.
    При fit запоминает категории из обучающих данных.
    При transform использует сохраненные категории. Неизвестные значения преобразуются в NaN.
    Остальные столбцы и исходный DataFrame не меняются
    Args:
        cat_cols: список названий категориальных столбцов
    """

    def __init__(self, cat_cols : list):
        self.category_dtypes_ = None
        self.cat_cols = cat_cols

    def fit(self, X : pd.DataFrame, y : pd.Series | None = None):
        self.category_dtypes_ = {
            col : pd.CategoricalDtype(categories=X[col].dropna().unique()) for col in self.cat_cols
        }
        return self

    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, dtype in self.category_dtypes_.items():
            X[col] = X[col].astype(dtype)
        return X

def build_category_caster_preprocessor(cat_cols) -> CategoryCaster:
    return CategoryCaster(cat_cols)

def build_embedding_preprocessor(X_train) -> ColumnTransformer:
    """
    Создает препроцессор для MLPEmb.
    Кодирует выбранные категориальные столбцы индексами для эмбеддингов, остальные - через one-hot
    Returns: необученный настроенный ColumnTransformer, возвращающий NumPy-массив после трансформации
    """
    embedding_cols = list(cfg.preprocessing.embedding_cols)
    onehot_cols = [col for col in X_train.select_dtypes(exclude='number').columns if col not in embedding_cols]

    numerical_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.numeric.strategy)),
        ('scaler', StandardScaler()),
    ])

    ordinal_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.categorical.strategy,
                                  fill_value=cfg.preprocessing.categorical.fill_value)),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)),
    ])

    onehot_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.categorical.strategy,
                                  fill_value=cfg.preprocessing.categorical.fill_value)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numerical_processing, make_column_selector(dtype_include='number')),
            ('onehot', onehot_processing, onehot_cols),
            ('categorical', ordinal_processing, embedding_cols)
        ]
    )
    return preprocessor

def prepare_embedding_inputs(X_processed, n_num, device):
    """
    Разбивает X_processed на два тензора, содержащих по порядку сначала численные(float32),
        а затем закодированные категориальные признаки(int32)
    Ко всем значениям категориальных признаков добавляется 1 (под неизвестную категорию)
    """
    X_num = torch.from_numpy(X_processed[:, :n_num].astype('float32')).to(device)
    X_cat = torch.from_numpy(X_processed[:, n_num:].astype('int32') + 1).to(device)
    return X_num, X_cat