import numpy as np
import pandas as pd
from config import config as cfg

class Dataset:
    """
    Загрузка train/test данных, а также их подготовка
    """
    def __init__(self):
        self.train = pd.read_csv(cfg.paths.TRAIN_PATH)
        self.test = pd.read_csv(cfg.paths.TEST_PATH)
        self.train_house_ids = self.train['Id']
        self.test_house_ids = self.test['Id']

        self.X_train, self.X_test, self.y_train, self.y_train_log = self._prepare_data()


    def _prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Подготовка данных:
            Назначает необходимым категориальным столбцам тип 'category'
            отделяет целевую переменную и удаляет столбец с Id + столбцы из конфига

        Returns: X_train, X_test, y_train, y_train_log
        """
        dropped_cols = cfg.dataset.dropped_columns + ['Id']
        categorical_overrides = list(cfg.dataset.categorical_overrides)
        additional_columns = list(cfg.dataset.additional_columns)

        #доп обработка (fe)
        X_train = self.train.copy()
        X_test = self.test.copy()


        # YrSold - YearBuilt
        if 'HouseAge' in additional_columns:
            X_train['HouseAge'] = X_train['YrSold'] - X_train['YearBuilt']
            X_test['HouseAge'] = X_test['YrSold'] - X_test['YearBuilt']
        # YrSold - YearRemodAdd`
        if 'RepairAge' in additional_columns:
            X_train['RepairAge'] = X_train['YrSold'] - X_train['YearRemodAdd']
            X_test['RepairAge'] = X_test['YrSold'] - X_test['YearRemodAdd']
        # 2ndFlrSF > 0
        if '2ndFlrPresent' in additional_columns:
            X_train['2ndFlrPresent'] = (X_train['2ndFlrSF'] > 0).astype(int)
            X_test['2ndFlrPresent'] = (X_test['2ndFlrSF'] > 0).astype(int)
        # GarageArea > 0
        if 'garagePresent' in additional_columns:
            X_train['garagePresent'] = (X_train['GarageArea'] > 0).astype(int)
            X_test['garagePresent'] = (X_test['GarageArea'] > 0).astype(int)
        # YrSold + '_' + MoSold
        if 'dateSold' in additional_columns:
            X_train['date'] = X_train['YrSold'].astype(str) + '-' + X_train['MoSold'].astype(str).str.zfill(2)
            X_test['date'] = X_test['YrSold'].astype(str) + '-' + X_test['MoSold'].astype(str).str.zfill(2)
        if 'LivAreaQual' in additional_columns:
            X_train['LivAreaQual'] = X_train['GrLivArea'] * X_train['OverallQual']
            X_test['LivAreaQual'] = X_test['GrLivArea'] * X_test['OverallQual']
        if 'BsmtSF' in additional_columns:
            total_train = X_train['TotalBsmtSF']
            total_test = X_test['TotalBsmtSF']
            X_train['BsmtSF'] = (total_train - X_train['BsmtUnfSF']) / total_train.where(total_train > 0)
            X_test['BsmtSF'] = (total_test - X_test['BsmtUnfSF']) / total_test.where(total_test > 0)
        if 'OverallQualCond' in additional_columns:
            X_train['OverallQualCond'] = X_train['OverallQual'] * X_train['OverallCond']
            X_test['OverallQualCond'] = X_test['OverallQual'] * X_test['OverallCond']
        if 'BathCnt' in additional_columns:
            X_train['BathCnt'] = 0.5 * X_train['HalfBath'] + 0.5 * X_train['BsmtHalfBath'] + X_train['FullBath'] + X_train['BsmtFullBath']
            X_test['BathCnt'] = 0.5 * X_test['HalfBath'] + 0.5 * X_test['BsmtHalfBath'] + X_test['FullBath'] + X_test['BsmtFullBath']
        if 'TotalSF' in additional_columns:
            X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
            X_test['TotalSF'] = X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']

        X_train = X_train.astype({col : 'category' for col in categorical_overrides})
        X_test = X_test.astype({col : 'category' for col in categorical_overrides})

        #отделяем значения целевой переменной
        y_train = self.train['SalePrice']
        y_train_log = np.log(self.train['SalePrice'])

        X_train = X_train.drop(columns=['SalePrice'] + dropped_cols)
        X_test = X_test.drop(columns=dropped_cols)


        return X_train, X_test, y_train, y_train_log