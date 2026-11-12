import time

import pandas as pd
from omegaconf import OmegaConf

from dataset import Dataset
from inference import run_inference
from train.ensembles import ensemble_experiment
from train.train_classical import  train
from config import config as cfg
from train.train_dl import train_dl

if __name__ == "__main__":
    start = time.time()
    dataset = Dataset()

    if cfg.general.train:
        trained_models, oof_preds, classic_scores = train(dataset)
        oof_preds, results, dl_scores = train_dl(dataset, oof_preds)
        cv_scores = {**classic_scores, **dl_scores}
        res = ensemble_experiment(dataset, oof_preds)
        cv_scores.update(res)
        summary = pd.DataFrame(cv_scores.items(), columns=['model', 'cv_rmse'])
        print(summary.to_string(index=False))

    if cfg.general.inference:
        run_inference(dataset)
        models_to_inference = []
        for key, value in OmegaConf.to_container(cfg.to_inference, resolve = True).items():
            if value: models_to_inference.append(key)
        print(f'Имя эксперимента : {cfg.general.experiment_name}')
        print(f'Модели для инференса : {models_to_inference}')
        print('Модели лежат в папке saved_models, путь к каждой модели задается через config (раздел inference)')
        print("Полученные предсказания лежат в папке inference_predicts и содержат названия текущего эксперимента")

    end = time.time()
    print(f'Time: {end - start}')