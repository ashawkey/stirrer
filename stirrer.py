import os
import gc
import sys
import glob
import time
import json
import signal
import shutil
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received: 
            self.old_handler(*self.signal_received)


lgb2xgb = {
        "regression": "reg:squarederror",
        "binary": "binary:logistic",
    }

lgb2cat = {
        "rmse": "RMSE",
    }

class Trainer(object):
    def __init__(
            self,
            workspace,
            model_name='model',
            backend='lgb',
            lr=0.05,
            objective='regression',
            metric='rmse',
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=10,
            seed=42,
            feature_fraction=0.8,
            reg_lambda=1,
            reg_alpha=0,
            max_depth=10,
            feval=None,
        ):

        self.workspace = workspace
        self.model_name = model_name
        self.backend = backend

        os.makedirs(self.workspace, exist_ok=True)

        self.num_boost_round = num_boost_round # num_iterations, num_trees
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval # frequency for eval (evary #verbose iteration)
        self.metric = metric
        self.objective = objective
        self.seed = seed
        self.lr = lr
        self.feval = feval

        if backend == 'lgb':
            self.params = {
                "objective": objective,
                "metric": metric,
                "seed": seed,
                "learning_rate": lr,
                "boosting": "gbdt",
                "num_threads": 0, # use default number of threads in OpenMP (OMP_NUM_THREADS)
                "max_depth": max_depth, # max depth of tree
                "num_leaves": 2**8, # max number of leaves in one tree
                "lambda_l2": reg_lambda, # L2 regularization
                "lambda_l1": reg_alpha, # L1 regularization
                "feature_fraction": feature_fraction, # only select #fraction of features on each iteration
                "bagging_fraction": 0.75, # only select #fraction of data
                "bagging_freq": 10, # frequency for bagging (every #freq iteration)
                "verbose": 1,
            }

        elif backend == 'xgb':
            self.params = {
                "objective": lgb2xgb[objective],
                "eval_metric": metric,
                "seed": seed,
                "learning_rate": lr, # eta
                "booster": "gbtree",
                "nthread": 0,
                "max_depth": max_depth,
                "lambda": reg_lambda, # L2 regularization
                "alpha": reg_alpha, # L1 regularization
                "colsample_bytree": feature_fraction, # only select #fraction of features on each iteration
                "subsample": 0.75, # only select #fraction of data
                "verbosity": 1,
            }

        elif backend == 'cat':
            print("Catboost is not supported now.")
            exit()
            self.params = None



    def train(self, X_train, y_train, X_valid, y_valid):
        evals_result = {}
        if self.backend == 'lgb':
            train_set = lgb.Dataset(X_train, label=y_train)
            valid_set = lgb.Dataset(X_valid, y_valid)
            
            model = lgb.train(
                    params=self.params,
                    num_boost_round=self.num_boost_round,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=self.verbose_eval,
                    feval=self.feval,
                    train_set=train_set,
                    valid_sets=[train_set, valid_set],
                    evals_result=evals_result,
                )

            ax = lgb.plot_importance(model, max_num_features=10)
            plt.savefig(f"{self.workspace}/{self.model_name}_importance.png")

        elif self.backend == 'xgb':
            train_set = xgb.DMatrix(X_train, label=y_train)
            valid_set = xgb.DMatrix(X_valid, label=y_valid)

            model = xgb.train(
                    params=self.params,
                    num_boost_round=self.num_boost_round,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=self.verbose_eval,
                    dtrain=train_set,
                    evals=[(train_set, 'train'), (valid_set, 'valid')],
                    evals_result=evals_result,
                )

            ax = xgb.plot_importance(model)
            plt.savefig(f"{self.workspace}/{self.model_name}_importance.png")

        elif self.backend == 'cat':

            model = CatBoostRegressor(
                    iterations=self.num_boost_round,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    eval_metric=lgb2cat[self.metric],
                    random_seed=self.seed,
                    metric_period=self.verbose_eval,
                    od_type='Iter', # overfitting detection
                    od_wait=20,
                    train_dir=self.workspace,
                )

            model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    use_best_model=True,
                    verbose=self.verbose_eval,
                )


            
        with open(f"{self.workspace}/{self.model_name}_results.json", 'w') as f:
            json.dump(evals_result, f)

        self.save_checkpoint(model)

        return model
    
    def save_checkpoint(self, model):
        with DelayedKeyboardInterrupt():
            file_path = f"{self.workspace}/{self.model_name}.txt"
            print(f"--> Saved model: {file_path}")
            os.makedirs(self.workspace, exist_ok=True)
            if self.backend == 'lgb':
                model.save_model(file_path)
            elif self.backend == 'xgb':
                model.save_model(file_path)
            elif self.backend == 'cat':
                model.save_model(file_path)



class Predictor(object):
    def __init__(
            self, 
            workspace,
        ):
        self.workspace = workspace
        self.models = []
        self.backends = []

    def load_checkpoint(self, path, backend='lgb'):
        print(f"--> Loaded model: {path}")
        if backend == 'lgb':
            model = lgb.Booster(model_file=path)
        elif backend == 'xgb':
            model = xgb.Booster(model_file=path) 
        elif backend == 'cat':
            model = CatBoostRegressor()
            model.load_model(path)

        self.models.append(model)
        self.backends.append(backend)

    def predict(self, X):
        preds = []
        for model, backend in zip(self.models, self.backends):
            if backend == 'lgb':
                pred = model.predict(X, num_iteration=model.best_iteration)
            elif backend == 'xgb':
                pred = model.predict(xgb.DMatrix(X), ntree_limit=model.best_ntree_limit)
            elif backend == 'cat':
                pred = model.predict(X)
            preds.append(pred)
        # figure out best weight?
        preds = np.mean(np.stack(preds, 0), 0)
        return preds

"""
    if OBJECTIVE == 'multi_class':
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds = np.argmax(preds, axis=1)
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds = np.argmax(test_preds, axis=1)
    elif OBJECTIVE == 'regression':
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds = np.round(np.clip(preds, 0, 10)).astype(int)
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)
"""
