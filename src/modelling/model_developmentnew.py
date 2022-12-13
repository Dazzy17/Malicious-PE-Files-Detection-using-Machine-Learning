import logging

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .application_logger import CustomApplicationLogger


class Hyperparameters_Optimization:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize_lg(self, trial):
        C = trial.suggest_loguniform("C", 1e-7, 10.0)
        solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))
        clf = LogisticRegression(C=C, solver=solver)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)

        return val_accuracy

    def optimize_decisiontrees(self, trial):
        criterion = trial.suggest_categorical("criterion", ("gini", "entropy"))
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_xgboost(self, trial):
        logging.info("optimize_xgboost")
        train_data = xgb.DMatrix(self.x_train, label=self.y_train)
        test_data = xgb.DMatrix(self.x_test, label=self.y_test)

        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 10.0),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-7, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-7, 10.0),
            "scale_pos_weight": trial.suggest_loguniform("scale_pos_weight", 1e-7, 10.0),
        }
        clf = xgb.XGBClassifier(**param) 
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_randomforest(self, trial):
        logging.info("optimize_randomforest")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def Optimize_LightGBM(self, trial):

        logging.info("Optimize_LightGBM")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy


class ModelTraining:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logistic_regression(self, fine_tuning=True):
        logging.info("Entered for training Logistic regression model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_logistic_regression, n_trials=100)
                trial = study.best_trial
                C = trial.params["C"]
                solver = trial.params["solver"]
                max_iter = trial.params["max_iter"]
                clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
                clf.fit(self.x_train, self.y_train)

                print("C : {} and solver : {} and max_iter : {}".format(C, solver, max_iter))

                logging.info("Logistic Regression model fine tuned ands trained")
                return clf

            else:
                print(self.y_train)
                logging.info("Logistic Regression model is being trained")
                model = LogisticRegression(C=0.00015200974256076671, solver="lbfgs", max_iter=641)
                model.fit(self.x_train, self.y_train)
                logging.info("Logistic Regression model trained")
                return model

        except Exception as e:
            logging.error("Error in training Logistic regression model")
            logging.error(e)
            return None

    def decision_trees(self, fine_tuning=True):
        logging.info("Entered for training Decision Trees model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_decisiontrees, n_trials=100)
                trial = study.best_trial
                criterion = trial.params["criterion"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                clf = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, min_samples_split=7
                )

                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Decision Trees model")
            logging.error(e)
            return None

    def random_forest(self, fine_tuning=True):
        logging.info("Entered for training Random Forest model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = RandomForestClassifier(n_estimators=92, max_depth=19, min_samples_split=4)
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def LightGBM(self, fine_tuning=True):
        logging.info("Entered for training LightGBM model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.Optimize_LightGBM, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                learning_rate = trial.params["learning_rate"]
                reg = LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = LGBMRegressor(n_estimators=200, learning_rate=0.01, max_depth=20)
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training LightGBM model")
            logging.error(e)
            return None

    def xgboost(self, fine_tuning=True):
        logging.info("Entered for training XGBoost model")
        try:
            if fine_tuning:
                hy_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                max_depth = trial.params["max_depth"]
                min_child_weight = trial.params["min_child_weight"]
                subsample = trial.params["subsample"]
                colsample_bytree = trial.params["colsample_bytree"]
                reg_alpha = trial.params["reg_alpha"]
                reg_lambda = trial.params["reg_lambda"]
                clf = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                )
                print("Best parameters : ", trial.params)
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                params = {
                    "objective": "binary:logistic",
                    "use_label_encoder": True,
                    "base_score": 0.5,
                    "booster": "gbtree",
                    "colsample_bylevel": 1,
                    "colsample_bynode": 1,
                    "colsample_bytree": 0.9865465799558366,
                    "enable_categorical": False,
                    "gamma": 0,
                    "gpu_id": -1,
                }
                clf = XGBClassifier(**params)
                clf.fit(self.x_train, self.y_train)
                return clf

        except Exception as e:
            logging.error("Error in training XGBoost model")
            logging.error(e)
            return None


class BestModelFinder:
    def __init__(self) -> None:
        self.file_object = open(
            r"C:\Users\User\Desktop\pro\logs\BestModelFinderLogs.txt",
            "a+",
        )
        self.logger = CustomApplicationLogger()

    def train_all_models(self, x_train, y_train, x_test, y_test):
        try:
            # Logisitc Regression Model

            model_train = ModelTraining(x_train, y_train, x_test, y_test)
            lg_model = model_train.logistic_regression(fine_tuning=False)

            if lg_model is not None:
                self.logger.logger(
                    self.file_object,
                    "Logistic Regression model is trained successfully",
                )

            # decision_trees
            model_train = ModelTraining(x_train, y_train, x_test, y_test)
            dt_model = model_train.decision_trees(fine_tuning=False)
            if dt_model is not None:
                self.logger.logger(
                    self.file_object,
                    "Decision Tree model is trained successfully",
                )
            # Random Forest

            model_train = ModelTraining(x_train, y_train, x_test, y_test)
            rf_model = model_train.random_forest(fine_tuning=False)
            if rf_model is not None:
                self.logger.logger(
                    self.file_object,
                    "Random Forest model is trained successfully",
                )
            # LightGBM

            model_train = ModelTraining(x_train, y_train, x_test, y_test)
            lgbm_model = model_train.LightGBM(fine_tuning=False)
            if lgbm_model is not None:
                self.logger.logger(
                    self.file_object,
                    "LightGBM model is trained successfully",
                )

            # XGBoost

            model_train = ModelTraining(x_train, y_train, x_test, y_test)
            xgb_model = model_train.xgboost(fine_tuning=True)
            if xgb_model is not None:
                self.logger.logger(
                    self.file_object,
                    "XGBoost model is trained successfully",
                )

            # print the best model among these by comparing the score of all models

            return dt_model, rf_model, xgb_model, lgbm_model, lg_model
        except Exception as e:
            # self.logger.logger(self.file_object, str(e))
            raise e
