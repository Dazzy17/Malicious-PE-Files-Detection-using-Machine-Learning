import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    log_loss,
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns

#from ..application_logger.custom_logger import CustomApplicationLogger


class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_tpfptnfn(self):
        logging.info("Entered the method named get_tpfptnfn")
        try:
            TP = np.sum(np.logical_and(self.y_pred == 1, self.y_true == 1))

            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = np.sum(np.logical_and(self.y_pred == 0, self.y_true == 0))

            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = np.sum(np.logical_and(self.y_pred == 1, self.y_true == 0))

            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = np.sum(np.logical_and(self.y_pred == 0, self.y_true == 1))
            logging.info("Exited the method named get_tpfptnfn")
            return TP, FP, TN, FN
        except Exception as e:
            logging.error("Exception occured in method named get_tpfptnfn: " + str(e))
            return None

    def get_confusion_metrics(self):
        logging.info("Entered the method named get_confusion_metrics")
        try:
            confusion = confusion_matrix(self.y_true, self.y_pred)
            plt.figure(figsize=(20, 4))
            labels = [0, 1]
            cmap = sns.light_palette("blue")
            plt.subplot(1, 3, 1)
            sns.heatmap(
                confusion,
                annot=True,
                cmap=cmap,
                fmt=".3f",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.xlabel("Predicted Class")
            plt.ylabel("Original Class")
            plt.title("Confusion matrix")

        except Exception as e:
            logging.error("Error occured: " + str(e))
            return None

    def get_precision_recall_score(self):
        logging.info("Entered the method get_precision_recall_score")
        try:
            precision_sc = precision_score(self.y_true, self.y_pred)
            recall_fc = recall_score(self.y_true, self.y_pred)
            logging.info("Exited get_precision_recall_score method")
            return precision_sc, recall_fc
        except Exception as e:
            logging.error(
                "Exception occured in method named get_precision_recall_score: "
                + str(e)
            )
            return None

    def f1_score(self):
        logging.info("Entered the method f1_score")
        try:
            f1_scores = f1_score(self.y_true, self.y_pred)
            logging.info("Exited get_precision_recall_score method")
            return f1_scores
        except Exception as e:
            logging.error(
                "Exception occured in method named get_precision_recall_score: "
                + str(e)
            )
            return None

    def get_score(self):
        logging.info("Entered the method get_score")
        try:
            score = 100 * (f1_score(self.y_true, self.y_pred, average="macro"))
            logging.info("Exited the method get_score")
            return score
        except Exception as e:
            logging.error("Exception occured in method named get_score: " + str(e))
            return None


class ModelEvaluater:
    def __init__(self, x_test, y_test) -> None:
        self.x_test = x_test
        self.y_test = y_test
        self.file_object = open(r"ModelEvaluation.txt", "a+")
        #self.logger = CustomApplicationLogger()

    def evaluate_trained_models(self, dt_model, rf_model, lgbm_model, xgb_model):

        dt_predictions = dt_model.predict(self.x_test)
        dt_evaluation = Evaluation(self.y_test, dt_predictions)
        dt_TP, dt_FP, dt_TN, dt_FN = dt_evaluation.get_tpfptnfn()

        # log the TP, FP, TN, FN values
        self.logger.logger(
            self.file_object,
            "Decision Trees:- TP : {} and FP : {} and TN : {} and FN : {}".format(
                dt_TP, dt_FP, dt_TN, dt_FN
            ),
        )
        dt_precision_sc, dt_recall_fc = dt_evaluation.get_precision_recall_score()
        dt_f1_score = dt_evaluation.f1_score()
        dt_score = dt_evaluation.get_score()
        self.logger.logger(
            self.file_object,
            "Decision Trees Precision Score is {} and Recall Score is {} and F1 Score is {} and Accuracy Score is {}".format(
                dt_precision_sc, dt_recall_fc, dt_f1_score, dt_score
            ),
        )

        rf_predictions = rf_model.predict(self.x_test)
        rf_evaluation = Evaluation(self.y_test, rf_predictions)
        rf_TP, rf_FP, rf_TN, rf_FN = rf_evaluation.get_tpfptnfn()

        # log the TP, FP, TN, FN values
        self.logger.logger(
            self.file_object,
            "Random Forest:- TP : {} and FP : {} and TN : {} and FN : {}".format(
                rf_TP, rf_FP, rf_TN, rf_FN
            ),
        )

        rf_precision_sc, rf_recall_fc = rf_evaluation.get_precision_recall_score()
        rf_f1_score = rf_evaluation.f1_score()
        rf_score = rf_evaluation.get_score()
        self.logger.logger(
            self.file_object,
            "Random Forest:- Precision Score is {} and Recall Score is {} and F1 Score is {} and Accuracy Score is {}".format(
                rf_precision_sc, rf_recall_fc, rf_f1_score, rf_score
            ),
        )

        lgbm_predictions = lgbm_model.predict(self.x_test)
        # round the lgbm predictions to 0 or 1
        lgbm_predictions = np.round(lgbm_predictions)
        lgbm_predictions = lgbm_predictions.astype(int)
        lgbm_evaluation = Evaluation(self.y_test, lgbm_predictions)
        lgbm_TP, lgbm_FP, lgbm_TN, lgbm_FN = lgbm_evaluation.get_tpfptnfn()
        print(
            "LightGBM TP : {} and FP : {} and TN : {} and FN : {}".format(
                lgbm_TP, lgbm_FP, lgbm_TN, lgbm_FN
            )
        )
        self.logger.logger(
            self.file_object,
            "LGBM:- TP : {} and FP : {} and TN : {} and FN : {}".format(
                lgbm_TP, lgbm_FP, lgbm_TN, lgbm_FN
            ),
        )

        lgbm_precision_sc, lgbm_recall_fc = lgbm_evaluation.get_precision_recall_score()
        lgbm_f1_score = lgbm_evaluation.f1_score()
        lgbm_score = lgbm_evaluation.get_score()
        self.logger.logger(
            self.file_object,
            "LightGBM Precision Score is {} and Recall Score is {} and F1 Score is {} and Accuracy Score is {}".format(
                lgbm_precision_sc, lgbm_recall_fc, lgbm_f1_score, lgbm_score
            ),
        )

        xgb_predictions = xgb_model.predict(self.x_test)
        xgb_evaluation = Evaluation(self.y_test, xgb_predictions)
        xgb_TP, xgb_FP, xgb_TN, xgb_FN = xgb_evaluation.get_tpfptnfn()
        self.logger.logger(
            self.file_object,
            "XGBoost:- TP : {} and FP : {} and TN : {} and FN : {}".format(
                xgb_TP, xgb_FP, xgb_TN, xgb_FN
            ),
        )
        xgb_precision_sc, xgb_recall_fc, = xgb_evaluation.get_precision_recall_score()
        xgb_f1_score = xgb_evaluation.f1_score()
        xgb_score = xgb_evaluation.get_score()
        self.logger.logger(
            self.file_object,
            "Precision Score is {} and Recall Score is {} and F1 Score is {} and Accuracy Score is {}".format(
                xgb_precision_sc, xgb_recall_fc, xgb_f1_score, xgb_score
            ),
        )
        if dt_score > rf_score and dt_score > lgbm_score and dt_score > xgb_score:
            self.logger.logger(
                self.file_object, "Decision Tree model is selected as the best model",
            )
            return dt_model

        if rf_score > dt_score and rf_score > lgbm_score and rf_score > xgb_score:
            self.logger.logger(
                self.file_object, "Random Forest model is selected as the best model",
            )
            return rf_model
        if lgbm_score > dt_score and lgbm_score > rf_score and lgbm_score > xgb_score:
            self.logger.logger(
                self.file_object, "LightGBM model is selected as the best model",
            )
            return lgbm_model
        if xgb_score > dt_score and xgb_score > rf_score and xgb_score > lgbm_score:
            self.logger.logger(
                self.file_object, "XGBoost model is selected as the best model",
            )
            return xgb_model
