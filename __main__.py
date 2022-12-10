from .src.data.data_ingeston import DataIngestion
from .src.data.data_cleaning import DataCleaning
from .src.modelling.model_developmentnew import ModelTraining, BestModelFinder
from .src.evaluation.evaluation import ModelEvaluater
from sklearn.model_selection import train_test_split
import joblib

data_clean = DataCleaning()

train = data_clean.categorical_encoding()
X = train.drop(["Label"], axis=1)
y = train["Label"]



# 0  1 0 1 0 1 0 1 0 1 0 1 0 10 1 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# drop OH_DLLchar9_0, OH_DLLchar9_1
X_train.drop(["OH_DLLchar9_0", "OH_DLLchar9_1"], axis=1, inplace=True)
X_test.drop(["OH_DLLchar9_0", "OH_DLLchar9_1"], axis=1, inplace=True)

print(X_train.columns)
BestModelFinders = BestModelFinder()
dt_model, rf_model, xgb_model, lgbm_model = BestModelFinders.train_all_models(
X_train, y_train, X_test, y_test)

print(X_test.shape)
print('Hello')


# # pickle import
# import joblib

# joblib.dump(dt_model, "dt_model.pkl")
# joblib.dump(rf_model, "rf_model.pkl")
# joblib.dump(xgb_model, "xgb_model.pkl")
# joblib.dump(lgbm_model, "lgbm_model.pkl")
#dt_model = joblib.load(
#    r"E:\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\results\dt_model.pkl"
#)
#rf_model = joblib.load(
#    r"E:\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\results\rf_model.pkl"
#)
#lgbm_model = joblib.load(
    #r"E:\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\results\lgbm_model.pkl"
#)
xgb_model = joblib.load(
    r"C:\Users\User\Desktop\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\results\xgb_model.pkl")

    #only used xgb model here
    

#model_evaluater = ModelEvaluater(X_test, y_test)
#model_evaluater.evaluate_trained_models(dt_model, rf_model, lgbm_model, xgb_model)
