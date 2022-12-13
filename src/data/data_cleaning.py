import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# from transformers import LukeModel
from .data_ingeston import DataIngestion


class DataCleaning:
    def __init__(self):
        pass

    def read_data(self):
        try:
            class1 = pd.read_csv(r"C:\Users\User\Desktop\pro\data\bengin.csv")
            class2 = pd.read_csv(r"C:\Users\User\Desktop\pro\data\malware.csv")
            class1["Label"] = 0
            class2["Label"] = 1
            df = pd.concat([class1, class2])
            df = df.sample(frac=1).reset_index(drop=True)
            return df

        except Exception as e:
            raise Exception(f"Exception occured while reading data: {e}")

    def drop_columns_having_nan_vals(self):
        # remove column where nan values > 0.8%
        train = self.read_data()
        stats = []
        for col in train.columns:
            stats.append(
                (
                    col,
                    train[col].nunique(),
                    train[col].isnull().sum() * 100 / train.shape[0],
                    train[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                    train[col].dtype,
                )
            )

        stats_df = pd.DataFrame(
            stats,
            columns=[
                "Feature",
                "Unique_values",
                "Percentage of missing values",
                "Percentage of values in the biggest category",
                "type",
            ],
        )

        # get the columns where the percentage of nan values greater than 0.8%
        colsZ_to_drop = stats_df[stats_df["Percentage of missing values"] > 0.7][
            "Feature"
        ].tolist()
        train = train.drop(colsZ_to_drop, axis=1)
        return train

    def fill_nan_values(self):
        train = self.drop_columns_having_nan_vals()
        # fill nan values with mean of the column
        train = train.fillna(train.mean())
        return train

    def categorical_encoding(self):
        train = self.fill_nan_values()
        # select categorical columns
        cat_cols = [col for col in train.columns if train[col].dtype == "object"]
        # encode categorical columns
        train = pd.get_dummies(train, columns=cat_cols)
        return train

    def remove_features_which_have_high_correlation(self):
        train = self.categorical_encoding()
        # select features with high correlation
        corr_matrix = train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        train = train.drop(to_drop, axis=1)
        return train

    def remove_features_with_low_variance(self):
        train = self.remove_features_which_have_high_correlation()
        # select features with low variance
        sel = VarianceThreshold(threshold=0.8)
        train = sel.fit_transform(train)
        return train
