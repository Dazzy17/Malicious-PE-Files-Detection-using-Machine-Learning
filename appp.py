import os

import pandas as pd
import streamlit as st
from extract_features import pe_features
from nbformat import write
from numpy import zeros

st.title("Malware Prediction App")
st.write("Welcome to my first app")


## make someone upload a file
uploaded_file = st.file_uploader("Upload a file", type=["exe"])

st.write("Here's your file:", uploaded_file)
# get the file name


class DataCleaning:
    def __init__(self):
        pass

    def read_data(self):
        try:
            #Creating classes to read those stored benign and malicious datasets
            class1 = pd.read_csv(
                r"C:\Users\User\Desktop\Computing Project\new-project\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\data\bengin.csv"
            )
            class2 = pd.read_csv(
                r"C:\Users\User\Desktop\Computing Project\new-project\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\data\malware.csv"
            )
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

    def fill_nan_values(self, train):
        # train = self.drop_columns_having_nan_vals()
        # fill nan values with mean of the column
        train = train.fillna(train.mean())
        return train

    def categorical_encoding(self, train):
        train = self.fill_nan_values(train)
        # select categorical columns
        cat_cols = [col for col in train.columns if train[col].dtype == "object"]
        # encode categorical columns
        train = pd.get_dummies(train, columns=cat_cols)
        return train


if uploaded_file is not None:
    st.write("File uploaded successfully")
    # save the uploaded exe file
    with open(
        r"C:\Users\User\Desktop\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\inference\{}".format(
            uploaded_file.name
        ),
        "wb",
    ) as f:
        # write it in inference folder
        f.write(uploaded_file.read())
        f.close()
        uploaded_file.close()

        features = pe_features(
            r"C:\Users\User\Desktop\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\inference",
            "pred.csv",
        )

        features.create_dataset()
        pred = pd.read_csv("pred.csv")
        data_clean = DataCleaning()
        # os.remove(
        # r"C:\Users\User\Desktop\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\inference\{}".format(
        #  uploaded_file.name
        # )
        # )

        train = data_clean.categorical_encoding(pred)

        cols_to_save = [
            "e_cblp",
            "e_cp",
            "e_cparhdr",
            "e_maxalloc",
            "e_sp",
            "e_lfanew",
            "NumberOfSections",
            "CreationYear",
            "FH_char0",
            "FH_char1",
            "FH_char2",
            "FH_char3",
            "FH_char4",
            "FH_char5",
            "FH_char6",
            "FH_char7",
            "FH_char8",
            "FH_char9",
            "FH_char10",
            "FH_char11",
            "FH_char12",
            "FH_char13",
            "FH_char14",
            "MajorLinkerVersion",
            "MinorLinkerVersion",
            "SizeOfCode",
            "SizeOfInitializedData",
            "SizeOfUninitializedData",
            "AddressOfEntryPoint",
            "BaseOfCode",
            "BaseOfData",
            "ImageBase",
            "SectionAlignment",
            "FileAlignment",
            "MajorOperatingSystemVersion",
            "MinorOperatingSystemVersion",
            "MajorImageVersion",
            "MinorImageVersion",
            "MajorSubsystemVersion",
            "MinorSubsystemVersion",
            "SizeOfImage",
            "SizeOfHeaders",
            "CheckSum",
            "Subsystem",
            "OH_DLLchar0",
            "OH_DLLchar1",
            "OH_DLLchar2",
            "OH_DLLchar3",
            "OH_DLLchar4",
            "OH_DLLchar5",
            "OH_DLLchar6",
            "OH_DLLchar7",
            "OH_DLLchar8",
            "OH_DLLchar10",
            "SizeOfStackReserve",
            "SizeOfStackCommit",
            "SizeOfHeapReserve",
            "SizeOfHeapCommit",
            "LoaderFlags",
            "OH_DLLchar9_NoPacker",
        ]

        # delete all the columns ex
        train = train[cols_to_save]
        # fill NA values in train df with 0
        train = train.fillna(0)
        st.write(train)
        st.write(train.shape)

        import joblib

        # load the dt_model
        xgb_model = joblib.load(
            r"C:\Users\User\Desktop\Malicious-Portable-Executable-Files-Detection-using-Machine-Learning\results\xgb_model.pkl"
        )
        # predict the class of the file
        pred_class = xgb_model.predict(train)
        # if pred_class == 0:
        #     st.write("The file is benign")
        # else
        #     st.write("The file is malicious")
        # take the last value of the dataframe

        if pred_class == 0:
            st.write("Result: This file is not malicious")
        else:
            st.write("Result: This file is malicious")

        # remove the file from the inference folder

        # cloes the working of the file

        # remove the file from the pred.csv
        # os.remove("pred.csv")
