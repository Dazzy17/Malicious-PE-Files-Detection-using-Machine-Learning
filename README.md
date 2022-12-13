# Malicious Portable Executable Files-Detection-using-Machine-Learning

This project is about to find a best model to detect the maliciousness of the portable documents giving the best and accurate results.

### Project Organization

    ├── README.md
    ├── data (all data are collected from various datasets)
    │   ├── benign.csv                   <- Data that are benign files (not malicious)
    │   ├── malware.csv                  <- Data that are malicious 
    │             
    ├── results                       <- Trained models
    ├── requirements.txt              <- The requirements file for reproducing the analysis. The listed libraries are required to install in order to run this properly.
    └── src                           <- Source code for use in this project.
        │
        ├── application_logger
        |   ├── custom_logger.py      <- Custom Application Logger to write Logs
        |
        ├── data
        │   ├── data_cleaning.py        <- cleaning dataset from duplicates, null values, etc.
        │   ├── data_ingestion.py       <- Script to generate data
        |
        ├── modeling                  <- Scripts to create the model's architecture
        |   ├── application_logger.py   <- Logging all the application developments.
            ├── modeldevelopmentnew.py  <- Developing models for training purposes.
        |
        ├── eda                   <- explorotary data analysis
        │   ├── eda.ipynb
        |
        ├── evaluation             <- evaluation of the data
        |   ├──  evaluation.py     <- this will return accuracy values of all algorithms.
    
    ├── app.py                   <- application to run for the results
    ├── distribution_of_the_class.png    <- the visualization of how data spread
    ├── extract_features.py          <- extracting features from the dataset to train purposes
    ├── ModelEvaluation.txt   <- this is where accuracy can seen.
    └── __main__.py                   <- File to train the model 
    
  

## Running this project

```bash
pip install -r requirements.txt
```

```bash
cd ..
```

```bash
python -m pro (folder name)
```


```bash
run the program
streamlit run <app name>
```

When the application run on the sceeen asking for the file to submit to check whether they are malicious or not :

👇

![image](https://user-images.githubusercontent.com/65618412/207314682-3475aefb-ad99-481a-bcee-947f2f38b585.png)

(Submission of an application called Obisidian.exe - note taking application)

The outcome is 0 which is benign for the submitted file as following :

![image](https://user-images.githubusercontent.com/65618412/207315977-a94c7ce3-8082-413d-8295-f1beb9bc8ee4.png)

Please contact me via LinkedIn if any problems arise on your side 😀
