# Malicious Portable Executable Files-Detection-using-Machine-Learning

This project is about to find a best model to detect the maliciousness of the portable documents giving the best and accurate results.

### Project Organization

    ├── README.md
    ├── data
    │   ├── train                    <- Data from third-party sources.
    │   ├── test                     <- Intermediate data that has been transformed.
    │   ├── val                      <- The final, canonical data sets for modeling.
    │
    │
    ├── results                       <- Trained models
    ├── requirements.txt              <- The requirements file for reproducing the analysis environment,
    │                                    generated with `pip freeze > requirements.txt`
    └── src                           <- Source code for use in this project.
        ├── config
        │   ├── config.yaml           <- The default configs for the model.
        │
        ├── application_logger
        |   ├── custom_logger.py      <- Custom Application Logger to write Logs
        |
        ├── data
        │   ├── data_loader.py        <- DataLoader functions
        │   ├── ingest_data.py        <- Script to generate data
        │   └── make_dataset.py       <- Custom Implementation of data classes for train and test
        │
        ├── modeling                  <- Scripts to create the model's architecture
        │
        ├── tools                     <- Training loop and custom tools for the project
        │   ├── train.py              <- Script for combining all parts for training
    └── __main__.py                   <- File to train the model

## Running this project

```bash
pip install -r requirements.txt
```

```bash
cd ..
```

```bash
python -m Malicious-PE-Files-Detection-using-Machine-Learning
```


```bash
run the program
streamlit run <app name>
```
