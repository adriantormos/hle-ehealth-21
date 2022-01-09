# HLE: eHealth-KD Challenge 2021
This repository contains the code for the implementation of the proposed approach by Adrián Tormos Llorente and Ferran Agulló López for the eHealth-KD Challenge of the 2021 year.

## Files structure
The files are distributed in the following way:
- 2021/ref: datasets used for the training and validating
- 2021/eval: datasets used for the evaluation
- 2021/submissions: submission results shown in the report
- models: empty directory where the output models of the code are stored
- scripts: code from the challenge
- src: our own code for the implementation of the proposed approach
- main_prepare_models.py: script used to run the preparation of the models
- main_tasks.py: script used to run the multiple tasks of the challenge

## How to set up

- install python-3.9 and pip
- install the required python packages that appear in the requirements file (pip install -r requirements.txt)

## How to run

- prepare models: Run the main_prepare_models script with the desired input parameters (check the argparse parameters)
- run tasks: RUn the main_tasks script with the desired input parameters (check the argparse parameters)
