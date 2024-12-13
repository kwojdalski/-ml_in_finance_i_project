# Machine Learning in Finance I

## Overview

This project aims to leverage machine learning techniques to address specific challenges and opportunities in the finance domain. 
The focus is on developing predictive models and data-driven insights that can enhance decision-making processes in financial applications.

## Table of Contents

- [Machine Learning in Finance I](#machine-learning-in-finance-i)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [For venv](#for-venv)
    - [For conda](#for-conda)
  - [Usage](#usage)
  - [Data](#data)
    - [Data Sources](#data-sources)
  - [Models](#models)
  - [Results](#results)

## Installation

To set up this project on your local machine, follow these steps:

1. **Clone the repository**:
   
```bash
git clone https://github.com/kwojdalski/ml_in_finance_i_project.git
cd ml_in_finance_i_project
```

2. **Install required packages**: It is recommended to create a virtual environment to manage dependencies. You can do this using venv or conda:

### For venv

```python
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### For conda

```python
conda create --name finance_ml python=3.12
conda activate finance_ml
conda install --file requirements.txt
```

Make sure you have Python and pip installed on your machine.

## Usage

To run the project, execute the main script or the appropriate Jupyter notebook:

```python
python main.py
```

or

Launch Jupyter Notebook:
```python
jupyter notebook
Open notebook.ipynb and run the cells.
Provide specific instructions on how to input data and interpret results if applicable.
```

## Data

### Data Sources

Detail the datasets used in this project. If applicable, include sources like:

- Historical stock prices ([QRT Problem](https://www.quantrocket.com/qrt-problem/))


## Models

Model Selection
Outline the machine learning models used in this project, including:

Linear Regression
Random Forest
Neural Networks
Hyperparameter Tuning
Describe the methods used for hyperparameter tuning (e.g., Grid Search, Random Search).

## Results

Summarize the main findings from the model evaluations, including:

Performance metrics (Accuracy, F1 Score, AUC-ROC, etc.)
Visualizations of the results (confusion matrix, ROC curve, etc.)
Key Insights
Discuss any significant insights or takeaways from the results.


