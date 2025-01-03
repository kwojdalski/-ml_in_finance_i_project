import logging as log

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

kfold = 2
ID_COLS = [
    "ID",
    "STOCK",
    "DATE",
    "INDUSTRY",
    "INDUSTRY_GROUP",
    "SECTOR",
    "SUB_INDUSTRY",
]


# %%
def evaluation(model, X: pd.DataFrame, Y: pd.Series, kfold: int) -> None:
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
    scores2 = cross_val_score(model, X, Y, cv=kfold, scoring="precision")
    scores3 = cross_val_score(model, X, Y, cv=kfold, scoring="recall")
    # The mean score and standard deviation of the score estimate
    log.info(
        "Cross Validation Accuracy: %0.5f (+/- %0.2f)" % (scores1.mean(), scores1.std())
    )
    log.info(
        "Cross Validation Precision: %0.5f (+/- %0.2f)"
        % (scores2.mean(), scores2.std())
    )
    log.info(
        "Cross Validation Recall: %0.5f (+/- %0.2f)" % (scores3.mean(), scores3.std())
    )
    return


# %%
def compute_roc(
    Y: pd.Series, y_pred: pd.Series, plot: bool = True
) -> tuple[float, float, float]:
    fpr = dict()
    tpr = dict()
    auc_score = dict()
    fpr, tpr, _ = roc_curve(Y, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="blue", label="ROC curve (area = %0.2f)" % auc_score)
        plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
        plt.legend(loc="upper right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.show()
    return fpr, tpr, auc_score


# %%
def feature_importance(model, features: list[str]) -> plt.Axes:
    feature_importances = pd.DataFrame(model.feature_importances_)
    feature_importances = feature_importances.T
    feature_importances.columns = [features]

    sns.set(rc={"figure.figsize": (13, 12)})
    # fig = sns.barplot(data=feature_importances.values.tolist(), orient='h', order=feature_importances.mean().sort_values(ascending=False).index)
    fig = sns.barplot(
        data=feature_importances,
        orient="h",
        order=feature_importances.mean()
        .sort_values(ascending=False)
        .reset_index()["level_0"],
    )
    fig.set(title="Feature importance", xlabel="features", ylabel="features_importance")

    return fig


# %%
def model_fit(
    model,
    X: pd.DataFrame,
    Y: pd.Series,
    features: list[str],
    performCV: bool = True,
    roc: bool = False,
    printFeatureImportance: bool = False,
) -> None:
    # Fitting the model on the data_set
    model.fit(X[features], Y)

    # Predict training set:
    predictions = model.predict(X[features])
    predprob = model.predict_proba(X[features])[:, 1]

    # Create and print confusion matrix
    cfm = confusion_matrix(Y, predictions)
    log.info("\nModel Confusion matrix")
    log.info(cfm)

    # Print model report:
    log.info("\nModel Report")
    log.info("Accuracy : %.4g" % accuracy_score(Y.values, predictions))

    # Perform cross-validation: evaluate using 10-fold cross validation
    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    if performCV:
        evaluation(model, X[features], Y, kfold)
    if roc:
        compute_roc(Y, predictions, plot=True)

    # Print Feature Importance:
    if printFeatureImportance:
        feature_importance(model, features)


def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data, preprocess by:
    - Loading CSV files
    - Dropping NA values and ID columns
    - Computing moving averages and mean returns
    - Converting target to binary

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - train_df: Preprocessed training dataframe
            - test_df: Preprocessed test dataframe
    """
    # Load data
    x_train: pd.DataFrame = pd.read_csv("./data/x_train.csv")
    y_train: pd.DataFrame = pd.read_csv("./data/y_train.csv")
    train_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)
    test_df: pd.DataFrame = pd.read_csv("./data/x_test.csv")

    # Clean data
    train_df = train_df.dropna()
    train_df = train_df.drop(ID_COLS, axis=1)
    test_df = test_df.dropna()
    test_df = test_df.drop(ID_COLS, axis=1)

    # Calculate mean returns and moving averages
    for df in [train_df, test_df]:
        df["Mean"] = df[[f"RET_{i}" for i in range(1, 21)]].mean(axis=1)
        df["MA5"] = df[[f"RET_{i}" for i in range(1, 7)]].mean(axis=1)
        df["MA10"] = df[[f"RET_{i}" for i in range(1, 11)]].mean(axis=1)
        df["MA15"] = df[[f"RET_{i}" for i in range(1, 16)]].mean(axis=1)

    # Convert target to binary
    sign_of_return: LabelEncoder = LabelEncoder()
    train_df["RET"] = sign_of_return.fit_transform(train_df["RET"])

    return train_df, test_df
