import numpy as np
import pandas as pd
import argparse

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
)

# from imblearn.metrics import geometric_mean_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import recall_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight


from engine.datasets import get_dataset


LIST_ENCODE = [
    "subproj",
    "sex",
    "llkk_txt",
    "llkk_county_letter",
    "vdc",
    "bmi",
    "predict_cohort",
]


def geometric_mean_score(y_true, y_pred, labels=None, average="weighted"):
    """
    Computes the geometric mean score (G-Mean) for binary or multiclass classification.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True labels.
    - y_pred: array-like of shape (n_samples,) - Predicted labels.
    - labels: array-like of shape (n_classes,) - The unique labels for the classification (optional).
    - average: str - {'binary', 'weighted'} - If 'binary', returns the G-Mean for binary classification.
                If 'weighted', returns the weighted average G-Mean across classes.

    Returns:
    - g_mean: float - Geometric mean score.
    """
    if labels is None:
        labels = np.unique(y_true)

    g_means = []
    weights = []

    for label in labels:
        # Confusion matrix components for each class
        TP = np.sum((y_true == label) & (y_pred == label))
        TN = np.sum((y_true != label) & (y_pred != label))
        FP = np.sum((y_true != label) & (y_pred == label))
        FN = np.sum((y_true == label) & (y_pred != label))

        # Sensitivity or Recall for the current class
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Specificity or Recall for the other classes
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Geometric Mean for the current class
        g_mean = np.sqrt(sensitivity * specificity)
        g_means.append(g_mean)

        # Weight by the number of true instances in this class
        weights.append(np.sum(y_true == label))

    if average == "weighted":
        # Weighted average of the geometric means
        g_mean = np.average(g_means, weights=weights)
    else:  # Default to binary (first class)
        g_mean = g_means[0]

    return g_mean


def evaluate(
    X, y, X_test, y_test, model, output="classification", sample_weights_xgb_multi=None
):
    def clean(
        X, y, X_test, y_test
    ):  # exclude all rows containing non-mutual class in y_train and y_test
        list_notmutual = np.setxor1d(y, y_test).tolist()
        list_notmutual_index_train = []
        list_notmutual_index_test = []
        if list_notmutual:
            for item in list_notmutual:
                itemindex = np.where(y == item)[0].tolist()
                list_notmutual_index_train.extend(itemindex)
                itemindex = np.where(y_test == item)[0].tolist()
                list_notmutual_index_test.extend(itemindex)

        if list_notmutual_index_train:
            X = np.delete(X, list_notmutual_index_train, axis=0)
            y = np.delete(y, list_notmutual_index_train, axis=0)

        if list_notmutual_index_test:
            X_test = np.delete(X_test, list_notmutual_index_test, axis=0)
            y_test = np.delete(y_test, list_notmutual_index_test, axis=0)

        # fix xgboost's bug: difference in number of classes in y_train and y_test
        le = LabelEncoder()
        y = le.fit_transform(y)
        y_test = le.fit_transform(y_test)

        return X, y, X_test, y_test

    X = X.to_numpy()
    y = y.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    if output == "classification":
        X, y, X_test, y_test = clean(X, y, X_test, y_test)

        try:
            if sample_weights_xgb_multi is None:
                model.fit(X, y)
            else:
                model.fit(X, y, sample_weight=sample_weights_xgb_multi)
        except:  # OOM
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        # acc = model.score(X_test, y_test)

        # fixed RuntimeError: exception occurred! file=/opt/conda/conda-bld/work/cpp/src/randomforest/randomforest.cu line=263: More than one variable expected for classification problems. Obtained 64 stack frames

        try:
            y_test_predict = model.predict(X_test)
            acc = balanced_accuracy_score(y_test_predict, y_test)
            precision = precision_score(y_test_predict, y_test, average="weighted")
            recall = recall_score(y_test_predict, y_test, average="weighted")
            f1 = f1_score(y_test_predict, y_test, average="weighted")
            gmean = geometric_mean_score(y_test_predict, y_test, average="weighted")
        except:
            acc, precision, recall, f1, gmean = np.nan, np.nan, np.nan, np.nan, np.nan

        try:  # multiclass
            roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
        except:  # binary
            try:
                roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except:
                roc = 0  # The model was not trained to output probabilities (LinearSVMParams.probability == false)

        # try:
        #     # Get feature importance scores
        #     importance_scores = model.feature_importances_

        #     # Print the scores along with feature indices (or names if you have them)

        #     print()
        #     print()
        #     print()
        #     print()
        #     print("Feature Importances:")
        #     for i, score in enumerate(importance_scores):
        #         print(f"Feature {i}: {score}")
        #     print()
        #     print()
        #     print()
        #     print()
        # except:
        #     pass

        return (acc, precision, recall, f1, gmean, roc)

    else:  # regression
        try:
            model.fit(X, y)
        except:  # OOM
            return (np.nan, np.nan, np.nan)

        mae = mean_absolute_error(model.predict(X_test), y_test)
        mse = mean_squared_error(model.predict(X_test), y_test)
        r2 = r2_score(model.predict(X_test), y_test)

        return (mae, mse, r2)


# ===========================================================================
# ML methods
# ---------------------------------------------------------------------------
def perform_linear_regression(
    X,
    y,
    X_test,
    y_test,
    output="classification",
    params={"max_iter": 100, "solver": "saga"},
    device="gpu",
):
    if device == "gpu":
        try:
            use_device = "gpu"
            from cuml import LogisticRegression as LogisticRegression
            from cuml import ElasticNet as ElasticNet
        except:
            use_device = "cpu"
            from sklearn.linear_model import LogisticRegression
            from sklearn.linear_model import ElasticNet
    else:
        use_device = "cpu"
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import ElasticNet

    if device == "gpu" and use_device == "cpu":
        raise Warning("Can't import cuml. Use sklearn instead")

    if use_device == "gpu":
        params["solver"] = "qn"

    print("use_device:", use_device)

    if output == "classification":
        # dealing with imbalanced datasets
        params["class_weight"] = "balanced"
        model = LogisticRegression(**params)  # Linear Kernel
    else:
        model = ElasticNet(**params)

    for col in list(X.columns):
        X[col] = pd.to_numeric(X[col])
        X_test[col] = pd.to_numeric(X_test[col])

    scores = evaluate(X, y, X_test, y_test, model, output)
    return scores


def perform_svm(
    X,
    y,
    X_test,
    y_test,
    output="classification",
    params={
        # "C": 9.194358446658468,
        # "lbfgs_memory": 9,
        # "loss": "hinge",
        # "penalty": "l1",
    },
    device="gpu",
):
    if device == "gpu":
        try:
            use_device = "gpu"
            from cuml import LinearSVC as SVM
            from cuml import LinearSVR as SVR
        except:
            use_device = "cpu"
            from sklearn.svm import SVC as SVM
            from sklearn.svm import SVR as SVR
    else:
        use_device = "cpu"
        from sklearn.svm import SVC as SVM
        from sklearn.svm import SVR as SVR

    if device == "gpu" and use_device == "cpu":
        raise Warning("Can't import cuml. Use sklearn instead")

    print("use_device:", use_device)

    if output == "classification":
        # dealing with imbalanced datasets
        params["class_weight"] = "balanced"
        model = SVM(**params)  # Linear Kernel
    else:
        model = SVR(**params)

    for col in list(X.columns):
        X[col] = pd.to_numeric(X[col])
        X_test[col] = pd.to_numeric(X_test[col])

    scores = evaluate(X, y, X_test, y_test, model, output)
    return scores


# We use logistic regression as base estimator instead of svm (bug)
def perform_bagging(
    X,
    y,
    X_test,
    y_test,
    output="classification",
    params_bagging={"max_iter": 100, "solver": "saga"},
    params={},
    n_estimators=3,
    device="gpu",
):
    if device == "gpu":
        try:
            use_device = "gpu"
            from cuml import LogisticRegression as LogisticRegression
            from cuml import ElasticNet as ElasticNet
        except:
            use_device = "cpu"
            from sklearn.linear_model import LogisticRegression
            from sklearn.linear_model import ElasticNet
    else:
        use_device = "cpu"
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import ElasticNet

    if device == "gpu" and use_device == "cpu":
        raise Warning("Can't import cuml. Use sklearn instead")

    if use_device == "gpu":
        params_bagging["solver"] = "qn"

    print("use_device:", use_device)

    if output == "classification":
        # dealing with imbalanced datasets
        # params["class_weight"] = "balanced"
        model = BaggingClassifier(
            LogisticRegression(**params),
            max_samples=1.0 / n_estimators,
            n_estimators=n_estimators,
            n_jobs=-1,
        )
    else:
        model = BaggingRegressor(
            ElasticNet(**params),
            max_samples=1.0 / n_estimators,
            n_estimators=n_estimators,
            n_jobs=-1,
        )

    for col in list(X.columns):
        X[col] = pd.to_numeric(X[col])
        X_test[col] = pd.to_numeric(X_test[col])

    try:
        scores = evaluate(X, y, X_test, y_test, model, output)
        return scores
    except:  # OOM for News dataset
        return 0, 0, 0


def perform_randomforest(
    X,
    y,
    X_test,
    y_test,
    output="classification",
    params={"n_estimators": 100, "n_jobs": -1},
    device="gpu",
):
    if device == "gpu":
        try:
            from cuml import RandomForestClassifier as RandomForestClassifier
            from cuml import RandomForestRegressor as RandomForestRegressor

            use_device = "gpu"
        except:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.ensemble import RandomForestRegressor

            use_device = "cpu"
    else:
        use_device = "cpu"
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor

    if device == "gpu" and use_device == "cpu":
        raise Warning("Can't import cuml. Use sklearn instead")

    if use_device == "gpu":
        params.pop("n_jobs", None)

    print("use_device:", use_device)

    if output == "classification":
        model = RandomForestClassifier(**params)
    else:
        model = RandomForestRegressor(**params)

    for col in list(X.columns):
        X[col] = pd.to_numeric(X[col])
        X_test[col] = pd.to_numeric(X_test[col])

    scores = evaluate(X, y, X_test, y_test, model, output)
    return scores


def perform_xgboost(
    X,
    y,
    X_test,
    y_test,
    output="classification",
    params={},
    device="gpu",
):
    if device == "gpu":
        params["gpu_id"] = 0
        params["tree_method"] = "gpu_hist"

    print("use_device:", device)

    sample_weights_xgb_multi = None
    if output == "classification":
        sample_weights_xgb_multi = compute_sample_weight("balanced", y)
        model = XGBClassifier(**params)
    else:
        model = XGBRegressor(**params)

    for col in list(X.columns):
        X[col] = pd.to_numeric(X[col])
        X_test[col] = pd.to_numeric(X_test[col])

    scores = evaluate(X, y, X_test, y_test, model, output)
    return scores


# ---------------------------------------------------------------------------
# ML methods - end
# ===========================================================================


# ===========================================================================
# main functions - start
# ---------------------------------------------------------------------------
def main():
    D = get_dataset(args.dataset, args.arch)
    discrete_columns = D.discrete_columns
    continuous_columns = [c for c in list(D.data_train) if c not in discrete_columns]
    print(discrete_columns)
    print(continuous_columns)

    # mode = "optimal"
    # # mode = "last"

    # print(
    #     perform_xgboost(
    #         X=D.data_train[D.features],
    #         y=D.data_train[D.target],
    #         X_test=D.data_test[D.features],
    #         y_test=D.data_test[D.target],
    #         output=D.output,
    #     )
    # )

    print(
        perform_randomforest(
            X=D.data_train[D.features],
            y=D.data_train[D.target],
            X_test=D.data_test[D.features],
            y_test=D.data_test[D.target],
            output=D.output,
            device="cpu",
        )
    )

    # print(
    #     perform_bagging(
    #         X=D.data_train[D.features],
    #         y=D.data_train[D.target],
    #         X_test=D.data_test[D.features],
    #         y_test=D.data_test[D.target],
    #         output=D.output,
    #     )
    # )

    # print(
    #     perform_svm(
    #         X=D.data_train[D.features],
    #         y=D.data_train[D.target],
    #         X_test=D.data_test[D.features],
    #         y_test=D.data_test[D.target],
    #         output=D.output,
    #     )
    # )

    # print(
    #     perform_linear_regression(
    #         X=D.data_train[D.features],
    #         y=D.data_train[D.target],
    #         X_test=D.data_test[D.features],
    #         y_test=D.data_test[D.target],
    #         output=D.output,
    #     )
    # )

    a = 2


# ---------------------------------------------------------------------------
# main functions - end
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--dir_logs",
        type=str,
        default="database/gan/",
        help="dir logs",
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="ctgan",
        choices=["ctgan", "tvae", "copulagan", "dpcgans", "ctab"],
    )
    parser.add_argument(
        "--dataset",
        default="diabetesbalanced",
        type=str,
    )
    parser.add_argument(
        "-g",
        "--group_dataset",
        default=4,
        type=int,
        choices=[0, 1, 2, 3, 4],
    )

    # Parse the arguments
    args = parser.parse_args()
    main()
