"""Helper tools for the cow behaviour practical (notebooks_code/01_cow_behaviour.ipynb).

The check_* functions run technical checks of your result. Every check is listed
with a short explanation of why it matters, so you can see what was tested. The
checks do not give the solution and do not replace your own evaluation.

"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.metrics import accuracy_score

PASS_SYMBOL = "&#10004;"  # check mark
FAIL_SYMBOL = "&#10008;"  # cross


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _show_checks(task_name: str, checks: list[tuple[bool, str]], extra_html: str = "") -> None:
    """Show every check with its result. `checks` is a list of (passed, explanation)."""
    all_passed = all(passed for passed, _ in checks)
    colour, border, text = ("#e8f5e9", "#2e7d32", "#103d14") if all_passed else ("#fff3e0", "#ef6c00", "#4e2a00")
    status = "all checks passed" if all_passed else "some checks failed"
    items = "".join(
        f'<li style="list-style:none;margin-left:-18px">'
        f'<span style="color:{"#2e7d32" if passed else "#c62828"};font-weight:bold">'
        f'{PASS_SYMBOL if passed else FAIL_SYMBOL}</span> {explanation}</li>'
        for passed, explanation in checks
    )
    html = (
        f'<div style="background:{colour};border-left:6px solid {border};padding:8px 14px;'
        f'border-radius:6px;color:{text}"><b>{task_name}: {status}</b><ul>{items}</ul>{extra_html}</div>'
    )
    display(HTML(html))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_train_test_split(train, test, df_data, train_cows, test_cow) -> None:
    """Check the split by cow."""
    checks = []
    both_frames = isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    checks.append((both_frames, "`train` and `test` are pandas DataFrames."))
    if not both_frames:
        _show_checks("Train-test split by cow", checks)
        return

    has_columns = set(df_data.columns).issubset(train.columns) and set(df_data.columns).issubset(test.columns)
    checks.append((has_columns, "Both keep all columns of `df_data` (you need `time` and `cow_id` later)."))

    train_found = sorted(map(str, train["cow_id"].unique()))
    test_found = sorted(map(str, test["cow_id"].unique()))
    checks.append((train_found == sorted(train_cows),
                   f"`train` contains exactly the training cows {sorted(train_cows)} (yours: {train_found})."))
    checks.append((test_found == [test_cow], f"`test` contains only the test cow {test_cow} (yours: {test_found})."))

    overlap = sorted(set(train_found) & set(test_found))
    checks.append((not overlap,
                   "No cow is both in `train` and in `test`. Otherwise the model is tested on an animal it already knows "
                   f"(data leakage). Overlap: {overlap if overlap else 'none'}."))

    n_expected_train = int(df_data["cow_id"].isin(train_cows).sum())
    n_expected_test = int((df_data["cow_id"] == test_cow).sum())
    checks.append((len(train) == n_expected_train and len(test) == n_expected_test,
                   f"All rows of the cows are used: {n_expected_train:,} training rows and {n_expected_test:,} test rows "
                   f"(yours: {len(train):,} and {len(test):,}). No random sampling of rows."))
    _show_checks("Train-test split by cow", checks)


def check_predictions(y_true, y_pred, model_name: str = "model", class_mapping: dict | None = None) -> None:
    """Check predictions and compare the predicted class shares with the true class shares."""
    checks = []
    y_true = pd.Series(np.asarray(y_true))
    y_pred_array = np.asarray(y_pred)

    checks.append((len(y_pred_array) == len(y_true),
                   f"One prediction per window: {len(y_true):,} expected, yours: {len(y_pred_array):,}."))
    checks.append((y_pred_array.ndim == 1,
                   "Predictions are one class label per window (use `.predict()`, not `.predict_proba()`)."))
    if not all(passed for passed, _ in checks):
        _show_checks(f"Predictions {model_name}", checks)
        return

    unknown = sorted(set(np.unique(y_pred_array)) - set(y_true.unique()))
    checks.append((not unknown, f"Only known classes are predicted (unknown: {unknown if unknown else 'none'})."))

    accuracy = accuracy_score(y_true, y_pred_array)
    majority_share = y_true.value_counts(normalize=True).iloc[0]
    checks.append((accuracy >= majority_share,
                   f"Accuracy ({accuracy:.3f}) is at least the share of the largest class ({majority_share:.3f}). "
                   "A model below this level is worse than always predicting the largest class."))

    # Table: how often is every class predicted, compared with how often it really occurs?
    shares = pd.DataFrame({
        "true share": y_true.value_counts(normalize=True),
        "predicted share": pd.Series(y_pred_array).value_counts(normalize=True),
    }).fillna(0).sort_index()
    if class_mapping is not None:
        shares.index = [f"{c} = {class_mapping.get(c, c)}" for c in shares.index]
    never_predicted = shares.index[shares["predicted share"] == 0].tolist()
    checks.append((True,
                   "Class shares below: a class that is never or seldom predicted cannot be found by the model "
                   f"(never predicted: {never_predicted if never_predicted else 'none'})."))
    table = (shares * 100).round(1).to_html(classes="dataframe", border=0)
    _show_checks(f"Predictions {model_name}", checks, f"<p><b>Share of windows (%)</b></p>{table}")


def check_pipeline(model, X_train, name: str = "Pipeline", model_step: str = "rf",
                   model_type: str = "RandomForestClassifier") -> None:
    """Check a fitted pipeline: StandardScaler + classifier, fitted on the training data only."""
    checks = []
    steps = getattr(model, "named_steps", None)
    checks.append((steps is not None, "The model is a scikit-learn `Pipeline`."))
    if steps is None:
        _show_checks(name, checks)
        return

    step_names = list(steps)
    checks.append(("scaler" in steps and type(steps["scaler"]).__name__ == "StandardScaler",
                   f"Step `scaler` is a `StandardScaler` (your steps: {step_names})."))
    checks.append((model_step in steps and type(steps[model_step]).__name__ == model_type and step_names[-1] == model_step,
                   f"The last step `{model_step}` is a `{model_type}`."))

    scaler = steps.get("scaler")
    scaler_fitted = scaler is not None and hasattr(scaler, "mean_")
    checks.append((scaler_fitted and hasattr(steps.get(model_step), "n_features_in_"),
                   "The pipeline is fitted (`.fit(X_train, y_train)` was called)."))
    if scaler_fitted:
        checks.append((scaler.n_features_in_ == X_train.shape[1],
                       f"The model uses the {X_train.shape[1]} features of `X_train` (yours: {scaler.n_features_in_})."))
        if scaler.n_features_in_ == X_train.shape[1]:
            same_means = np.allclose(scaler.mean_, X_train.mean().to_numpy(), rtol=1e-6)
            checks.append((same_means,
                           "The scaler learned its means from `X_train` only. If the test windows were included, "
                           "information of the test cow would leak into the model."))
    _show_checks(name, checks)


def check_extra_features(X_train, X_test, X_train_ext, X_test_ext,
                         names=("X_train", "X_test", "X_train_ext", "X_test_ext")) -> None:
    """Check the extended feature tables and show how strongly the new features correlate with the old ones.

    `names` are the variable names shown in the output (original train, original test, extended train, extended test).
    """
    checks = []
    for original_name, extended_name, original, extended in (
        (names[0], names[2], X_train, X_train_ext),
        (names[1], names[3], X_test, X_test_ext),
    ):
        if not isinstance(extended, pd.DataFrame):
            checks.append((False, f"`{extended_name}` should be a pandas DataFrame."))
            continue
        checks.append((len(extended) == len(original) and extended.index.equals(original.index),
                       f"`{extended_name}` has the same windows, in the same order, as `{original_name}` "
                       "(otherwise features and labels no longer belong together)."))
        checks.append((set(original.columns).issubset(extended.columns),
                       f"`{extended_name}` keeps all original features."))
    if not all(passed for passed, _ in checks):
        _show_checks("Extra features", checks)
        return

    new_columns = [c for c in X_train_ext.columns if c not in X_train.columns]
    checks.append((bool(new_columns), f"There is at least one new feature: {new_columns}."))
    checks.append((list(X_train_ext.columns) == list(X_test_ext.columns),
                   "Training and test tables have the same columns in the same order."))
    if new_columns:
        values = X_train_ext[new_columns].to_numpy(dtype=float)
        checks.append((np.isfinite(values).all(), "The new features have no missing or infinite values."))
        checks.append((bool((X_train_ext[new_columns].std() > 0).all()), "No new feature is constant."))

    extra_html = ""
    if new_columns and all(passed for passed, _ in checks):
        # For every new feature: the existing feature with the strongest correlation
        correlations = X_train_ext.corr()
        rows = []
        for column in new_columns:
            others = correlations.loc[column, X_train.columns].abs()
            rows.append({"new feature": column, "most similar existing feature": others.idxmax(),
                         "|r|": round(float(others.max()), 2)})
        extra_html = (
            "<p><b>Correlation with the existing features</b> (|r| close to 1: the feature adds little new information)</p>"
            + pd.DataFrame(rows).to_html(index=False, border=0)
        )
    _show_checks("Extra features", checks, extra_html)


def check_grid_search(search, param_grid: dict | None = None) -> None:
    """Check a fitted GridSearchCV: grouped cross-validation, refit, and the best value at the edge of the grid."""
    checks = []
    checks.append((hasattr(search, "cv_results_"), "The grid search is fitted (`.fit(X, y, groups=...)`)."))
    if not hasattr(search, "cv_results_"):
        _show_checks("Grid search", checks)
        return

    cv_name = type(search.cv).__name__
    checks.append((cv_name in ("LeaveOneGroupOut", "GroupKFold"),
                   f"Cross-validation splits by cow (`cv={cv_name}`). A random split of windows gives optimistic "
                   "scores, because neighbouring windows of the same cow are very similar."))
    checks.append((search.scoring == "f1_macro",
                   f"Scoring is macro-F1 (yours: {search.scoring}), so that every behaviour counts equally."))
    checks.append((hasattr(search, "best_estimator_"),
                   "`refit=True`: the best setting is trained again on all training windows (`best_estimator_`)."))
    n_candidates = len(search.cv_results_["params"])
    checks.append((n_candidates > 1, f"The grid contains {n_candidates} combinations of settings."))

    grid = param_grid if param_grid is not None else getattr(search, "param_grid", {})
    if isinstance(grid, dict):
        for parameter, values in grid.items():
            values = list(values)
            numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            best_value = search.best_params_.get(parameter)
            if len(numeric) >= 3 and best_value in numeric and best_value in (min(numeric), max(numeric)):
                checks.append((False,
                               f"The best `{parameter}` ({best_value}) is at the edge of your grid {values}. "
                               "Extend the grid in that direction: the optimum may lie outside it."))
    _show_checks("Grid search", checks,
                 f"<p>Best setting: <code>{search.best_params_}</code>, mean macro-F1 over the validation cows: "
                 f"{search.best_score_:.3f}</p>")


def check_saved_model(model_path, example_raw_data: pd.DataFrame) -> None:
    """Load the saved model in a NEW Python process and let it predict example raw data.

    This proves that the file contains everything that is needed (feature extraction, scaler,
    model, threshold) and does not depend on variables of this notebook.
    """
    checks = []
    model_path = Path(model_path)
    checks.append((model_path.exists(), f"The model file exists: `{model_path.name}`."))
    if not model_path.exists():
        _show_checks("Saved model", checks)
        return

    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "example_raw_data.pkl"
        example_raw_data.to_pickle(data_path)
        script = (
            "import pickle, pandas as pd\n"
            f"model = pickle.load(open(r'{model_path}', 'rb'))\n"
            f"raw = pd.read_pickle(r'{data_path}')\n"
            "result = model.predict(raw)\n"
            "print(type(result).__name__, len(result), sorted(map(str, pd.Series(result['behaviour']).unique())))\n"
        )
        run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    loaded = run.returncode == 0
    checks.append((loaded,
                   "A new Python process (without this notebook) can load the file and call `.predict(raw_data)`."
                   + ("" if loaded else f"<pre>{run.stderr[-1500:]}</pre>")))
    if loaded:
        output = run.stdout.strip()
        checks.append((output.startswith("DataFrame"),
                       f"`.predict()` returns a DataFrame with one row per window and a column `behaviour`: {output}."))
    _show_checks("Saved model", checks)
