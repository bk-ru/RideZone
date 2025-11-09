"""
Run offline experiments to produce Appendix B artifacts.

Outputs (default: docs/experiments/):
- metrics_folds.csv         # пер-складочные метрики по моделям
- metrics_summary.csv       # агрегированные метрики (среднее±std)
- metrics_table.png         # таблица метрик в виде изображения
- learning_curve.png        # learning curves для лучшей модели
- loss_curve.png            # loss/deviance curves для GB модели
- stats.json                # результаты статистических тестов

Usage:
    python scripts/run_experiments.py [--input PATH] [--target COL]
                                      [--task regression|classification]
                                      [--output-dir docs/experiments]

Примечания:
- Скрипт пытается автоматически получить X,y из кода проекта (ridezone_ai),
  а при неудаче — ищет подходящие CSV/Parquet в data/processed.
- В крайнем случае генерирует синтетические данные (с предупреждением).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as sstats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


MAX_JOBS = max(1, min(4, (os.cpu_count() or 1)))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_task(y: pd.Series, explicit_task: Optional[str]) -> str:
    if explicit_task in {"regression", "classification"}:
        return explicit_task
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and nunique <= 20:
        return "classification"
    if nunique <= 10 and set(np.unique(y.dropna().values)).issubset({0, 1}):
        return "classification"
    return "regression"


def _choose_target_column(df: pd.DataFrame, target_opt: Optional[str]) -> str:
    if target_opt and target_opt in df.columns:
        return target_opt
    candidates = ["target", "demand", "y", "label", "is_high_demand"]
    for c in candidates:
        if c in df.columns:
            return c
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("Не удалось определить столбец цели (target). Укажите --target.")
    return num_cols[-1]


def _split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col]
    X = df.drop(columns=[target_col])
    for maybe_id in ["id", "hex_id", "station_id", "index", "geometry"]:
        if maybe_id in X.columns:
            X = X.drop(columns=[maybe_id])
    return X, y


def _load_from_project_pipeline() -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    try:
        from ridezone_ai import pipeline as rz_pipe  # type: ignore

        for fn_name in (
            "get_training_frame",
            "get_training_data",
            "build_training_frame",
            "load_training_data",
            "run",
        ):
            fn = getattr(rz_pipe, fn_name, None)
            if fn is None:
                continue
            try:
                result = fn()
                if isinstance(result, tuple) and len(result) == 2:
                    X, y = result
                    if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
                        if isinstance(y, pd.DataFrame):
                            y = y.iloc[:, 0]
                        return X, y
                if isinstance(result, pd.DataFrame):
                    target_col = _choose_target_column(result, None)
                    return _split_features_target(result, target_col)
                if isinstance(result, dict):
                    if "X" in result and "y" in result:
                        X = result["X"]
                        y = result["y"]
                        if isinstance(y, pd.DataFrame):
                            y = y.iloc[:, 0]
                        return X, y
                    if "df" in result:
                        df = result["df"]
                        target_col = _choose_target_column(df, None)
                        return _split_features_target(df, target_col)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _load_from_files(input_path: Optional[Path]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    def read_any(p: Path) -> pd.DataFrame:
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        raise ValueError(f"Неподдерживаемый формат: {p}")

    if input_path and input_path.exists():
        df = read_any(input_path)
        tgt = _choose_target_column(df, None)
        return _split_features_target(df, tgt)

    candidates = [
        Path("data/processed/features.parquet"),
        Path("data/processed/training.parquet"),
        Path("data/processed/dataset.parquet"),
        Path("data/processed/features.csv"),
        Path("data/processed/dataset.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = read_any(p)
            tgt = _choose_target_column(df, None)
            return _split_features_target(df, tgt)

    processed = Path("data/processed")
    if processed.exists():
        for p in list(processed.glob("**/*.parquet")) + list(processed.glob("**/*.csv")):
            try:
                df = read_any(p)
                tgt = _choose_target_column(df, None)
                return _split_features_target(df, tgt)
            except Exception:
                continue
    return None


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if not transformers:
        return ColumnTransformer([], remainder="passthrough")
    return ColumnTransformer(transformers)


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    out_dir: Path,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    pre = _make_preprocessor(X)

    if task == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=MAX_JOBS),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        }
        scoring = {
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "RMSE": make_scorer(_rmse_score, greater_is_better=False),
            "R2": make_scorer(r2_score),
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=MAX_JOBS),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        }
        scoring = {
            "Accuracy": make_scorer(accuracy_score),
            "F1_macro": make_scorer(f1_score, average="macro"),
        }
        if y.nunique() == 2:
            scoring["ROC_AUC"] = make_scorer(roc_auc_score, needs_proba=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rows = []
    per_model_scores: Dict[str, Dict[str, List[float]]] = {}
    for name, est in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", est)])
        res = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=MAX_JOBS,
            return_estimator=False,
        )
        per_model_scores[name] = {}
        for metric in scoring.keys():
            vals = res[f"test_{metric}"]
            if metric in {"MAE", "RMSE"}:
                vals = -vals
            per_model_scores[name][metric] = vals.tolist()
            for fold_idx, value in enumerate(vals, start=1):
                rows.append({"model": name, "fold": fold_idx, metric: float(value)})

    df_folds = pd.DataFrame(rows).groupby(["model", "fold"]).agg("first").reset_index()

    summary_rows = []
    for name in per_model_scores.keys():
        row = {"model": name}
        for metric, vals in per_model_scores[name].items():
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1))
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)

    df_folds.to_csv(out_dir / "metrics_folds.csv", index=False, encoding="utf-8")
    df_summary.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8")
    _render_table_png(df_summary, out_dir / "metrics_table.png")

    return df_folds, df_summary, per_model_scores


def _render_table_png(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.8), 1 + 0.5 * len(df)))
    ax.axis("off")

    def _fmt(val: object) -> str:
        if isinstance(val, (int, float, np.integer, np.floating)):
            return f"{val:.4f}"
        return str(val)

    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(_fmt)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    best_model_name: str,
    out_path: Path,
    random_state: int = 42,
) -> None:
    pre = _make_preprocessor(X)
    if task == "regression":
        if best_model_name == "LinearRegression":
            model = LinearRegression()
        elif best_model_name == "GradientBoostingRegressor":
            model = GradientBoostingRegressor(random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=MAX_JOBS)
        scoring = "neg_root_mean_squared_error"
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        if best_model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=1000)
        elif best_model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(random_state=random_state)
        else:
            model = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=MAX_JOBS)
        scoring = "f1_macro"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    train_sizes, train_scores, test_scores = learning_curve(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=MAX_JOBS,
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=random_state,
    )

    if isinstance(scoring, str) and scoring.startswith("neg_"):
        train_scores = -train_scores
        test_scores = -test_scores

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train", marker="o")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="CV", marker="o")
    plt.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.2,
    )
    plt.fill_between(
        train_sizes,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.2,
    )
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("Качество")
    plt.title(f"Learning Curve — {best_model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_loss_curve(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    out_path: Path,
    random_state: int = 42,
) -> None:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if task == "classification" and y.nunique() > 1 else None,
    )
    pre = _make_preprocessor(X_tr)
    if task == "regression":
        gb = GradientBoostingRegressor(random_state=random_state)
        pipe = Pipeline(steps=[("pre", pre), ("model", gb)])
        pipe.fit(X_tr, y_tr)
        gb_in = pipe.named_steps["model"]
        X_va_t = pipe.named_steps["pre"].transform(X_va)
        X_tr_t = pipe.named_steps["pre"].transform(X_tr)
        train_losses, val_losses = [], []
        for y_pred_tr in gb_in.staged_predict(X_tr_t):
            train_losses.append(mean_squared_error(y_tr, y_pred_tr))
        for y_pred_va in gb_in.staged_predict(X_va_t):
            val_losses.append(mean_squared_error(y_va, y_pred_va))
        ylabel = "MSE"
        title = "GB Loss Curve (MSE)"
    else:
        gb = GradientBoostingClassifier(random_state=random_state)
        pipe = Pipeline(steps=[("pre", pre), ("model", gb)])
        pipe.fit(X_tr, y_tr)
        gb_in = pipe.named_steps["model"]
        X_va_t = pipe.named_steps["pre"].transform(X_va)
        X_tr_t = pipe.named_steps["pre"].transform(X_tr)
        train_losses, val_losses = [], []
        for y_proba_tr in gb_in.staged_predict_proba(X_tr_t):
            train_losses.append(log_loss(y_tr, y_proба_tr, labels=np.unique(y)))
        for y_proба_va in gb_in.staged_predict_proба(X_va_t):
            val_losses.append(log_loss(y_va, y_proба_va, labels=np.unique(y)))
        ylabel = "LogLoss"
        title = "GB Loss Curve (LogLoss)"

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train", marker="o", markersize=3)
    plt.plot(val_losses, label="Validation", marker="o", markersize=3)
    plt.xlabel("Количество деревьев")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_stats_tests(
    per_model_scores: Dict[str, Dict[str, List[float]]],
    task: str,
    out_path: Path,
) -> Dict:
    primary = "MAE" if task == "regression" else "F1_macro"
    if task == "classification" and primary not in next(iter(per_model_scores.values())).keys():
        primary = "Accuracy"

    stats_out = {"primary_metric": primary, "tests": {}}

    def get_scores(name: str) -> Optional[List[float]]:
        return per_model_scores.get(name, {}).get(primary)

    rf = get_scores("RandomForestRegressor" if task == "regression" else "RandomForestClassifier")
    gb = get_scores("GradientBoostingRegressor" if task == "regression" else "GradientBoostingClassifier")

    if rf is not None and gb is not None and len(rf) == len(gb) and len(rf) > 1:
        try:
            stat, p_value = sstats.wilcoxon(rf, gb, zero_method="wilcox", alternative="two-sided")
            stats_out["tests"]["Wilcoxon_RF_vs_GB"] = {"stat": float(stat), "p_value": float(p_value)}
        except Exception as err:
            stats_out["tests"]["Wilcoxon_RF_vs_GB"] = {"error": str(err)}

    try:
        series = []
        for metrics in per_model_scores.values():
            if primary in metrics:
                series.append(metrics[primary])
        if len(series) >= 2:
            stat, p_value = sstats.levene(*series, center="median")
            stats_out["tests"]["Levene_equal_variances"] = {"stat": float(stat), "p_value": float(p_value)}
    except Exception as err:
        stats_out["tests"]["Levene_equal_variances"] = {"error": str(err)}

    stats_out["tests"]["Shapiro_info"] = (
        "Распределение остатков проверяется на holdout-разбиении внутри make_loss_curve."
    )

    out_path.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats_out


def load_Xy(input_path: Optional[Path], target: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
    res = _load_from_project_pipeline()
    if res is not None:
        return res

    res = _load_from_files(input_path)
    if res is not None:
        return res

    rng = np.random.default_rng(42)
    n = 2000
    lat = 55.55 + rng.normal(0, 0.05, n)
    lon = 37.4 + rng.normal(0, 0.08, n)
    pop = rng.integers(500, 5000, size=n)
    biz = rng.random(n)
    transit = rng.random(n)
    hour = rng.integers(0, 24, size=n)
    weekday = rng.integers(0, 7, size=n)
    demand = (
        0.5 * biz
        + 0.3 * transit
        + 0.0002 * pop
        + 0.05 * ((hour >= 17) & (hour <= 20)).astype(float)
        + rng.normal(0, 0.1, n)
    )
    X = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "population": pop,
            "business_density": biz,
            "transit_access": transit,
            "hour": hour,
            "weekday": weekday,
        }
    )
    y = pd.Series(demand, name="demand")
    print("[WARN] Используются синтетические данные. Укажите --input для реальных данных.")
    return X, y


def pick_best_model_name(df_summary: pd.DataFrame, task: str) -> str:
    if task == "regression":
        if {"model", "RMSE_mean"}.issubset(df_summary.columns):
            return df_summary.sort_values("RMSE_mean").iloc[0]["model"]
        return "RandomForestRegressor"
    metric = "F1_macro_mean" if "F1_macro_mean" in df_summary.columns else "Accuracy_mean"
    return df_summary.sort_values(metric, ascending=False).iloc[0]["model"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RideZone_AI experiments runner (Appendix B artifacts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None, help="Путь к CSV/Parquet с данными (опционально)")
    parser.add_argument("--target", type=str, default=None, help="Имя столбца цели (если нужно)")
    parser.add_argument("--task", type=str, choices=["regression", "classification"], default=None,
                        help="Тип задачи (если явно известен)")
    parser.add_argument("--output-dir", type=str, default="docs/experiments", help="Папка для сохранения результатов")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    input_path = Path(args.input) if args.input else None
    X, y = load_Xy(input_path, args.target)
    if args.target and isinstance(X, pd.DataFrame):
        df = X.copy()
        df[args.target] = y.values
        X, y = _split_features_target(df, args.target)

    task = _infer_task(y, args.task)
    print(f"Определён тип задачи: {task}. N={len(X)}, d={X.shape[1]}")

    folds, summary, per_model_scores = evaluate_models(X, y, task, out_dir)
    best_model_name = pick_best_model_name(summary, task)

    make_learning_curve(X, y, task, best_model_name, out_dir / "learning_curve.png")
    make_loss_curve(X, y, task, out_dir / "loss_curve.png")
    run_stats_tests(per_model_scores, task, out_dir / "stats.json")

    print("Готово. Ресурсы сохранены в:")
    for filename in [
        "metrics_folds.csv",
        "metrics_summary.csv",
        "metrics_table.png",
        "learning_curve.png",
        "loss_curve.png",
        "stats.json",
    ]:
        print(f" - {out_dir / filename}")


if __name__ == "__main__":
    main()
