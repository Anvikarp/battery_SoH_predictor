"""
Predict discharge capacity from NASA Ames PCOE Li-ion aging data stored in MATLAB .mat files.

The script loads cells B0005, B0006, B0007, and B0018, keeps discharge cycles only, and turns
each cycle into features such as voltage at the start and end of the curve and a trapezoid
integral of voltage over time. A gradient boosting regressor with standardized inputs is trained
on the first three cells and evaluated only on B0018, which mimics deploying on a battery that
never appeared in training. Results are printed to the terminal and two PNGs are written, one
comparing predicted and actual capacity fade on the test cell and one summarizing feature
importance from the ensemble.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# four cells from the nasa pcoe aging arc release used here
BATTERY_IDS = ("B0005", "B0006", "B0007", "B0018")
# train on three cells generalize to the fourth no random split
TRAIN_IDS = ("B0005", "B0006", "B0007")
TEST_IDS = ("B0018",)

FEATURE_COLUMNS = [
    "voltage_start",
    "voltage_end",
    "internal_resistance",
    "mean_temperature",
    "discharge_time",
    "voltage_integral",
    "cap_rolling_mean_5",
    "cap_diff_1",
    "cycle_norm",
]


def _as_1d_f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def _scalar_float(x: Any) -> float:
    return float(np.asarray(x).squeeze())


def find_mat_path(data_root: Path, battery_id: str) -> Path:
    # nested zips unpack to different folders so search recursively
    matches = sorted(data_root.rglob(f"{battery_id}.mat"))
    if not matches:
        raise FileNotFoundError(
            f"No {battery_id}.mat under {data_root}. "
            "Download NASA '5. Battery Data Set' and unzip nested archives."
        )
    return matches[0]


def parse_battery_mat(mat_path: Path, battery_id: str) -> pd.DataFrame:
    # mat structs need squeeze so scalar matlab fields become python scalars
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    batt = mat[battery_id]
    cycles = batt.cycle

    rows: list[dict[str, Any]] = []
    # track latest re from impedance cycles in file order for discharge rows
    last_re: float | None = None
    discharge_index = 0

    for _ci, c in enumerate(cycles):
        ctype = str(c.type).strip().lower()
        if ctype == "impedance":
            d = c.data
            if hasattr(d, "Re"):
                last_re = _scalar_float(d.Re)
            continue

        if ctype != "discharge":
            continue

        d = c.data
        v = _as_1d_f64(d.Voltage_measured)
        t = _as_1d_f64(d.Time)
        temp = _as_1d_f64(d.Temperature_measured)
        cap = _scalar_float(d.Capacity)

        if v.size == 0 or t.size == 0:
            continue

        # trapezoid integral of v over discharge duration energy proxy
        t_rel = t - t[0]
        if hasattr(np, "trapezoid"):
            v_int = float(np.trapezoid(v, t_rel))
        else:
            v_int = float(np.trapz(v, t_rel))

        rows.append(
            {
                "battery_id": battery_id,
                "discharge_cycle_idx": discharge_index,
                "Capacity": cap,
                "voltage_start": float(v[0]),
                "voltage_end": float(v[-1]),
                "internal_resistance": last_re if last_re is not None else np.nan,
                "mean_temperature": float(np.mean(temp)) if temp.size else np.nan,
                "discharge_time": float(t[-1] - t[0]),
                "voltage_integral": v_int,
            }
        )
        discharge_index += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("discharge_cycle_idx").reset_index(drop=True)
    # fill early cycles before first impedance using later re values
    df["internal_resistance"] = df["internal_resistance"].ffill().bfill()
    df["cap_rolling_mean_5"] = (
        df["Capacity"].rolling(window=5, min_periods=1).mean()
    )
    df["cap_diff_1"] = df["Capacity"].diff().fillna(0.0)
    # map discharge index to zero to one within this battery for trend features
    n = len(df) - 1
    if n <= 0:
        df["cycle_norm"] = 0.0
    else:
        df["cycle_norm"] = df["discharge_cycle_idx"] / n

    return df


def load_all_batteries(data_root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for bid in BATTERY_IDS:
        path = find_mat_path(data_root, bid)
        df = parse_battery_mat(path, bid)
        if df.empty:
            raise ValueError(f"No discharge rows parsed for {bid} from {path}")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def mean_abs_pct_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # skip near zero capacity to avoid divide by zero in percent error
    mask = np.abs(y_true) > 1e-12
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def train_and_evaluate(df: Path | str, out_dir: Path | str) -> None:
    data_root = Path(df)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = load_all_batteries(data_root)
    # cross battery split b0018 never seen during fit
    train_df = full[full["battery_id"].isin(TRAIN_IDS)].copy()
    test_df = full[full["battery_id"].isin(TEST_IDS)].copy()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["Capacity"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["Capacity"].values

    # rare nan fallback use train medians only so no test leakage
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        train_med = pd.DataFrame(X_train, columns=FEATURE_COLUMNS).median()
        fill = train_med.values
        X_train = np.where(np.isnan(X_train), fill, X_train)
        X_test = np.where(np.isnan(X_test), fill, X_test)

    # scale then boost user requested hyperparameters
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mape = mean_abs_pct_error(y_test, y_pred)

    print("Held-out test battery: B0018")
    print(f"  RMSE (Ah):              {rmse:.4f}")
    print(f"  MAE (Ah):               {mae:.4f}")
    print(f"  R²:                     {r2:.4f}")
    print(f"  Mean abs. % error:      {mape:.2f}%")
    print(f"  Train discharge rows:   {len(train_df)}")
    print(f"  Test discharge rows:    {len(test_df)}")

    # figure 1 fade curve for the held out test cell
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    cyc = test_df["discharge_cycle_idx"].values
    ax1.plot(cyc, y_test, label="Actual capacity", color="#1f77b4", linewidth=2)
    ax1.plot(cyc, y_pred, label="Predicted", color="#ff7f0e", linewidth=2, alpha=0.9)
    ax1.set_xlabel("Discharge cycle index")
    ax1.set_ylabel("Capacity (Ah)")
    ax1.set_title("B0018 — predicted vs. actual capacity fade")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = out / "b0018_capacity_fade.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Saved {p1}")

    # figure 2 which inputs the ensemble relied on most
    gbr = model.named_steps["gbr"]
    imp = gbr.feature_importances_
    order = np.argsort(imp)[::-1]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    labels_ord = [FEATURE_COLUMNS[i] for i in order]
    ax2.barh(labels_ord, imp[order], color="#4c72b0")
    ax2.invert_yaxis()
    ax2.set_xlabel("Importance")
    ax2.set_title("Gradient boosting — feature importance")
    fig2.tight_layout()
    p2 = out / "feature_importance.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"Saved {p2}")


def main() -> int:
    # stable defaults for headless runs and sandboxed environments
    _root = Path(__file__).resolve().parent
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(_root / ".mpl"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    default_root = _root / "data" / "nasa_raw"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help="Directory tree that contains B0005.mat … B0018.mat (recursive search)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Where to write PNG figures",
    )
    args = parser.parse_args()
    if not args.data_root.exists():
        print(
            "Data root not found. Download:\n"
            "  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip\n"
            "Unzip, then unzip nested BatteryAgingARC-*.zip files so .mat files appear.",
            file=sys.stderr,
        )
        return 1
    train_and_evaluate(args.data_root, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
