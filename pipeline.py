import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEEDS = [11, 22, 33]
EQUIP = ["Dry Van", "Flatbed", "Reefer"]
CV_TEST_MONTHS = [6, 7, 8, 9, 10]      # expanding-window, out-of-time folds
BASE_FEATURES = [
    "log_dist", "distance", "hav", "dist_ratio", "eq", "weight", "weight_neg",
    "weight_na", "weight_cap", "market_index", "dmi", "dow", "pickup_lat",
    "pickup_lon", "delivery_lat", "delivery_lon", "dlat", "dlon", "mi_na",
]
QS_FEATURES = BASE_FEATURES + ["quote_signal", "dqs"]
BLEND_W_QS = 0.5   # weight of the model that uses quote_signal (see README / report)


# ----------------------------------------------------------------------------- io
def find_file(data_dir, stem, required=True):
    """Look for <stem>.csv (or the dashed name) in several likely folders."""
    here = Path(__file__).resolve().parent
    folders = [Path(data_dir), Path(data_dir).resolve(), Path.cwd(), Path.cwd() / "data",
               here, here / "data", here.parent, here.parent / "data"]
    for folder in folders:
        for name in (f"{stem}.csv", f"{stem.replace('_', '-')}.csv"):
            p = folder / name
            if p.exists():
                return p
    if required:
        raise FileNotFoundError(
            f"Could not find {stem}.csv. Put the CSV files in the same folder as pipeline.py "
            f"or in a 'data' folder next to it. Searched: {[str(f) for f in folders]}")
    return None


def load_data(data_dir):
    train = pd.read_csv(find_file(data_dir, "train_test"), parse_dates=["date"])
    val = pd.read_csv(find_file(data_dir, "validation"), parse_dates=["date"])
    template = pd.read_csv(find_file(data_dir, "validation_predictions_template"))
    return train, val, template


# ----------------------------------------------------------------------------- helpers
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def mape(y, p):
    return float(np.mean(np.abs(y - p) / y) * 100)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAPE_%": mape(y, p),
        "MedianAPE_%": float(np.median(np.abs(y - p) / y) * 100),
    }


# ----------------------------------------------------------------------------- cleaning
class Cleaner:
    """Learns imputation values / outlier rules from TRAIN only, applies anywhere."""

    def fit(self, train):
        self.w_med = train.assign(w=train.weight.abs()).groupby("equipment").w.median().to_dict()
        d = train.copy()
        d["rpm"] = d.posted_rate / d.distance
        self.edges = np.quantile(d.distance, np.linspace(0, 1, 21)[1:-1])
        d["db"] = np.searchsorted(self.edges, d.distance)
        self.rpm_med = d.groupby(["db", "equipment"]).rpm.median().to_dict()
        return self

    def rel_rate(self, df):
        """Rate-per-mile relative to the typical rate for that distance bucket / equipment."""
        db = np.searchsorted(self.edges, df.distance)
        med = np.array([self.rpm_med[(b, e)] for b, e in zip(db, df.equipment)])
        return (df.posted_rate / df.distance).values / med

    def label_outlier_mask(self, df):
        """True = label looks corrupt (rate-per-mile ~4.5x too high or ~0.2x too low)."""
        r = self.rel_rate(df)
        return (r >= 1.8) | (r <= 0.55)

    def transform(self, df, dmi_override=None, dqs_override=None):
        df = df.copy()
        df["weight_neg"] = (df.weight < 0).astype(int)        # sign-flipped sentinel values
        df["weight_na"] = df.weight.isna().astype(int)
        df["mi_na"] = df.market_index.isna().astype(int)
        df["weight"] = df.weight.abs()                        # -47500 -> 47500
        df["weight_cap"] = (df.weight >= 47500).astype(int)   # censored at 47,500
        df["weight"] = df.weight.fillna(df.equipment.map(self.w_med))
        # market_index is a daily series (+ small row noise) -> impute with same-day median
        day_med = df.groupby("date").market_index.transform("median")
        df["market_index"] = df.market_index.fillna(day_med)
        df["dmi"] = df.groupby("date").market_index.transform("median")
        df["dqs"] = df.groupby("date").quote_signal.transform("median")
        if dmi_override is not None:
            df["dmi"] = df.date.map(dmi_override)
        if dqs_override is not None:
            df["dqs"] = df.date.map(dqs_override)
        df["log_dist"] = np.log(df.distance)
        df["hav"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
        df["dist_ratio"] = df.distance / df.hav
        df["dow"] = df.date.dt.dayofweek
        df["eq"] = df.equipment.map({e: i for i, e in enumerate(EQUIP)})
        df["dlat"] = df.delivery_lat - df.pickup_lat
        df["dlon"] = df.delivery_lon - df.pickup_lon
        return df


# ----------------------------------------------------------------------------- models
def fit_lgb(X, y, seed):
    m = lgb.LGBMRegressor(
        objective="l1", n_estimators=700, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1, n_jobs=-1,
    )
    return m.fit(X, y)


class RateModel:
    """Two LightGBM families (with / without quote_signal), seed-averaged, blended in log space.
    Target = log(posted_rate), L1 objective (robust to the heavy-tailed label noise)."""

    def fit(self, trn):
        y = np.log(trn.posted_rate.values)
        self.no_qs = [fit_lgb(trn[BASE_FEATURES], y, s) for s in SEEDS]
        self.qs = [fit_lgb(trn[QS_FEATURES], y, s) for s in SEEDS]
        return self

    def predict_parts(self, df):
        a = np.mean([m.predict(df[BASE_FEATURES]) for m in self.no_qs], axis=0)
        b = np.mean([m.predict(df[QS_FEATURES]) for m in self.qs], axis=0)
        return a, b

    def predict(self, df):
        a, b = self.predict_parts(df)
        return np.exp((1 - BLEND_W_QS) * a + BLEND_W_QS * b)


class NaiveBaseline:
    """median rate-per-mile by (distance bucket, equipment) x distance"""

    def __init__(self, cleaner):
        self.c = cleaner

    def predict(self, df):
        db = np.searchsorted(self.c.edges, df.distance)
        med = np.array([self.c.rpm_med[(b, e)] for b, e in zip(db, df.equipment)])
        return med * df.distance.values


# ----------------------------------------------------------------------------- analysis
def data_quality_report(train, val, cleaner, out):
    tr_mask = cleaner.label_outlier_mask(train)
    tr_cities = set(train.pickup) | set(train.delivery)
    va_cities = set(val.pickup) | set(val.delivery)
    new_cities = sorted(va_cities - tr_cities)
    new_rows = int((val.pickup.isin(new_cities) | val.delivery.isin(new_cities)).sum())
    lanes = set(zip(train.pickup, train.delivery))
    rep = {
        "train_rows": len(train), "val_rows": len(val),
        "train_date_range": [str(train.date.min().date()), str(train.date.max().date())],
        "val_date_range": [str(val.date.min().date()), str(val.date.max().date())],
        "duplicate_load_ids_train": int(train.load_id.duplicated().sum()),
        "missing_weight_train": int(train.weight.isna().sum()), "missing_weight_val": int(val.weight.isna().sum()),
        "missing_market_index_train": int(train.market_index.isna().sum()),
        "missing_market_index_val": int(val.market_index.isna().sum()),
        "negative_weight_train": int((train.weight < 0).sum()), "negative_weight_val": int((val.weight < 0).sum()),
        "weight_at_47500_cap_train": int((train.weight.abs() == 47500).sum()),
        "weight_at_47500_cap_val": int((val.weight.abs() == 47500).sum()),
        "label_outliers_train(rate/mile off by ~4.5x or ~0.2x)": int(tr_mask.sum()),
        "label_outliers_train_pct": float(tr_mask.mean() * 100),
        "cities_train": len(tr_cities), "cities_val": len(va_cities),
        "cities_only_in_val": new_cities, "val_rows_touching_new_cities": new_rows,
        "val_rows_with_unseen_lane_pct": float(np.mean([(p, d) not in lanes for p, d in zip(val.pickup, val.delivery)]) * 100),
        "market_index_mean_train": float(train.market_index.mean()), "market_index_mean_val": float(val.market_index.mean()),
        "quote_signal_std_train": float(train.quote_signal.std()), "quote_signal_std_val": float(val.quote_signal.std()),
    }
    Path(out / "data_quality.json").write_text(json.dumps(rep, indent=2))
    return rep


def quote_signal_by_month(train, cleaner, out):
    d = train.copy()
    d["rel"] = cleaner.rel_rate(d)
    d = d[~cleaner.label_outlier_mask(d)]
    d["lr"] = np.log(d.rel)
    d["month"] = d.date.dt.month
    t = d.groupby("month").apply(lambda g: pd.Series({
        "mean_quote_signal": g.quote_signal.mean(),
        "corr_quote_signal_vs_rate": g.lr.corr(g.quote_signal),
        "corr_market_index_vs_rate": g.lr.corr(g.market_index),
    }))
    t.to_csv(out / "quote_signal_by_month.csv")
    return t


# ----------------------------------------------------------------------------- validation
def time_cv(train, out):
    """Expanding-window, out-of-time CV: train on months < m, test on month m (Jun..Oct).
    Test folds are NOT cleaned of label outliers (honest estimate); training folds are."""
    rows = []
    for m in CV_TEST_MONTHS:
        trn_raw = train[train.date.dt.month < m]
        tst_raw = train[train.date.dt.month == m]
        cl = Cleaner().fit(trn_raw)
        trn = cl.transform(trn_raw)
        trn = trn[~cl.label_outlier_mask(trn_raw).astype(bool)]
        tst = cl.transform(tst_raw)
        model = RateModel().fit(trn)
        a, b = model.predict_parts(tst)
        preds = {
            "naive_median_rate_per_mile": NaiveBaseline(cl).predict(tst),
            "lgbm_no_quote_signal": np.exp(a),
            "lgbm_with_quote_signal": np.exp(b),
            "lgbm_blend(final)": model.predict(tst),
        }
        for name, p in preds.items():
            rows.append({"test_month": m, "model": name, "n": len(tst), **metrics(tst.posted_rate, p)})
        print(f"  fold month {m}: done")
    res = pd.DataFrame(rows)
    res.to_csv(out / "cv_results.csv", index=False)
    summary = res.groupby("model")[["MAE", "RMSE", "MAPE_%", "MedianAPE_%"]].mean().round(3)
    return res, summary


# ----------------------------------------------------------------------------- December chart
def december_chart(chart_path, train, val, cleaner, model, out):
    chart = pd.read_csv(chart_path, parse_dates=["date"])
    pred_col = "predicted_rate"
    coords = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        val[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        val[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates("city").set_index("city")
    df = chart.copy()
    df["pickup_lat"] = df.pickup.map(coords.lat); df["pickup_lon"] = df.pickup.map(coords.lon)
    df["delivery_lat"] = df.delivery.map(coords.lat); df["delivery_lon"] = df.delivery.map(coords.lon)
    # market_index / quote_signal are not part of the chart inputs -> use the daily
    # market conditions observed in validation.csv for the same date (features only, no labels)
    day_mi = val.groupby("date").market_index.median()
    day_qs = val.groupby("date").quote_signal.median()
    df["market_index"] = df.date.map(day_mi)
    df["quote_signal"] = df.date.map(day_qs)
    df["posted_rate"] = np.nan
    feats = cleaner.transform(df, dmi_override=day_mi, dqs_override=day_qs)
    df[pred_col] = model.predict(feats).round(2)
    chart_out = chart.copy()
    chart_out[pred_col] = df[pred_col].values
    chart_out.to_csv(out / "december_chart_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(chart_out.date, chart_out[pred_col], marker="o", lw=1.8)
    lane = f"{chart.pickup.iloc[0]} → {chart.delivery.iloc[0]}, {chart.equipment.iloc[0]}, {int(chart.distance.iloc[0])} mi, {int(chart.weight.iloc[0]):,} lb"
    ax.set_title(f"Predicted rate, December 2025\n{lane}")
    ax.set_ylabel("Predicted rate ($)"); ax.grid(alpha=.3)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out / "december_chart.png", dpi=150)
    plt.close(fig)
    return chart_out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--skip-cv", action="store_true", help="skip the (slower) time-based CV")
    args = ap.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.out_dir)
    if not out.is_absolute() and not (Path.cwd() / "src").exists():
        out = Path(__file__).resolve().parent / out
    out.mkdir(parents=True, exist_ok=True)

    print("1/6 loading data")
    train, val, template = load_data(data_dir)

    print("2/6 data-quality report")
    cleaner = Cleaner().fit(train)
    rep = data_quality_report(train, val, cleaner, out)
    print(json.dumps(rep, indent=2))
    print(quote_signal_by_month(train, cleaner, out).round(3))

    if not args.skip_cv:
        print("3/6 time-based cross-validation (train on earlier months, test on the next month)")
        _, summary = time_cv(train, out)
        print("\nMean over folds (months 6-10):\n", summary)
    else:
        print("3/6 CV skipped")

    print("4/6 fitting final model on ALL train_test.csv data")
    tr_feat = cleaner.transform(train)
    keep = ~cleaner.label_outlier_mask(train)
    model = RateModel().fit(tr_feat[keep])
    imp = pd.DataFrame({
        "feature": QS_FEATURES,
        "importance_qs_model": np.mean([m.feature_importances_ for m in model.qs], axis=0),
    })
    imp.sort_values("importance_qs_model", ascending=False).to_csv(out / "feature_importance.csv", index=False)

    print("5/6 predicting validation.csv")
    va_feat = cleaner.transform(val)
    pred = pd.DataFrame({"load_id": val.load_id.values, "predicted_rate": model.predict(va_feat).round(2)})
    final = template[["load_id"]].merge(pred, on="load_id", how="left")
    assert len(final) == len(template) == len(val), "row count mismatch"
    assert final.predicted_rate.notna().all(), "missing predictions"
    assert (final.predicted_rate > 0).all(), "non-positive predictions"
    final.to_csv(out / "validation_predictions.csv", index=False)
    rpm = final.predicted_rate.values / val.set_index("load_id").loc[final.load_id, "distance"].values
    print(f"   wrote {len(final)} rows; predicted rate/mile median={np.median(rpm):.2f}, "
          f"min={rpm.min():.2f}, max={rpm.max():.2f}")

    print("6/6 December chart")
    chart_path = find_file(data_dir, "december_chart_inputs", required=False)
    if chart_path is not None:
        ch = december_chart(chart_path, train, val, cleaner, model, out)
        print(ch.head(3).to_string(index=False), "\n   ...")
    else:
        print("   december_chart_inputs.csv not found in data dir - skipped")
    print("\nDone. Files in", out.resolve())


if __name__ == "__main__":
    main()