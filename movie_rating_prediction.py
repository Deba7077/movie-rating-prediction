

import os
import warnings

import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
CSV_PATH = "IMDb Movies India.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


USE_VOTES = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------------------------------
def load_data(path):
   
    df = pd.read_csv(path, encoding="latin-1")
    print("=" * 70)
    print("STEP 1: DATA LOADED")
    print("=" * 70)
    print("Shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head().to_string())
    print("\nMissing values per column:\n", df.isna().sum())
    return df


# --------------------------------------------------------------------------
# STEP 2: CLEANING / PREPROCESSING
# --------------------------------------------------------------------------
def clean_data(df):
    df = df.copy()

    # Year "(2019)" -> 2019
    df["Year"] = pd.to_numeric(df["Year"].astype(str).str.extract(r"(\d{4})")[0])

    # Duration "109 min" -> 109
    df["Duration"] = pd.to_numeric(
        df["Duration"].astype(str).str.extract(r"(\d+)")[0])

    # Votes "1,086" -> 1086
    df["Votes"] = pd.to_numeric(
        df["Votes"].astype(str).str.replace(",", "", regex=False),
        errors="coerce")

    # Rows without a rating cannot be used to train/evaluate a model.
    before = len(df)
    df = df.dropna(subset=["Rating"]).reset_index(drop=True)
    print("\n" + "=" * 70)
    print("STEP 2: CLEANING")
    print("=" * 70)
    print(f"Removed {before - len(df)} movies with no rating -> {len(df)} left")

    # Missing text -> "Unknown"
    for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
        df[col] = df[col].fillna("Unknown").str.strip()

    return df


# --------------------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (charts saved to outputs/)
# --------------------------------------------------------------------------
def explore(df):
    print("\n" + "=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(df[["Year", "Duration", "Votes", "Rating"]].describe().round(2))

    # 3a. Rating distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Rating"], bins=30, kde=True, color="steelblue")
    plt.title("Distribution of Movie Ratings")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_rating_distribution.png", dpi=120)
    plt.close()

    # 3b. Average rating by genre (top 12 individual genres)
    g = df.assign(G=df["Genre"].str.split(",")).explode("G")
    g["G"] = g["G"].str.strip()
    gstats = g.groupby("G")["Rating"].agg(["mean", "count"])
    gstats = gstats[gstats["count"] >= 30].sort_values("mean", ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=gstats["mean"].head(12), y=gstats.head(12).index,
                palette="viridis")
    plt.xlabel("Average Rating")
    plt.title("Top Genres by Average Rating (min 30 movies)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_rating_by_genre.png", dpi=120)
    plt.close()

    # 3c. Average rating over the years
    yearly = df.dropna(subset=["Year"]).groupby("Year")["Rating"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(yearly.index, yearly.values, color="darkred")
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    plt.title("Average Rating by Release Year")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_rating_by_year.png", dpi=120)
    plt.close()

    # 3d. Correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[["Year", "Duration", "Votes", "Rating"]].corr(),
                annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=120)
    plt.close()

    # 3e. Top directors & actors
    top_dir = (df[df["Director"] != "Unknown"].groupby("Director")["Rating"]
               .agg(["mean", "count"]).query("count >= 5")
               .sort_values("mean", ascending=False).head(10))
    print("\nTop 10 directors by average rating (min 5 movies):")
    print(top_dir.round(2))

    print(f"\nEDA charts saved in '{OUTPUT_DIR}/'")


# --------------------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# --------------------------------------------------------------------------
class MeanEncoder(BaseEstimator, TransformerMixin):
    

    def __init__(self, smoothing=5, n_splits=5, random_state=42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def _stats(self, x, y):
        d = pd.DataFrame({"k": x.values, "y": y.values})
        s = d.groupby("k")["y"].agg(["sum", "count"])
        return s

    def _encode(self, x, s, gmean):
        sm = self.smoothing
        sums = x.map(s["sum"]).fillna(0)
        cnts = x.map(s["count"]).fillna(0)
        return (sums + sm * gmean) / (cnts + sm)

    def fit(self, x, y):
        x = pd.Series(x).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.gmean_ = y.mean()
        self.stats_ = self._stats(x, y)
        return self

    def transform(self, x):
        return self._encode(pd.Series(x).reset_index(drop=True),
                            self.stats_, self.gmean_).values

    def fit_transform(self, x, y=None):
        x = pd.Series(x).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.fit(x, y)
        out = np.zeros(len(x))
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(x):
            s = self._stats(x.iloc[tr], y.iloc[tr])
            out[va] = self._encode(x.iloc[va], s, y.iloc[tr].mean()).values
        return out


def build_features(train_df, test_df):
    """Build numeric feature matrices for train and test sets."""
    ytr = train_df["Rating"].reset_index(drop=True)
    Xtr, Xte = pd.DataFrame(), pd.DataFrame()

    # --- Numeric features -------------------------------------------------
    median_year = train_df["Year"].median()
    median_dur = train_df["Duration"].median()
    for X, d in ((Xtr, train_df), (Xte, test_df)):
        X["Year"] = d["Year"].fillna(median_year).values
        X["Duration"] = d["Duration"].fillna(median_dur).values
        X["Duration_missing"] = d["Duration"].isna().astype(int).values
        if USE_VOTES:
            X["Log_Votes"] = np.log1p(d["Votes"].fillna(0)).values

    # --- Genre: multi-hot encoding (Drama, Comedy, ...) -------------------
    all_genres = sorted({g.strip() for s in train_df["Genre"]
                         for g in s.split(",") if g.strip() != "Unknown"})
    for g in all_genres:
        Xtr["Genre_" + g] = train_df["Genre"].apply(
            lambda s: int(g in [i.strip() for i in s.split(",")])).values
        Xte["Genre_" + g] = test_df["Genre"].apply(
            lambda s: int(g in [i.strip() for i in s.split(",")])).values
    Xtr["Num_Genres"] = train_df["Genre"].str.count(",").values + 1
    Xte["Num_Genres"] = test_df["Genre"].str.count(",").values + 1

    # --- Director: mean-rating encoding -----------------------------------
    enc = MeanEncoder()
    Xtr["Director_enc"] = enc.fit_transform(train_df["Director"], ytr)
    Xte["Director_enc"] = enc.transform(test_df["Director"])

    
    acols = ["Actor 1", "Actor 2", "Actor 3"]
    long_tr = pd.concat([pd.DataFrame({"a": train_df[c].values, "y": ytr.values})
                         for c in acols], ignore_index=True)
    long_tr = long_tr[long_tr["a"] != "Unknown"]
    aenc = MeanEncoder().fit(long_tr["a"], long_tr["y"])
    gm = ytr.mean()

    # Out-of-fold encoding for the TRAIN set to prevent leakage
    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    tr_actor = np.zeros((len(train_df), 3))
    tr_dir_idx = np.arange(len(train_df))
    for tr_i, va_i in kf.split(tr_dir_idx):
        sub = train_df.iloc[tr_i]
        ysub = ytr.iloc[tr_i]
        lt = pd.concat([pd.DataFrame({"a": sub[c].values, "y": ysub.values})
                        for c in acols], ignore_index=True)
        lt = lt[lt["a"] != "Unknown"]
        e = MeanEncoder().fit(lt["a"], lt["y"])
        for j, c in enumerate(acols):
            tr_actor[va_i, j] = e.transform(train_df.iloc[va_i][c])
    te_actor = np.column_stack([aenc.transform(test_df[c]) for c in acols])

    for j, c in enumerate(["Actor1_enc", "Actor2_enc", "Actor3_enc"]):
        Xtr[c] = tr_actor[:, j]
        Xte[c] = te_actor[:, j]
    Xtr["Actor_mean"] = tr_actor.mean(axis=1)
    Xte["Actor_mean"] = te_actor.mean(axis=1)

    # Experience: how many movies does the director have in training data?
    dcount = train_df["Director"].value_counts()
    Xtr["Director_movies"] = train_df["Director"].map(dcount).values
    Xte["Director_movies"] = test_df["Director"].map(dcount).fillna(0).values
    Xtr.loc[train_df["Director"].values == "Unknown", "Director_movies"] = 0
    Xte.loc[test_df["Director"].values == "Unknown", "Director_movies"] = 0

    return Xtr, Xte


# --------------------------------------------------------------------------
# STEP 5: MODEL TRAINING & EVALUATION
# --------------------------------------------------------------------------
def evaluate(name, model, Xtr, ytr, Xte, yte, results, scale=False):
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr_, Xte_ = sc.transform(Xtr), sc.transform(Xte)
    else:
        Xtr_, Xte_ = Xtr, Xte
    model.fit(Xtr_, ytr)
    pred = np.clip(model.predict(Xte_), 1, 10)
    results[name] = {
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "R2": r2_score(yte, pred),
    }
    return model, pred


def main():
    df = load_data(CSV_PATH)
    df = clean_data(df)
    explore(df)

    # Split BEFORE feature engineering so test data never leaks into encoders
    train_df, test_df = train_test_split(df, test_size=0.2,
                                         random_state=RANDOM_STATE)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)
    Xtr, Xte = build_features(train_df, test_df)
    ytr, yte = train_df["Rating"].values, test_df["Rating"].values
    print(f"Train: {Xtr.shape}   Test: {Xte.shape}")
    print("Using Votes feature:", USE_VOTES)

    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    results, preds = {}, {}

    # Baseline: always predict the average rating
    base = np.full(len(yte), ytr.mean())
    results["Baseline (mean)"] = {
        "MAE": mean_absolute_error(yte, base),
        "RMSE": np.sqrt(mean_squared_error(yte, base)),
        "R2": r2_score(yte, base)}

    _, preds["Ridge Regression"] = evaluate(
        "Ridge Regression", Ridge(alpha=1.0), Xtr, ytr, Xte, yte,
        results, scale=True)

    rf, preds["Random Forest"] = evaluate(
        "Random Forest",
        RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                              n_jobs=-1, random_state=RANDOM_STATE),
        Xtr, ytr, Xte, yte, results)

    gb, preds["Gradient Boosting"] = evaluate(
        "Gradient Boosting",
        GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=4, subsample=0.8,
                                  random_state=RANDOM_STATE),
        Xtr, ytr, Xte, yte, results)

    res = pd.DataFrame(results).T.round(3)
    print(res)
    best = res.drop("Baseline (mean)")["RMSE"].idxmin()
    print(f"\nBest model (lowest RMSE): {best}")
    res.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")

    # Chart: model comparison
    plt.figure(figsize=(8, 5))
    res["RMSE"].plot(kind="bar", color="teal")
    plt.ylabel("RMSE (lower is better)")
    plt.title("Model Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_model_comparison.png", dpi=120)
    plt.close()

    # Chart: actual vs predicted for the best model
    bp = preds[best]
    plt.figure(figsize=(6, 6))
    plt.scatter(yte, bp, alpha=0.4, color="purple")
    plt.plot([1, 10], [1, 10], "r--")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Actual vs Predicted ({best})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_actual_vs_predicted.png", dpi=120)
    plt.close()

    # Chart: feature importance (tree model)
    model = gb if best == "Gradient Boosting" else rf
    imp = pd.Series(model.feature_importances_, index=Xtr.columns)
    imp = imp.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, palette="magma")
    plt.title("Top 15 Features Influencing Rating")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_feature_importance.png", dpi=120)
    plt.close()
    print("\nTop 10 most important features:")
    print(imp.head(10).round(3))

    # Sample predictions
    sample = test_df[["Name", "Genre", "Director", "Rating"]].head(10).copy()
    sample["Predicted"] = np.round(bp[:10], 1)
    print("\nSample predictions (first 10 test movies):")
    print(sample.to_string(index=False))

    print("\nDone. All charts and model_comparison.csv are in "
          f"'{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()