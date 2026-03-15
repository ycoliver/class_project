import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from itertools import combinations

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Weekly.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_weekly():
    return pd.read_csv(DATA_URL)


def summary_and_plots(df):
    cols = ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume", "Today"]
    corr = df[cols].corr()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap (Weekly)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "weekly_corr_heatmap.png"), dpi=200)
    plt.close(fig)

    # Volume time series
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["Year"] + df.index / len(df), df["Volume"], color="#2c7fb8", linewidth=1)
    ax.set_xlabel("Year (approx)")
    ax.set_ylabel("Volume")
    ax.set_title("Weekly Volume over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "weekly_volume_timeseries.png"), dpi=200)
    plt.close(fig)

    # Today by Direction
    fig, ax = plt.subplots(figsize=(5, 4))
    df.boxplot(column="Today", by="Direction", ax=ax)
    ax.set_title("Today Return by Direction")
    ax.set_xlabel("Direction")
    ax.set_ylabel("Today")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "weekly_today_box.png"), dpi=200)
    plt.close(fig)


def logistic_full(df):
    X = df[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]]
    y = (df["Direction"] == "Up").astype(int)
    X_sm = sm.add_constant(X)
    model = sm.Logit(y, X_sm).fit(disp=False)
    pred_prob = model.predict(X_sm)
    pred = (pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    acc = accuracy_score(y, pred)
    return model, cm, acc


def train_test_models(df):
    y = (df["Direction"] == "Up").astype(int)
    train = df["Year"] <= 2008
    test = df["Year"] >= 2009

    X_train = df.loc[train, ["Lag2"]]
    X_test = df.loc[test, ["Lag2"]]
    y_train = y[train]
    y_test = y[test]

    results = {}
    # Logistic
    logit = LogisticRegression(solver="liblinear")
    logit.fit(X_train, y_train)
    pred = logit.predict(X_test)
    results["Logit"] = (confusion_matrix(y_test, pred), accuracy_score(y_test, pred))

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    pred = lda.predict(X_test)
    results["LDA"] = (confusion_matrix(y_test, pred), accuracy_score(y_test, pred))

    # QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    pred = qda.predict(X_test)
    results["QDA"] = (confusion_matrix(y_test, pred), accuracy_score(y_test, pred))

    # KNN k=1
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    results["KNN (k=1)"] = (confusion_matrix(y_test, pred), accuracy_score(y_test, pred))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred = nb.predict(X_test)
    results["Naive Bayes"] = (confusion_matrix(y_test, pred), accuracy_score(y_test, pred))

    return results


def search_best_combo(df):
    y = (df["Direction"] == "Up").astype(int)
    train = df["Year"] <= 2008
    test = df["Year"] >= 2009
    y_train = y[train]
    y_test = y[test]

    base = df[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]].copy()
    base["Lag1_sq"] = df["Lag1"] ** 2
    base["Lag2_sq"] = df["Lag2"] ** 2
    base["Lag3_sq"] = df["Lag3"] ** 2
    base["Lag1_Lag2"] = df["Lag1"] * df["Lag2"]
    base["Lag2_Lag3"] = df["Lag2"] * df["Lag3"]
    base["LogVolume"] = np.log(df["Volume"])

    predictors = list(base.columns)
    results = []
    for r in [1, 2, 3]:
        for combo in combinations(predictors, r):
            Xtr = base.loc[train, list(combo)]
            Xte = base.loc[test, list(combo)]

            logit = LogisticRegression(solver="liblinear")
            logit.fit(Xtr, y_train)
            pred = logit.predict(Xte)
            results.append(("Logit", combo, None, accuracy_score(y_test, pred), confusion_matrix(y_test, pred)))

            lda = LinearDiscriminantAnalysis()
            lda.fit(Xtr, y_train)
            pred = lda.predict(Xte)
            results.append(("LDA", combo, None, accuracy_score(y_test, pred), confusion_matrix(y_test, pred)))

            qda = QuadraticDiscriminantAnalysis()
            qda.fit(Xtr, y_train)
            pred = qda.predict(Xte)
            results.append(("QDA", combo, None, accuracy_score(y_test, pred), confusion_matrix(y_test, pred)))

            nb = GaussianNB()
            nb.fit(Xtr, y_train)
            pred = nb.predict(Xte)
            results.append(("Naive Bayes", combo, None, accuracy_score(y_test, pred), confusion_matrix(y_test, pred)))

            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            for k in [1, 3, 5, 7, 9, 15, 21]:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(Xtr_s, y_train)
                pred = knn.predict(Xte_s)
                results.append(("KNN", combo, k, accuracy_score(y_test, pred), confusion_matrix(y_test, pred)))

    best = max(results, key=lambda x: x[3])
    return best


def power_plots():
    x = np.arange(1, 11)
    y = x ** 2
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("x^2")
    ax.set_title("f(x)=x^2 (1..10)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "power_x2.png"), dpi=200)
    plt.close(fig)

    y3 = x ** 3
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(x, y3, marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("x^3")
    ax.set_title("PlotPower(x, a=3)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "power_plotpower_a3.png"), dpi=200)
    plt.close(fig)


def main():
    df = load_weekly()
    summary_and_plots(df)

    model, cm_full, acc_full = logistic_full(df)
    print(model.summary())
    print("Full-data confusion matrix:\n", cm_full)
    print("Full-data accuracy:", acc_full)

    results = train_test_models(df)
    for name, (cm, acc) in results.items():
        print(name, "\n", cm, "acc=", acc)

    best = search_best_combo(df)
    print("Best search:", best)

    power_plots()


if __name__ == "__main__":
    main()
