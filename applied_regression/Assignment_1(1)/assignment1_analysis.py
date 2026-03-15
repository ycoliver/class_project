#!/usr/bin/env python3
"""Assignment 1 analysis helper (Python).

This script generates numeric summaries and (optionally) figures for the
College and Boston datasets. Outputs are written to ./outputs.

Usage:
  python assignment1_analysis.py

Notes:
- Uses only standard library by default.
- If pandas/matplotlib are available, it will also create figures.
"""

from __future__ import annotations

from pathlib import Path
import csv
import math
import statistics
import sys
import urllib.request

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BOSTON_URL = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"


# ----- helpers -----

def quantile_type7(x: list[float], q: float) -> float:
    """R default quantile (type=7)."""
    x = sorted(x)
    n = len(x)
    if n == 0:
        return float("nan")
    if q <= 0:
        return x[0]
    if q >= 1:
        return x[-1]
    h = (n - 1) * q + 1
    k = int(math.floor(h))
    d = h - k
    if k <= 0:
        return x[0]
    if k >= n:
        return x[-1]
    return x[k - 1] + d * (x[k] - x[k - 1])


def corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return cov / (sx * sy)


def read_csv_dict(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_text(path: Path, lines: list[str]) -> None:
    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


# ----- College dataset -----

def analyze_college() -> None:
    college_path = BASE_DIR / "College.csv"
    if not college_path.exists():
        print("College.csv not found in the assignment folder.")
        return

    cols, rows = read_csv_dict(college_path)
    if not cols:
        print("College.csv is empty or missing headers.")
        return

    name_col = cols[0]
    numeric_cols = [c for c in cols[1:] if c != "Private"]

    # Build numeric data
    num_data: dict[str, list[float]] = {c: [] for c in numeric_cols}
    private_vals: list[str] = []
    top10_vals: list[float] = []
    outstate_vals: list[float] = []

    for r in rows:
        private_vals.append(r["Private"])
        top10_vals.append(float(r["Top10perc"]))
        outstate_vals.append(float(r["Outstate"]))
        for c in numeric_cols:
            num_data[c].append(float(r[c]))

    # Summary table
    summary_rows: list[list[str]] = []
    for c in numeric_cols:
        vals = num_data[c]
        summary_rows.append([
            c,
            f"{min(vals):.6g}",
            f"{quantile_type7(vals, 0.25):.6g}",
            f"{quantile_type7(vals, 0.5):.6g}",
            f"{(sum(vals) / len(vals)):.6g}",
            f"{quantile_type7(vals, 0.75):.6g}",
            f"{max(vals):.6g}",
        ])

    write_csv(
        OUT_DIR / "college_summary.csv",
        ["Variable", "Min", "1stQu", "Median", "Mean", "3rdQu", "Max"],
        summary_rows,
    )

    # Private counts
    private_counts: dict[str, int] = {}
    for v in private_vals:
        private_counts[v] = private_counts.get(v, 0) + 1

    # Elite counts
    elite = ["Yes" if v > 50 else "No" for v in top10_vals]
    elite_counts = {"Yes": elite.count("Yes"), "No": elite.count("No")}

    # Group stats for Outstate
    def group_stats(groups: list[str], values: list[float]) -> dict[str, dict[str, float]]:
        bucket: dict[str, list[float]] = {}
        for g, v in zip(groups, values):
            bucket.setdefault(g, []).append(v)
        stats: dict[str, dict[str, float]] = {}
        for g, vals in bucket.items():
            stats[g] = {
                "Min": min(vals),
                "Q1": quantile_type7(vals, 0.25),
                "Median": quantile_type7(vals, 0.5),
                "Q3": quantile_type7(vals, 0.75),
                "Max": max(vals),
            }
        return stats

    outstate_by_private = group_stats(private_vals, outstate_vals)
    outstate_by_elite = group_stats(elite, outstate_vals)

    lines = [
        f"Rows: {len(rows)}",
        f"Columns: {len(cols) - 1} (excluding college name)",
        f"Private counts: {private_counts}",
        f"Elite counts (Top10perc > 50): {elite_counts}",
        f"Outstate by Private: {outstate_by_private}",
        f"Outstate by Elite: {outstate_by_elite}",
    ]
    write_text(OUT_DIR / "college_counts.txt", lines)

    # Correlation highlights for interpretation
    corr_pairs: list[tuple[float, float, str, str]] = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1 :]:
            r = corr(num_data[c1], num_data[c2])
            corr_pairs.append((abs(r), r, c1, c2))

    corr_pairs.sort(reverse=True)
    top_pairs = [f"{c1},{c2},{r:.4f}" for _, r, c1, c2 in corr_pairs[:15]]
    write_text(OUT_DIR / "college_top_correlations.txt", ["var1,var2,corr"] + top_pairs)


# ----- Boston dataset -----

def ensure_boston_csv() -> Path | None:
    candidates = [
        BASE_DIR / "Boston.csv",
        BASE_DIR / "boston.csv",
        BASE_DIR / "BostonHousing.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Try download if not present
    target = BASE_DIR / "Boston.csv"
    try:
        with urllib.request.urlopen(BOSTON_URL, timeout=30) as resp:
            target.write_text(resp.read().decode("utf-8"))
        return target
    except Exception as exc:
        print(f"Failed to download Boston dataset: {exc}")
        return None


def analyze_boston() -> None:
    boston_path = ensure_boston_csv()
    if not boston_path:
        return

    cols, rows = read_csv_dict(boston_path)
    if not cols:
        print("Boston dataset is empty or missing headers.")
        return

    data: dict[str, list[float]] = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            data[c].append(float(r[c]))

    # Basic info
    info_lines = [
        f"Rows: {len(rows)}",
        f"Columns: {len(cols)}",
    ]

    # Correlations with crim
    crim = data["crim"]
    corrs = {c: corr(crim, data[c]) for c in cols if c != "crim"}
    corrs_sorted = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    corr_lines = ["var,corr"] + [f"{c},{r:.4f}" for c, r in corrs_sorted]

    # Max locations
    def max_info(col: str) -> tuple[float, int]:
        vals = data[col]
        mx = max(vals)
        idx = vals.index(mx)
        return mx, idx + 1  # 1-based row

    max_crim, row_crim = max_info("crim")
    max_tax, row_tax = max_info("tax")
    max_ptratio, row_ptratio = max_info("ptratio")

    info_lines.extend([
        f"Max crim: {max_crim} (row {row_crim})",
        f"Max tax: {max_tax} (row {row_tax})",
        f"Max ptratio: {max_ptratio} (row {row_ptratio})",
    ])

    # chas count
    chas_count = sum(1 for v in data["chas"] if v == 1.0)
    info_lines.append(f"chas == 1 count: {chas_count}")

    # median ptratio
    ptratio_median = quantile_type7(data["ptratio"], 0.5)
    info_lines.append(f"Median ptratio: {ptratio_median}")

    # min medv tract
    medv_vals = data["medv"]
    min_medv = min(medv_vals)
    row_min_medv = medv_vals.index(min_medv) + 1
    info_lines.append(f"Min medv: {min_medv} (row {row_min_medv})")

    # rm counts
    rm_gt7 = sum(1 for v in data["rm"] if v > 7)
    rm_gt8 = sum(1 for v in data["rm"] if v > 8)
    info_lines.append(f"rm > 7: {rm_gt7}")
    info_lines.append(f"rm > 8: {rm_gt8}")

    write_text(OUT_DIR / "boston_info.txt", info_lines)
    write_text(OUT_DIR / "boston_corrs_crim.csv", corr_lines)

    # Ranges
    ranges = [
        [c, f"{min(data[c]):.6g}", f"{max(data[c]):.6g}"]
        for c in cols
    ]
    write_csv(OUT_DIR / "boston_ranges.csv", ["Variable", "Min", "Max"], ranges)

    # Values for min medv row
    tract_vals = [
        [c, f"{data[c][row_min_medv - 1]:.6g}"]
        for c in cols
    ]
    write_csv(OUT_DIR / "boston_min_medv_row.csv", ["Variable", "Value"], tract_vals)


# ----- Optional figures (pandas/matplotlib) -----

def try_make_figures() -> None:
    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("pandas/matplotlib not found; skipping figures.")
        return

    # College figures
    college_path = BASE_DIR / "College.csv"
    if college_path.exists():
        college = pd.read_csv(college_path)
        college.index = college.iloc[:, 0]
        college = college.iloc[:, 1:]

        # Boxplots
        plt.figure(figsize=(6, 4))
        college.boxplot(column="Outstate", by="Private")
        plt.title("Outstate vs Private")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "college_outstate_private.png", dpi=200)
        plt.close()

        # Elite variable
        college["Elite"] = pd.Series(
            ["Yes" if v > 50 else "No" for v in college["Top10perc"]], index=college.index
        )
        plt.figure(figsize=(6, 4))
        college.boxplot(column="Outstate", by="Elite")
        plt.title("Outstate vs Elite")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "college_outstate_elite.png", dpi=200)
        plt.close()

        # Histograms
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        college["Apps"].plot.hist(ax=axes[0, 0], bins=30, title="Apps")
        college["Outstate"].plot.hist(ax=axes[0, 1], bins=30, title="Outstate")
        college["Expend"].plot.hist(ax=axes[1, 0], bins=30, title="Expend")
        college["Grad.Rate"].plot.hist(ax=axes[1, 1], bins=30, title="Grad.Rate")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "college_histograms.png", dpi=200)
        plt.close()

        # Scatterplot matrix (first 10 vars)
        try:
            from pandas.plotting import scatter_matrix

            fig = scatter_matrix(college.iloc[:, :10], figsize=(12, 12), diagonal="kde")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "college_pairs_first10.png", dpi=200)
            plt.close()
        except Exception:
            print("Failed to create College scatter matrix.")

    # Boston figures
    boston_path = ensure_boston_csv()
    if boston_path and boston_path.exists():
        boston = pd.read_csv(boston_path)
        try:
            from pandas.plotting import scatter_matrix

            fig = scatter_matrix(boston.drop(columns=["medv"]), figsize=(14, 14), diagonal="kde")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "boston_pairs_predictors.png", dpi=200)
            plt.close()
        except Exception:
            print("Failed to create Boston scatter matrix.")


def main() -> None:
    analyze_college()
    analyze_boston()
    try_make_figures()
    print(f"Done. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
