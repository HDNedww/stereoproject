#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_rows(pattern: str, experiment_id: str):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if experiment_id and row.get("experiment_id", "") != experiment_id:
                    continue
                try:
                    gt = float(row.get("ground_truth_m", ""))
                    z = float(row.get("person_z_m", ""))
                except Exception:
                    continue
                if gt < 0.0 or z <= 0.0:
                    continue
                if row.get("depth_valid", "0") not in ("1", "True", "true"):
                    continue
                rows.append(
                    {
                        "ground_truth_m": gt,
                        "person_z_m": z,
                        "lighting": row.get("lighting", ""),
                        "scenario": row.get("scenario", ""),
                        "source_file": os.path.basename(path),
                    }
                )
    return rows


def mean(vals):
    return sum(vals) / max(1, len(vals))


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main():
    ap = argparse.ArgumentParser(description="Plot experiment #1 distance accuracy results.")
    ap.add_argument("--input-glob", default="/home/nedas/dev_ws/experiment_logs/*.csv")
    ap.add_argument("--experiment-id", default="distance_accuracy")
    ap.add_argument("--output-dir", default="/home/nedas/dev_ws/experiment_logs/plots")
    args = ap.parse_args()

    rows = read_rows(args.input_glob, args.experiment_id)
    if not rows:
        raise SystemExit("No valid rows found. Check --input-glob / --experiment-id.")

    os.makedirs(args.output_dir, exist_ok=True)

    by_gt = defaultdict(list)
    all_abs_errors = []
    sq_errors = []
    for r in rows:
        gt = r["ground_truth_m"]
        z = r["person_z_m"]
        by_gt[gt].append(z)
        e = abs(z - gt)
        all_abs_errors.append(e)
        sq_errors.append((z - gt) ** 2)

    gts = sorted(by_gt.keys())
    z_means = [mean(by_gt[g]) for g in gts]
    z_stds = [std(by_gt[g]) for g in gts]
    abs_mean_err = [mean([abs(z - g) for z in by_gt[g]]) for g in gts]

    mae = mean(all_abs_errors)
    rmse = math.sqrt(mean(sq_errors))

    # 1) Ground truth vs measured mean (+/- std)
    plt.figure(figsize=(8, 5))
    plt.errorbar(gts, z_means, yerr=z_stds, fmt="o-", capsize=4, label="Measured mean ± std")
    plt.plot(gts, gts, "--", label="Ideal y=x")
    plt.xlabel("Ground truth distance (m)")
    plt.ylabel("Measured distance (m)")
    plt.title(f"Experiment 1: Accuracy ({args.experiment_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p1 = os.path.join(args.output_dir, f"{args.experiment_id}_accuracy_curve.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # 2) Mean absolute error per point
    plt.figure(figsize=(8, 5))
    plt.bar([f"{g:.2f}" for g in gts], abs_mean_err)
    plt.xlabel("Ground truth distance (m)")
    plt.ylabel("Mean absolute error (m)")
    plt.title(f"Absolute error per distance point | MAE={mae:.3f} m, RMSE={rmse:.3f} m")
    plt.grid(True, axis="y", alpha=0.3)
    p2 = os.path.join(args.output_dir, f"{args.experiment_id}_abs_error_bar.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    # 3) Scatter all samples
    plt.figure(figsize=(8, 5))
    x = [r["ground_truth_m"] for r in rows]
    y = [r["person_z_m"] for r in rows]
    plt.scatter(x, y, s=10, alpha=0.7)
    mn = min(min(x), min(y))
    mx = max(max(x), max(y))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Ground truth distance (m)")
    plt.ylabel("Measured distance (m)")
    plt.title("All samples")
    plt.grid(True, alpha=0.3)
    p3 = os.path.join(args.output_dir, f"{args.experiment_id}_scatter.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()

    print(f"Rows used: {len(rows)}")
    print(f"Unique points: {len(gts)}")
    print(f"MAE:  {mae:.4f} m")
    print(f"RMSE: {rmse:.4f} m")
    print("Saved:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")


if __name__ == "__main__":
    main()
