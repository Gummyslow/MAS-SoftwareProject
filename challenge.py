"""
challenge.py – Entry point for the Fraud Detection MAS challenge.

Usage
-----
# Train on labelled data (single CSV or full 5-file dataset):
  python challenge.py --train transactions.csv
  python challenge.py --train transactions.csv --users users.json --locations locations.json \\
                      --sms sms.json --mails mails.json

# Inference only (model already trained):
  python challenge.py --test transactions.csv
  python challenge.py --test transactions.csv --users users.json --locations locations.json \\
                      --sms sms.json --mails mails.json

# Train + inference + evaluation:
  python challenge.py --train train.csv --test test.csv --labels labels.csv \\
                      --users users.json --locations locations.json \\
                      --sms sms.json --mails mails.json
"""

import argparse
import sys

import pandas as pd

from fraud_mas.config import SUBMISSION_PATH, XGB_MODEL_PATH, LABEL_ENC_PATH
from fraud_mas.data_io import (
    load_and_merge_dataset,
    load_csv,
    load_label_encoders,
    load_model,
    save_label_encoders,
    save_model,
)
from fraud_mas.features import engineer_features
from fraud_mas.behavioral import compute_behavioral_signals
from fraud_mas.geo import compute_geo_signals
from fraud_mas.nlp_risk import compute_nlp_signals
from fraud_mas.network_analysis import compute_network_signals
from fraud_mas.model import train, evaluate
from fraud_mas.pipeline import run_pipeline, write_submission_file


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(args) -> None:
    print(f"[challenge] Loading training data from {args.train}...")
    df = load_and_merge_dataset(
        transactions_path=args.train,
        users_path=args.users,
        locations_path=args.locations,
        sms_path=args.sms,
        mails_path=args.mails,
    )
    print(f"[challenge] Loaded {len(df):,} rows × {df.shape[1]} columns")

    print("[challenge] Running agents on training data...")
    df, encoders = engineer_features(df, fit=True)
    df = compute_behavioral_signals(df)
    df = compute_geo_signals(df)
    df = compute_nlp_signals(df)
    df = compute_network_signals(df)

    print("[challenge] Training XGBoost ensemble model...")
    model = train(df, label_col="label")

    save_label_encoders(encoders)
    print(f"[challenge] Model saved to {XGB_MODEL_PATH}")
    print(f"[challenge] Encoders saved to {LABEL_ENC_PATH}")

    # Quick in-sample sanity check
    from fraud_mas.model import predict_proba, apply_thresholds
    import numpy as np
    scores = predict_proba(df, model)
    metrics = evaluate(df, scores, label_col="label")
    print(f"[challenge] Train metrics → ROC-AUC: {metrics['roc_auc']}  F1: {metrics['f1']}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(args, output_path: str | None = None) -> pd.DataFrame:
    print(f"[challenge] Loading test data from {args.test}...")
    df = load_and_merge_dataset(
        transactions_path=args.test,
        users_path=args.users,
        locations_path=args.locations,
        sms_path=args.sms,
        mails_path=args.mails,
    )
    print(f"[challenge] Loaded {len(df):,} rows × {df.shape[1]} columns")

    model    = load_model()
    encoders = load_label_encoders()

    results = run_pipeline(df, model=model, encoders=encoders)

    out = output_path or str(SUBMISSION_PATH)
    write_submission_file(results, path=out)
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(results: pd.DataFrame, labels_path: str) -> None:
    print(f"[challenge] Loading ground-truth labels from {labels_path}...")
    labels_df = load_csv(labels_path)

    # Merge predictions with ground truth
    merged = results.merge(labels_df, on="transaction_id", suffixes=("_pred", "_true"))
    y_true = (merged["label_true"] == "fraud").astype(int)
    y_pred = (merged["label_pred"] == "fraud").astype(int)
    scores = merged["model_score"] if "model_score" in merged.columns else y_pred.astype(float)

    from sklearn.metrics import classification_report, roc_auc_score
    print("\n[challenge] === Evaluation Results ===")
    print(f"  ROC-AUC : {roc_auc_score(y_true, scores):.4f}")
    print(classification_report(y_true, y_pred, target_names=["legit", "fraud"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fraud Detection Multi-Agent System – Challenge Runner"
    )
    # Core data files
    parser.add_argument("--train",     type=str, help="Path to labelled training CSV (transactions)")
    parser.add_argument("--test",      type=str, help="Path to test CSV (transactions) for inference")
    parser.add_argument("--labels",    type=str, help="Path to ground-truth labels CSV (for evaluation)")
    parser.add_argument("--output",    type=str, help="Output path for submission file", default=None)
    # Supplementary data files (all optional)
    parser.add_argument("--users",     type=str, default=None, help="Path to users.json")
    parser.add_argument("--locations", type=str, default=None, help="Path to locations.json")
    parser.add_argument("--sms",       type=str, default=None, help="Path to sms.json")
    parser.add_argument("--mails",     type=str, default=None, help="Path to mails.json")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if not args.train and not args.test:
        print("Error: provide at least --train or --test. Use --help for usage.")
        sys.exit(1)

    if args.train:
        run_training(args)

    if args.test:
        results = run_inference(args, output_path=args.output)

        if args.labels:
            run_evaluation(results, args.labels)


if __name__ == "__main__":
    main()
