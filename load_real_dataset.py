"""
load_real_dataset.py
────────────────────
Adapts any of the three recommended real churn datasets into the
ChurnShield format (data/raw/customers.csv) so you can run the full
pipeline without changing a single line of train.py or the API.

Supported datasets:
  1. IBM Telco Customer Churn  (Kaggle: blastchar/telco-customer-churn)
     7,043 rows · 21 columns · ~26% churn
     BEST CHOICE — most used in Australian DS interviews

  2. Orange Telecom Churn      (Kaggle: mnassrib/telecom-churn-datasets)
     4,250 rows · 21 columns · ~14% churn — closest to real AU telco churn rate
     BEST for realistic class imbalance

  3. UCI / KDD Telecom Churn   (UCI ML Repository)
     3,333 rows · 21 columns · ~14% churn
     Same structure as our synthetic data — easiest drop-in

Usage:
    python load_real_dataset.py --dataset ibm    --file WA_Fn-UseC_-Telco-Customer-Churn.csv
    python load_real_dataset.py --dataset orange --file churn-bigml-80.csv
    python load_real_dataset.py --dataset uci    --file churn-bigml-80.csv

After running, just do:  python -m src.models.train
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

AU_STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
STATE_WEIGHTS = [0.32, 0.26, 0.20, 0.10, 0.07, 0.02, 0.02, 0.01]
AREA_CODES = {
    "NSW": "02", "VIC": "03", "QLD": "07",
    "WA": "08", "SA": "08", "TAS": "03", "ACT": "02", "NT": "08"
}

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: IBM Telco Customer Churn
# Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# File:   WA_Fn-UseC_-Telco-Customer-Churn.csv
#
# Columns we map:
#   customerID, tenure (months), MonthlyCharges, TotalCharges
#   Contract (Month-to-month / One year / Two year)
#   InternetService, PhoneService, MultipleLines
#   OnlineSecurity, OnlineBackup, TechSupport
#   StreamingTV, StreamingMovies
#   PaperlessBilling, PaymentMethod
#   gender, SeniorCitizen, Partner, Dependents
#   Churn (Yes/No)
# ─────────────────────────────────────────────────────────────────────────────

def load_ibm(filepath: str) -> pd.DataFrame:
    """
    Map IBM Telco dataset → ChurnShield customers.csv schema.

    IBM doesn't have minute-by-minute call data, so we reconstruct
    day/evening/night/intl usage from MonthlyCharges and known AU telco
    charge rates (same rates used in generate_data.py).
    """
    print(f"Loading IBM Telco dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")

    # Clean TotalCharges (has spaces in some rows)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # tenure is in months → convert to days
    df["account_length_days"] = (df["tenure"] * 30.44).round(0).astype(int)

    # Reconstruct call usage from MonthlyCharges
    # AU charge rates: day=0.17/min, evening=0.085/min, night=0.045/min, intl=0.27/min
    # Split monthly charge roughly: day 50%, evening 30%, night 15%, intl 5%
    monthly = df["MonthlyCharges"]

    df["day_charge_aud"]     = (monthly * 0.50).round(2)
    df["evening_charge_aud"] = (monthly * 0.30).round(2)
    df["night_charge_aud"]   = (monthly * 0.15).round(2)
    df["intl_charge_aud"]    = (monthly * 0.05).round(2)

    df["day_mins"]     = (df["day_charge_aud"]     / 0.17).round(1)
    df["evening_mins"] = (df["evening_charge_aud"] / 0.085).round(1)
    df["night_mins"]   = (df["night_charge_aud"]   / 0.045).round(1)
    df["intl_mins"]    = (df["intl_charge_aud"]    / 0.27).round(1)

    # Calls — estimate from minutes (avg ~2 min/call for AU telco)
    df["day_calls"]     = (df["day_mins"]     / 2.0).round(0).astype(int).clip(0, 999)
    df["evening_calls"] = (df["evening_mins"] / 2.0).round(0).astype(int).clip(0, 999)
    df["night_calls"]   = (df["night_mins"]   / 2.0).round(0).astype(int).clip(0, 999)
    df["intl_calls"]    = (df["intl_mins"]    / 3.5).round(0).astype(int).clip(0, 99)

    # Plans
    df["international_plan"] = df["PhoneService"].str.strip() == "Yes"
    df["voicemail_plan"]     = df.get("MultipleLines", pd.Series(["No"]*len(df))).str.strip() == "Yes"
    df["voicemail_messages"] = np.where(df["voicemail_plan"], np.random.randint(0, 20, len(df)), 0)

    # Customer service calls — estimate from contract type and support services
    # Longer contract + tech support → fewer CS calls
    cs_base = np.random.poisson(1.5, len(df))
    cs_contract_bonus = np.where(df["Contract"] == "Month-to-month", 1, 0)
    cs_support_reduction = np.where(df.get("TechSupport", "No") == "Yes", -1, 0)
    df["customer_service_calls"] = (cs_base + cs_contract_bonus + cs_support_reduction).clip(0, 9)

    # AU state — assign randomly weighted by population
    df["state"]     = np.random.choice(AU_STATES, size=len(df), p=STATE_WEIGHTS)
    df["area_code"] = df["state"].map(AREA_CODES)

    # Churn label
    df["churned"] = df["Churn"].str.strip().str.lower() == "yes"

    # Customer ID
    df["customer_id"] = df["customerID"].str.strip()

    out = df[[
        "customer_id", "state", "account_length_days", "area_code",
        "international_plan", "voicemail_plan", "voicemail_messages",
        "day_mins", "day_calls", "day_charge_aud",
        "evening_mins", "evening_calls", "evening_charge_aud",
        "night_mins", "night_calls", "night_charge_aud",
        "intl_mins", "intl_calls", "intl_charge_aud",
        "customer_service_calls", "churned",
    ]].copy()

    print(f"  Mapped {len(out)} customers | Churn rate: {out['churned'].mean():.1%}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 & 3: Orange Telecom / UCI (same schema — both are the KDD dataset)
# Kaggle: https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
#   Files: churn-bigml-80.csv  (training, 2,666 rows)
#          churn-bigml-20.csv  (test,     667 rows)
# UCI:    https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
#   Direct: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset
#
# Columns (same as our synthetic data):
#   State, Account length, Area code, International plan (yes/no),
#   Voice mail plan (yes/no), Number vmail messages,
#   Total day minutes, Total day calls, Total day charge,
#   Total eve minutes, Total eve calls, Total eve charge,
#   Total night minutes, Total night calls, Total night charge,
#   Total intl minutes, Total intl calls, Total intl charge,
#   Customer service calls, Churn (True/False or yes/no)
# ─────────────────────────────────────────────────────────────────────────────

def load_orange_or_uci(filepath: str, combine_test: str = None) -> pd.DataFrame:
    """
    Map Orange Telecom / UCI / KDD dataset → ChurnShield schema.
    This is the most direct mapping — column names are very similar.

    Args:
        filepath:     Path to main CSV (churn-bigml-80.csv or bigml_59c28831336c6604c800002a_20160325_023102.csv)
        combine_test: Optional path to test CSV to combine with training data
    """
    print(f"Loading Orange/UCI dataset from: {filepath}")
    df = pd.read_csv(filepath)

    if combine_test:
        df_test = pd.read_csv(combine_test)
        df = pd.concat([df, df_test], ignore_index=True)
        print(f"  Combined with {combine_test}")

    print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")

    # Normalise column names — handle both capitalised and lowercase variants
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Column name mapping (handles both Orange and UCI naming)
    col_map = {
        "account_length":        "account_length_days",
        "area_code":             "area_code",
        "international_plan":    "international_plan_raw",
        "voice_mail_plan":       "voicemail_plan_raw",
        "number_vmail_messages": "voicemail_messages",
        "total_day_minutes":     "day_mins",
        "total_day_calls":       "day_calls",
        "total_day_charge":      "day_charge_aud",
        "total_eve_minutes":     "evening_mins",
        "total_eve_calls":       "evening_calls",
        "total_eve_charge":      "evening_charge_aud",
        "total_night_minutes":   "night_mins",
        "total_night_calls":     "night_calls",
        "total_night_charge":    "night_charge_aud",
        "total_intl_minutes":    "intl_mins",
        "total_intl_calls":      "intl_calls",
        "total_intl_charge":     "intl_charge_aud",
        "customer_service_calls":"customer_service_calls",
        "churn":                 "churn_raw",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Fix account length — UCI has it in days already; Orange in months
    # Heuristic: if max < 250 it's months, otherwise days
    if df["account_length_days"].max() < 250:
        df["account_length_days"] = (df["account_length_days"] * 30.44).round(0).astype(int)

    # Boolean plans
    df["international_plan"] = df["international_plan_raw"].astype(str).str.lower().isin(["yes", "true", "1"])
    df["voicemail_plan"]     = df["voicemail_plan_raw"].astype(str).str.lower().isin(["yes", "true", "1"])

    # Churn label
    df["churned"] = df["churn_raw"].astype(str).str.lower().isin(["true", "yes", "1"])

    # Replace US state codes with AU states (weighted random)
    df["state"]     = np.random.choice(AU_STATES, size=len(df), p=STATE_WEIGHTS)
    df["area_code"] = df["state"].map(AREA_CODES)

    # Customer IDs
    df["customer_id"] = [f"AU-{i+1:06d}" for i in range(len(df))]

    # Ensure numeric types
    numeric_cols = [
        "day_mins", "day_calls", "day_charge_aud",
        "evening_mins", "evening_calls", "evening_charge_aud",
        "night_mins", "night_calls", "night_charge_aud",
        "intl_mins", "intl_calls", "intl_charge_aud",
        "customer_service_calls", "voicemail_messages",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    out = df[[
        "customer_id", "state", "account_length_days", "area_code",
        "international_plan", "voicemail_plan", "voicemail_messages",
        "day_mins", "day_calls", "day_charge_aud",
        "evening_mins", "evening_calls", "evening_charge_aud",
        "night_mins", "night_calls", "night_charge_aud",
        "intl_mins", "intl_calls", "intl_charge_aud",
        "customer_service_calls", "churned",
    ]].copy()

    print(f"  Mapped {len(out)} customers | Churn rate: {out['churned'].mean():.1%}")
    return out


def validate_output(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the mapped data."""
    required = [
        "customer_id", "state", "account_length_days",
        "international_plan", "voicemail_plan", "voicemail_messages",
        "day_mins", "day_calls", "day_charge_aud",
        "evening_mins", "evening_calls", "evening_charge_aud",
        "night_mins", "night_calls", "night_charge_aud",
        "intl_mins", "intl_calls", "intl_charge_aud",
        "customer_service_calls", "churned",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    churn_rate = df["churned"].mean()
    assert 0.01 <= churn_rate <= 0.50, f"Churn rate {churn_rate:.1%} is outside plausible range"
    assert df["customer_id"].nunique() == len(df), "Duplicate customer IDs found"
    assert not df[required].isnull().any().any(), "Null values found in output"

    print(f"\n  ✅ Validation passed:")
    print(f"     Rows:          {len(df):,}")
    print(f"     Churn rate:    {churn_rate:.1%}")
    print(f"     States:        {sorted(df['state'].unique())}")
    print(f"     Avg tenure:    {df['account_length_days'].mean():.0f} days")
    print(f"     Avg charge:    ${(df['day_charge_aud']+df['evening_charge_aud']+df['night_charge_aud']+df['intl_charge_aud']).mean():.2f}/mo")


def save(df: pd.DataFrame) -> None:
    out_path = Path("data/raw/customers.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  💾 Saved to {out_path}")
    print(f"     Ready for training: python -m src.models.train")


def main():
    parser = argparse.ArgumentParser(
        description="Load a real churn dataset into ChurnShield format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_real_dataset.py --dataset ibm    --file WA_Fn-UseC_-Telco-Customer-Churn.csv
  python load_real_dataset.py --dataset orange --file churn-bigml-80.csv
  python load_real_dataset.py --dataset orange --file churn-bigml-80.csv --test churn-bigml-20.csv
  python load_real_dataset.py --dataset uci    --file telecom_churn.csv

Download links:
  IBM Telco:  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
  Orange:     https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
  UCI/KDD:    https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset
        """
    )
    parser.add_argument(
        "--dataset", required=True, choices=["ibm", "orange", "uci"],
        help="Which dataset format to load"
    )
    parser.add_argument(
        "--file", required=True,
        help="Path to the downloaded CSV file"
    )
    parser.add_argument(
        "--test", default=None,
        help="(Orange only) Path to test split CSV to combine with training"
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"\n❌ File not found: {args.file}")
        print("\nDownload the dataset first — see links above.")
        sys.exit(1)

    print(f"\n ChurnShield — Real Dataset Loader")
    print(f"{'─'*50}")

    if args.dataset == "ibm":
        df = load_ibm(args.file)
    elif args.dataset in ("orange", "uci"):
        df = load_orange_or_uci(args.file, combine_test=args.test)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    validate_output(df)
    save(df)

    print("\nNext steps:")
    print("  1. python -m src.models.train")
    print("  2. make serve       (FastAPI at localhost:8000/docs)")
    print("  3. make dashboard   (Streamlit at localhost:8501)")


if __name__ == "__main__":
    main()
