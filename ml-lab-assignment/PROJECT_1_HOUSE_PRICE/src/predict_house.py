
"""Small terminal-based house-price prediction interface."""
from pathlib import Path
import argparse
import joblib
import pandas as pd


def ask_number(label, default):
    """Ask for a number and use a clear example value when Enter is pressed."""
    raw = input(f"{label} [{default}]: ").strip()
    return float(raw) if raw else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/best_house_model.joblib")
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError("Saved model not found. Run house_price_project.py first.")
    model = joblib.load(model_path)

    print("\nEnter house details. Press Enter to use the example value shown.")
    row = {
        "OverallQual": ask_number("Overall quality (1-10)", 7),
        "GrLivArea": ask_number("Above-ground living area (sq ft)", 1500),
        "TotalBsmtSF": ask_number("Total basement area (sq ft)", 800),
        "YearBuilt": ask_number("Year built", 2000),
        "GarageCars": ask_number("Garage capacity (cars)", 2),
        "FullBath": ask_number("Full bathrooms", 2),
        "HalfBath": ask_number("Half bathrooms", 1),
        "LotArea": ask_number("Lot area (sq ft)", 8000),
        "Neighborhood": input("Neighborhood category [NAmes]: ").strip() or "NAmes",
        "KitchenQual": input("Kitchen quality (Ex/Gd/TA/Fa/Po) [TA]: ").strip() or "TA",
        "BsmtQual": input("Basement quality (Ex/Gd/TA/Fa/Po/NA) [TA]: ").strip() or "TA",
    }

    # The feature names are kept exactly as in training so the saved pipeline works.
    house = pd.DataFrame([row])
    predicted_price = float(model.predict(house)[0])
    print(f"\nEstimated sale price: {predicted_price:,.0f}")
    print("This is a model estimate, not a professional property valuation.")


if __name__ == "__main__":
    main()
