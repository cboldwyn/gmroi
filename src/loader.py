"""
Data loader for GMROI analysis.
Loads and combines all weekly sales and inventory snapshot CSVs.
"""

import pandas as pd
import glob
import os


def load_sales(data_dir="data/sales"):
    """Load all sales CSVs into a single DataFrame."""
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        df["_source_file"] = os.path.basename(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Parse date
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")

    # Drop rows with no date (header dupes from concat)
    combined = combined.dropna(subset=["Date"])

    # Numeric conversions for key fields
    numeric_cols = ["Quantity Sold", "Unit Cost", "Net Sales", "Retail Price",
                    "Effective Retail Price", "Gross Sales", "COGS"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    # Deduplicate: same transaction could appear in overlapping weekly exports
    if "Unique ID" in combined.columns and "Trans No" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=["Date", "Shop", "Trans No", "Unique ID", "Product ID"],
            keep="first"
        )
        dupes = before - len(combined)
        if dupes > 0:
            print(f"  Removed {dupes:,} duplicate transaction rows")

    print(f"Sales loaded: {len(combined):,} rows from {len(files)} files")
    print(f"  Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    print(f"  Shops: {combined['Shop'].nunique()}")
    print(f"  Unique products: {combined['Product'].nunique()}")

    return combined


def load_inventory(data_dir="data/inventory"):
    """Load all inventory snapshot CSVs into a single DataFrame."""
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        df["_source_file"] = os.path.basename(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Parse date
    combined["Date"] = pd.to_datetime(combined["Date"], format="%m/%d/%Y", errors="coerce")
    combined = combined.dropna(subset=["Date"])

    # Numeric conversions
    numeric_cols = ["Quantity on Hand", "Unit Cost", "Inventory Value", "Unit Price"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    # Keep only one snapshot day per week to reduce memory (~7x reduction).
    # Each weekly file contains 7 daily snapshots with near-identical data.
    yw = (combined["Date"].dt.isocalendar().year.astype(str)
          + "-" + combined["Date"].dt.isocalendar().week.astype(str))
    first_dates = combined.groupby(yw)["Date"].min().values
    combined = combined[combined["Date"].isin(first_dates)]

    print(f"Inventory loaded: {len(combined):,} rows from {len(files)} files")
    print(f"  Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    print(f"  Snapshot dates kept: {combined['Date'].nunique()} (1 per week)")
    print(f"  Shops: {combined['Shop'].nunique()}")
    print(f"  Unique products: {combined['Product Name'].nunique()}")

    return combined


if __name__ == "__main__":
    print("=" * 60)
    print("Loading sales data...")
    print("=" * 60)
    sales = load_sales()

    print()
    print("=" * 60)
    print("Loading inventory data...")
    print("=" * 60)
    inv = load_inventory()

    # Quick field inventory
    print()
    print("=" * 60)
    print("Sales columns:")
    print("=" * 60)
    for col in sales.columns:
        print(f"  {col}")

    print()
    print("=" * 60)
    print("Inventory columns:")
    print("=" * 60)
    for col in inv.columns:
        print(f"  {col}")

    # Sales summary
    print()
    print("=" * 60)
    print("Sales quick stats:")
    print("=" * 60)
    print(f"  Total Net Sales: ${sales['Net Sales'].sum():,.2f}")
    print(f"  Total COGS (Unit Cost * Qty): ${(sales['Unit Cost'] * sales['Quantity Sold']).sum():,.2f}")
    print(f"  Total Quantity Sold: {sales['Quantity Sold'].sum():,.0f}")
    print(f"  Unique Brands: {sales['Brand'].nunique()}")
    print(f"  Unique Categories: {sales['Product Category'].nunique()}")
    print(f"  Categories: {sorted(sales['Product Category'].unique())}")

    # Inventory summary
    print()
    print("=" * 60)
    print("Inventory quick stats:")
    print("=" * 60)
    print(f"  Total snapshot rows: {len(inv):,}")
    print(f"  Weekly snapshots: {inv['Date'].nunique()} unique dates")
    print(f"  Avg Inventory Value per snapshot: ${inv.groupby('Date')['Inventory Value'].sum().mean():,.2f}")
