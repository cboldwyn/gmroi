"""
Parquet persistence layer for GMROI Dashboard.

Provides fast data loading by caching processed CSV data as Parquet files.
Raw CSVs remain the source of truth; Parquet is the processed cache.
"""

import glob
import json
import os
from datetime import datetime

import pandas as pd


CACHE_DIR = os.path.join("data", "cache")
SALES_PARQUET = os.path.join(CACHE_DIR, "sales.parquet")
INVENTORY_PARQUET = os.path.join(CACHE_DIR, "inventory.parquet")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.json")

EXCLUDE_CATEGORIES = [
    "Display", "Sample", "Promo", "Compassion", "Donation", "Non-Cannabis",
    "Boxes"
]

BRAND_CONSOLIDATION = {
    "Camino Gummies": "Camino",
}

IGNORE_VENDOR_CREDITS_BRANDS = ['Stiiizy']

# Only load columns the app actually uses (113 -> ~15 for sales, 23 -> 8 for inventory).
# This cuts memory from ~10 GB to ~1 GB for a full-year dataset.
SALES_CSV_COLS = [
    "Date", "Shop", "Product", "Product Category", "Brand",
    "Trans No", "Trans Status", "Trans Type",
    "Quantity Sold", "Unit Cost", "Net Sales", "Effective Retail Price",
]

# Columns kept after build-time filtering (dropped: Trans Status, Trans Type
# -- only needed during build)
SALES_DROP_AFTER_BUILD = ["Trans Status", "Trans Type"]

INVENTORY_CSV_COLS = [
    "Date", "Shop", "Product Name", "Product Category", "Brand",
    "Quantity on Hand", "Unit Cost", "Inventory Value",
]


def build_from_csvs(sales_dir, inventory_dir, credits_pattern, version=""):
    """
    Process raw CSV files into deduplicated, credit-merged DataFrames
    and persist them as Parquet files.

    Returns (sales_df, inventory_df).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # -- Sales --
    sales_files = sorted(glob.glob(os.path.join(sales_dir, "*.csv")))
    if not sales_files:
        return None, None

    frames = []
    for f in sales_files:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False,
                         usecols=lambda c: c in SALES_CSV_COLS)
        frames.append(df)

    sales = pd.concat(frames, ignore_index=True)
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    sales = sales.dropna(subset=["Date"])

    numeric_cols = ["Quantity Sold", "Unit Cost", "Net Sales",
                    "Effective Retail Price"]
    for col in numeric_cols:
        if col in sales.columns:
            sales[col] = pd.to_numeric(sales[col], errors="coerce").fillna(0)

    # No dedup: each weekly CSV is a self-contained Blaze export with no
    # cross-file overlap.  Multi-unit purchases from the same METRC package
    # share (Trans No, Unique ID, Product ID) but are distinct line items.

    if "Trans Status" in sales.columns:
        sales = sales[sales["Trans Status"] == "Completed"]
    if "Trans Type" in sales.columns:
        sales = sales[sales["Trans Type"] == "Sale"]

    sales = sales[~sales["Product Category"].isin(EXCLUDE_CATEGORIES)]

    # Drop columns only needed for filtering/dedup
    sales = sales.drop(columns=[c for c in SALES_DROP_AFTER_BUILD
                                if c in sales.columns])

    # -- Inventory --
    inv_files = sorted(glob.glob(os.path.join(inventory_dir, "*.csv")))
    if not inv_files:
        return None, None

    frames = []
    for f in inv_files:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False,
                         usecols=lambda c: c in INVENTORY_CSV_COLS)
        frames.append(df)

    inventory = pd.concat(frames, ignore_index=True)
    inventory["Date"] = pd.to_datetime(inventory["Date"], format="%m/%d/%Y",
                                        errors="coerce")
    inventory = inventory.dropna(subset=["Date"])

    for col in ["Quantity on Hand", "Unit Cost", "Inventory Value"]:
        if col in inventory.columns:
            inventory[col] = pd.to_numeric(inventory[col],
                                            errors="coerce").fillna(0)

    inventory = inventory[~inventory["Product Category"].isin(EXCLUDE_CATEGORIES)]

    # Keep only one snapshot day per week (7x reduction)
    inventory["_yw"] = (inventory["Date"].dt.isocalendar().year.astype(str)
                        + "-" + inventory["Date"].dt.isocalendar().week.astype(str))
    first_dates = inventory.groupby("_yw")["Date"].min().values
    inventory = inventory[inventory["Date"].isin(first_dates)].drop(columns=["_yw"])

    # -- Vendor Credits --
    credits_files = sorted(glob.glob(credits_pattern))
    credits_df = None
    if credits_files:
        cframes = []
        for f in credits_files:
            cdf = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
            cframes.append(cdf)
        credits_df = pd.concat(cframes, ignore_index=True)

        if "Brand" in credits_df.columns and IGNORE_VENDOR_CREDITS_BRANDS:
            credits_df = credits_df[
                ~credits_df["Brand"].str.strip().isin(IGNORE_VENDOR_CREDITS_BRANDS)
            ]
        for col in ["Vendor Pays", "Haven Pays"]:
            if col in credits_df.columns:
                credits_df[col] = pd.to_numeric(credits_df[col],
                                                 errors="coerce").fillna(0)
        if "Trans No" in credits_df.columns:
            credits_df["Trans No"] = pd.to_numeric(credits_df["Trans No"],
                                                     errors="coerce")

    # -- Apply Credits --
    sales["Vendor_Pays"] = 0.0
    sales["Haven_Pays"] = 0.0

    if credits_df is not None and not credits_df.empty:
        credits_df = credits_df.copy()
        credits_df["_product_lower"] = (credits_df["Product"].astype(str)
                                         .str.lower().str.strip())
        credit_agg = credits_df.groupby(["Trans No", "_product_lower"]).agg(
            Vendor_Pays=("Vendor Pays", "sum"),
            Haven_Pays=("Haven Pays", "sum"),
        ).reset_index()

        sales["_product_lower"] = (sales["Product"].astype(str)
                                    .str.lower().str.strip())
        sales = sales.merge(
            credit_agg,
            left_on=["Trans No", "_product_lower"],
            right_on=["Trans No", "_product_lower"],
            how="left", suffixes=("", "_credit")
        )
        sales["Vendor_Pays"] = sales["Vendor_Pays_credit"].fillna(0)
        sales["Haven_Pays"] = sales["Haven_Pays_credit"].fillna(0)
        sales = sales.drop(columns=["Vendor_Pays_credit", "Haven_Pays_credit",
                                     "_product_lower"])

    # Consolidate brand variants
    if BRAND_CONSOLIDATION:
        sales["Brand"] = sales["Brand"].replace(BRAND_CONSOLIDATION)
        if "Brand" in inventory.columns:
            inventory["Brand"] = inventory["Brand"].replace(BRAND_CONSOLIDATION)

    # -- Brand cleansing --
    # Fix known bad brand values from Blaze (apostrophe stripping, case,
    # misattribution). Only correct specific known issues.
    brand_corrections = {
        # Apostrophe stripping by Blaze
        "Uncle Arnie s": "Uncle Arnie's",
        "Dr. Norm s": "Dr. Norm's",
        "Not Your Father s": "Not Your Father's",
        "Juicy Jay s": "Juicy Jay's",
        "Lil Buzzies": "Lil' Buzzies",
        # Case mismatches
        "STIIIZY": "Stiiizy",
        "West Coast cure": "West Coast Cure",
        "VUZE": "Vuze",
        "Wvy": "WVY",
        # Misattributed brands (bad Blaze data)
        "House Weed": None,       # derive from product name
        "House Party": None,
        "No Brand Found": None,
        "Default Brand": None,
        "Made": None,             # likely "Made From Dirt" truncated
    }
    # Apply static corrections
    static_fixes = {k: v for k, v in brand_corrections.items() if v is not None}
    corrected = sales["Brand"].isin(static_fixes).sum()
    sales["Brand"] = sales["Brand"].replace(static_fixes)

    # For None entries: derive brand from product name ("Brand - Description")
    # but only if the parsed brand is in the catalog.
    derive_brands = {k for k, v in brand_corrections.items() if v is None}
    catalog_path = os.path.join("data", "catalog", "profile_templates.csv")
    if derive_brands and os.path.exists(catalog_path):
        _cat = pd.read_csv(catalog_path, encoding="utf-8-sig")
        brand_canon = {b.lower(): b for b in _cat["Brand"].dropna().unique()}
        needs_fix = sales["Brand"].isin(derive_brands)
        if needs_fix.any():
            parsed = (sales.loc[needs_fix, "Product"].astype(str)
                      .str.split(" - ", n=1).str[0].str.strip())
            canonical = parsed.str.lower().map(brand_canon)
            fixable = canonical.notna()
            sales.loc[needs_fix & fixable.reindex(sales.index, fill_value=False),
                      "Brand"] = canonical[fixable].values
            corrected += fixable.sum()

    if corrected > 0:
        print(f"  Brand cleansing: {corrected:,} rows corrected")

    # Base COGS per line (vendor credits only, COGS adjustments applied post-cache)
    sales["COGS_Calc"] = (sales["Unit Cost"] * sales["Quantity Sold"]
                           - sales["Vendor_Pays"] + sales["Haven_Pays"])

    # -- Profile Template Matching --
    catalog_path = os.path.join("data", "catalog", "profile_templates.csv")
    if os.path.exists(catalog_path):
        from src.matcher import match_products_to_templates
        catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
        unique_products = sales[["Product", "Brand", "Product Category"]].drop_duplicates()
        template_map = match_products_to_templates(unique_products, catalog)
        sales["Profile_Template"] = [
            template_map.get((p, b), "Unmatched")
            for p, b in zip(sales["Product"], sales["Brand"])
        ]
        matched = (sales["Profile_Template"] != "Unmatched").sum()
        total = len(sales)
        print(f"  Profile Template matching: {matched:,}/{total:,} "
              f"({matched/total*100:.1f}%) matched")
    else:
        sales["Profile_Template"] = "Unmatched"

    # Convert string columns to categorical (cuts memory ~80%, critical for
    # Streamlit Cloud's 1 GB limit)
    for col in ["Shop", "Product", "Product Category", "Brand", "Profile_Template"]:
        if col in sales.columns:
            sales[col] = sales[col].astype("category")
    for col in ["Shop", "Product Name", "Product Category", "Brand"]:
        if col in inventory.columns:
            inventory[col] = inventory[col].astype("category")

    # -- Write Parquet --
    sales.to_parquet(SALES_PARQUET, index=False)
    inventory.to_parquet(INVENTORY_PARQUET, index=False)

    # -- Write Metadata --
    metadata = {
        "build_timestamp": datetime.now().isoformat(),
        "app_version": version,
        "sales_file_count": len(sales_files),
        "inventory_file_count": len(inv_files),
        "credits_file_count": len(credits_files),
        "sales_row_count": len(sales),
        "inventory_row_count": len(inventory),
        "sales_date_range": [
            str(sales["Date"].min().date()),
            str(sales["Date"].max().date()),
        ],
        "inventory_date_range": [
            str(inventory["Date"].min().date()),
            str(inventory["Date"].max().date()),
        ],
        "sales_files": [os.path.basename(f) for f in sales_files],
        "inventory_files": [os.path.basename(f) for f in inv_files],
        "credits_files": [os.path.basename(f) for f in credits_files],
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Add Month column for the returned DataFrames (not saved to Parquet)
    sales["Month"] = sales["Date"].dt.to_period("M")

    return sales, inventory


def load_from_parquet():
    """
    Load cached Parquet data if available.

    Returns (sales_df, inventory_df, metadata) or (None, None, None).
    """
    if not os.path.exists(SALES_PARQUET) or not os.path.exists(METADATA_PATH):
        return None, None, None

    sales = pd.read_parquet(SALES_PARQUET)
    inventory = pd.read_parquet(INVENTORY_PARQUET)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Recreate Month column (Period type doesn't serialize to Parquet)
    sales["Month"] = sales["Date"].dt.to_period("M")

    return sales, inventory, metadata


def load_metadata():
    """Load metadata dict from cache, or return None."""
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


def detect_new_files(sales_dir, inventory_dir, metadata):
    """
    Compare current CSV files against the metadata's processed file list.

    Returns (new_sales_files, new_inventory_files) as lists of filenames.
    """
    if metadata is None:
        return [], []

    current_sales = {os.path.basename(f)
                     for f in glob.glob(os.path.join(sales_dir, "*.csv"))}
    current_inv = {os.path.basename(f)
                   for f in glob.glob(os.path.join(inventory_dir, "*.csv"))}

    cached_sales = set(metadata.get("sales_files", []))
    cached_inv = set(metadata.get("inventory_files", []))

    new_sales = sorted(current_sales - cached_sales)
    new_inv = sorted(current_inv - cached_inv)

    return new_sales, new_inv
