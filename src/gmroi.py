"""
GMROI Analysis for Haven Cannabis
Calculates Gross Margin Return on Inventory Investment.

GMROI = Gross Margin $ / Average Inventory Cost (at cost)

Where:
- Gross Margin = Net Sales - COGS
- COGS = Unit Cost * Quantity Sold (from sales data)
- Average Inventory Cost = mean of weekly inventory value snapshots

Groupable by: Product, Brand, Category, and combinations thereof.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from loader import load_sales, load_inventory


def filter_completed_sales(sales):
    """Filter to completed sales only, exclude non-revenue transaction types."""
    # Only completed transactions
    if "Trans Status" in sales.columns:
        sales = sales[sales["Trans Status"] == "Completed"]

    # Only actual sales (not returns processed separately)
    if "Trans Type" in sales.columns:
        sales = sales[sales["Trans Type"].isin(["Sale"])]

    # Exclude non-sellable categories
    exclude_cats = ["Display", "Sample", "Promo", "Compassion", "Donation", "Non-Cannabis"]
    sales = sales[~sales["Product Category"].isin(exclude_cats)]

    return sales


def compute_sales_metrics(sales, group_cols):
    """Aggregate sales data by grouping columns."""
    sales = sales.copy()
    sales["COGS_Calc"] = sales["Unit Cost"] * sales["Quantity Sold"]

    agg = sales.groupby(group_cols, dropna=False).agg(
        Net_Sales=("Net Sales", "sum"),
        COGS=("COGS_Calc", "sum"),
        Qty_Sold=("Quantity Sold", "sum"),
        Transactions=("Trans No", "nunique"),
        Avg_Unit_Cost=("Unit Cost", "mean"),
        Avg_Sell_Price=("Effective Retail Price", "mean"),
    ).reset_index()

    agg["Gross_Margin"] = agg["Net_Sales"] - agg["COGS"]
    agg["Margin_Pct"] = np.where(
        agg["Net_Sales"] != 0,
        agg["Gross_Margin"] / agg["Net_Sales"] * 100,
        0
    )

    return agg


def compute_avg_inventory(inventory, group_cols):
    """
    Compute average inventory cost from weekly snapshots.

    The inventory data has daily dates within weekly snapshot files.
    We group by week to avoid overweighting weeks with more snapshot days.
    """
    inv = inventory.copy()

    # Exclude non-sellable categories to match sales filter
    exclude_cats = ["Display", "Sample", "Promo", "Compassion", "Donation", "Non-Cannabis"]
    inv = inv[~inv["Product Category"].isin(exclude_cats)]

    # Map inventory column names to match sales grouping
    col_map = {"Product Name": "Product", "Product Category": "Product Category"}
    for inv_col, sales_col in col_map.items():
        if inv_col in inv.columns and sales_col in group_cols and inv_col != sales_col:
            inv = inv.rename(columns={inv_col: sales_col})

    # Create week period for averaging
    inv["Week"] = inv["Date"].dt.to_period("W")

    # First: sum inventory value per group per week
    weekly = inv.groupby(group_cols + ["Week"], dropna=False).agg(
        Weekly_Inv_Value=("Inventory Value", "sum"),
        Weekly_Qty_On_Hand=("Quantity on Hand", "sum"),
    ).reset_index()

    # Then: average across weeks
    avg_inv = weekly.groupby(group_cols, dropna=False).agg(
        Avg_Inventory_Cost=("Weekly_Inv_Value", "mean"),
        Avg_Qty_On_Hand=("Weekly_Qty_On_Hand", "mean"),
    ).reset_index()

    return avg_inv


def compute_gmroi(sales, inventory, group_cols):
    """
    Compute GMROI for given grouping.

    GMROI = Gross Margin / Average Inventory Cost
    """
    print(f"\nComputing GMROI grouped by: {group_cols}")

    sales_filtered = filter_completed_sales(sales)
    print(f"  Sales after filtering: {len(sales_filtered):,} rows")

    sales_agg = compute_sales_metrics(sales_filtered, group_cols)
    print(f"  Sales groups: {len(sales_agg):,}")

    inv_agg = compute_avg_inventory(inventory, group_cols)
    print(f"  Inventory groups: {len(inv_agg):,}")

    # Merge
    merged = sales_agg.merge(inv_agg, on=group_cols, how="left")

    # GMROI calculation
    merged["GMROI"] = np.where(
        merged["Avg_Inventory_Cost"] > 0,
        merged["Gross_Margin"] / merged["Avg_Inventory_Cost"],
        np.nan
    )

    # Inventory turnover (for reference)
    merged["Inventory_Turns"] = np.where(
        merged["Avg_Inventory_Cost"] > 0,
        merged["COGS"] / merged["Avg_Inventory_Cost"],
        np.nan
    )

    # Sort by GMROI descending
    merged = merged.sort_values("GMROI", ascending=False)

    return merged


def print_summary(df, title, top_n=25):
    """Print a formatted summary table."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")

    # Network totals
    total_sales = df["Net_Sales"].sum()
    total_cogs = df["COGS"].sum()
    total_gm = df["Gross_Margin"].sum()
    total_avg_inv = df["Avg_Inventory_Cost"].sum()
    network_gmroi = total_gm / total_avg_inv if total_avg_inv > 0 else 0
    network_turns = total_cogs / total_avg_inv if total_avg_inv > 0 else 0
    network_margin = total_gm / total_sales * 100 if total_sales > 0 else 0

    print(f"\n  NETWORK TOTALS:")
    print(f"  Net Sales:           ${total_sales:>14,.2f}")
    print(f"  COGS:                ${total_cogs:>14,.2f}")
    print(f"  Gross Margin:        ${total_gm:>14,.2f}")
    print(f"  Margin %:            {network_margin:>14.1f}%")
    print(f"  Avg Inventory Cost:  ${total_avg_inv:>14,.2f}")
    print(f"  GMROI:               {network_gmroi:>14.2f}")
    print(f"  Inventory Turns:     {network_turns:>14.2f}")

    # Top performers
    valid = df[df["GMROI"].notna() & (df["Net_Sales"] >= 1000)].head(top_n)
    if len(valid) > 0:
        print(f"\n  TOP {top_n} by GMROI (min $1,000 net sales):")
        print(f"  {'-' * 76}")

        # Determine display columns based on grouping
        group_cols = [c for c in valid.columns if c not in [
            "Net_Sales", "COGS", "Qty_Sold", "Transactions", "Avg_Unit_Cost",
            "Avg_Sell_Price", "Gross_Margin", "Margin_Pct", "Avg_Inventory_Cost",
            "Avg_Qty_On_Hand", "GMROI", "Inventory_Turns"
        ]]

        for _, row in valid.iterrows():
            label = " | ".join(str(row[c]) for c in group_cols)
            if len(label) > 40:
                label = label[:37] + "..."
            print(f"  {label:<40s}  GMROI: {row['GMROI']:>7.2f}  "
                  f"Sales: ${row['Net_Sales']:>12,.0f}  "
                  f"Margin: {row['Margin_Pct']:>5.1f}%  "
                  f"Turns: {row['Inventory_Turns']:>5.1f}")


def export_results(df, filename, output_dir="output"):
    """Export results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"\n  Exported: {path} ({len(df):,} rows)")


def main():
    print("=" * 80)
    print("HAVEN GMROI ANALYSIS - 2025")
    print("=" * 80)

    # Load data
    sales = load_sales()
    inventory = load_inventory()

    # -- By Brand --
    brand = compute_gmroi(sales, inventory, ["Brand"])
    print_summary(brand, "GMROI BY BRAND")
    export_results(brand, "gmroi_by_brand.csv")

    # -- By Category --
    category = compute_gmroi(sales, inventory, ["Product Category"])
    print_summary(category, "GMROI BY CATEGORY")
    export_results(category, "gmroi_by_category.csv")

    # -- By Brand + Category --
    brand_cat = compute_gmroi(sales, inventory, ["Brand", "Product Category"])
    print_summary(brand_cat, "GMROI BY BRAND + CATEGORY", top_n=30)
    export_results(brand_cat, "gmroi_by_brand_category.csv")

    # -- By Product --
    product = compute_gmroi(sales, inventory, ["Product", "Brand", "Product Category"])
    print_summary(product, "GMROI BY PRODUCT", top_n=30)
    export_results(product, "gmroi_by_product.csv")

    # -- By Shop --
    shop = compute_gmroi(sales, inventory, ["Shop"])
    print_summary(shop, "GMROI BY SHOP")
    export_results(shop, "gmroi_by_shop.csv")

    print("\n" + "=" * 80)
    print("DONE. CSVs exported to output/")
    print("=" * 80)


if __name__ == "__main__":
    main()
