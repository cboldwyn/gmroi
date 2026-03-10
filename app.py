"""
GMROI Dashboard v2.0.0
Gross Margin Return on Inventory Investment analysis for Haven Cannabis

Provides interactive analysis of product performance across Brand, Category,
Product, and Shop dimensions with vendor credit integration, monthly trends,
margin-turnover portfolio analysis, and store-level variance detection.

CHANGELOG:
v2.1.0 (2026-03-10)
- Flower combine/separate toggle in sidebar (Indica/Sativa/Hybrid)
- Added requirements.txt

v2.0.0 (2026-03-10)
- NEW: Trends tab - Monthly GMROI by category with rolling 3-month average
- NEW: Portfolio tab - Margin vs Turns scatter with GMROI iso-lines and
  quadrant classification (Stars/Sleepers/Traffic Drivers/Dogs)
- NEW: Store Variance tab - Flags brands/products with high GMROI variance
  across stores to support store-specific assortment decisions

v1.3.0 (2026-03-10)
- All COGS includes vendor credits (True COGS)
- Fixed sorting with Streamlit column_config

v1.2.0 - Filters in main content area (Price Inspector pattern)
v1.1.0 - Flower consolidation, Boxes excluded
v1.0.0 - Initial release

Author: Haven Supply Chain
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import io

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "2.1.0"

st.set_page_config(
    page_title=f"GMROI Dashboard v{VERSION}",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = "data"
SALES_DIR = os.path.join(DATA_DIR, "sales")
INVENTORY_DIR = os.path.join(DATA_DIR, "inventory")
CREDITS_PATH = os.path.join(DATA_DIR, "2025 Vendor Credits.csv")

EXCLUDE_CATEGORIES = [
    "Display", "Sample", "Promo", "Compassion", "Donation", "Non-Cannabis",
    "Boxes"
]

FLOWER_CATEGORIES = ["Indica", "Sativa", "Hybrid"]

# Portfolio quadrant thresholds: set dynamically from median of data
# GMROI iso-line values to draw on scatter plot
GMROI_ISO_VALUES = [1.0, 2.0, 3.0, 5.0]


# ============================================================================
# DATA LOADING & PREPARATION (one-time, cached)
# ============================================================================

@st.cache_data(show_spinner="Loading and preparing data...")
def load_and_prepare():
    """Load sales, inventory, credits. Clean, merge, return ready DataFrames."""
    # -- Sales --
    sales_files = sorted(glob.glob(os.path.join(SALES_DIR, "*.csv")))
    if not sales_files:
        return None, None

    frames = []
    for f in sales_files:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        frames.append(df)

    sales = pd.concat(frames, ignore_index=True)
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    sales = sales.dropna(subset=["Date"])

    numeric_cols = ["Quantity Sold", "Unit Cost", "Net Sales",
                    "Effective Retail Price", "Gross Sales"]
    for col in numeric_cols:
        if col in sales.columns:
            sales[col] = pd.to_numeric(sales[col], errors="coerce").fillna(0)

    if "Unique ID" in sales.columns and "Trans No" in sales.columns:
        sales = sales.drop_duplicates(
            subset=["Date", "Shop", "Trans No", "Unique ID", "Product ID"],
            keep="first"
        )

    if "Trans Status" in sales.columns:
        sales = sales[sales["Trans Status"] == "Completed"]
    if "Trans Type" in sales.columns:
        sales = sales[sales["Trans Type"] == "Sale"]

    sales = sales[~sales["Product Category"].isin(EXCLUDE_CATEGORIES)]

    # -- Inventory --
    inv_files = sorted(glob.glob(os.path.join(INVENTORY_DIR, "*.csv")))
    if not inv_files:
        return None, None

    frames = []
    for f in inv_files:
        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
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

    # -- Vendor Credits --
    credits_df = None
    if os.path.exists(CREDITS_PATH):
        credits_df = pd.read_csv(CREDITS_PATH, encoding="utf-8-sig",
                                  low_memory=False)
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

    # True COGS per line
    sales["COGS_Calc"] = (sales["Unit Cost"] * sales["Quantity Sold"]
                           - sales["Vendor_Pays"] + sales["Haven_Pays"])

    # Month column for trend analysis
    sales["Month"] = sales["Date"].dt.to_period("M")

    return sales, inventory


# ============================================================================
# COMPUTATION
# ============================================================================

def compute_gmroi(sales, inventory, group_cols):
    """Compute GMROI for given grouping columns. COGS includes vendor credits."""
    if len(sales) == 0:
        return pd.DataFrame()

    sales_agg = sales.groupby(group_cols, dropna=False).agg(
        Net_Sales=("Net Sales", "sum"),
        COGS=("COGS_Calc", "sum"),
        Qty_Sold=("Quantity Sold", "sum"),
        Transactions=("Trans No", "nunique"),
        Avg_Sell_Price=("Effective Retail Price", "mean"),
        Avg_Unit_Cost=("Unit Cost", "mean"),
        Vendor_Credits=("Vendor_Pays", "sum"),
    ).reset_index()

    sales_agg["Gross_Margin"] = sales_agg["Net_Sales"] - sales_agg["COGS"]
    sales_agg["Margin_Pct"] = np.where(
        sales_agg["Net_Sales"] != 0,
        sales_agg["Gross_Margin"] / sales_agg["Net_Sales"] * 100, 0
    )

    # Inventory aggregation
    inv = inventory.copy()
    col_map = {"Product Name": "Product"}
    for inv_col, sales_col in col_map.items():
        if inv_col in inv.columns and sales_col in group_cols:
            inv = inv.rename(columns={inv_col: sales_col})

    inv["Week"] = inv["Date"].dt.to_period("W")
    weekly = inv.groupby(group_cols + ["Week"], dropna=False).agg(
        Weekly_Inv_Value=("Inventory Value", "sum"),
        Weekly_Qty=("Quantity on Hand", "sum"),
    ).reset_index()

    avg_inv = weekly.groupby(group_cols, dropna=False).agg(
        Avg_Inv_Cost=("Weekly_Inv_Value", "mean"),
        Avg_Qty_On_Hand=("Weekly_Qty", "mean"),
    ).reset_index()

    merged = sales_agg.merge(avg_inv, on=group_cols, how="left")

    merged["GMROI"] = np.where(
        merged["Avg_Inv_Cost"] > 0,
        merged["Gross_Margin"] / merged["Avg_Inv_Cost"], np.nan
    )
    merged["Inv_Turns"] = np.where(
        merged["Avg_Inv_Cost"] > 0,
        merged["COGS"] / merged["Avg_Inv_Cost"], np.nan
    )

    return merged.sort_values("GMROI", ascending=False)


def compute_monthly_gmroi(sales, inventory, group_col):
    """
    Compute GMROI by month for a given grouping dimension.
    Returns a DataFrame with Month, group_col, and GMROI.
    """
    if len(sales) == 0:
        return pd.DataFrame()

    # Monthly sales aggregation
    monthly_sales = sales.groupby([group_col, "Month"], dropna=False).agg(
        Net_Sales=("Net Sales", "sum"),
        COGS=("COGS_Calc", "sum"),
    ).reset_index()
    monthly_sales["Gross_Margin"] = monthly_sales["Net_Sales"] - monthly_sales["COGS"]

    # Monthly inventory: avg within each month's weeks
    inv = inventory.copy()
    col_map = {"Product Name": "Product"}
    for inv_col, sales_col in col_map.items():
        if inv_col in inv.columns and sales_col == group_col:
            inv = inv.rename(columns={inv_col: sales_col})

    inv["Month"] = inv["Date"].dt.to_period("M")
    monthly_inv = inv.groupby([group_col, "Month"], dropna=False).agg(
        Avg_Inv_Cost=("Inventory Value", "mean"),
    ).reset_index()

    merged = monthly_sales.merge(monthly_inv, on=[group_col, "Month"], how="left")
    merged["GMROI"] = np.where(
        merged["Avg_Inv_Cost"] > 0,
        merged["Gross_Margin"] / merged["Avg_Inv_Cost"], np.nan
    )

    # Convert Period to string for plotting
    merged["Month_str"] = merged["Month"].astype(str)

    return merged


def compute_store_variance(sales, inventory):
    """
    Compute GMROI by Brand x Shop and by Product x Shop.
    Returns DataFrames with variance metrics to identify assortment opportunities.
    """
    if len(sales) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Brand x Shop
    brand_shop = compute_gmroi(sales, inventory, ["Brand", "Shop"])

    # For each brand, compute stats across shops
    if not brand_shop.empty:
        brand_stats = brand_shop.groupby("Brand").agg(
            Shop_Count=("Shop", "nunique"),
            GMROI_Mean=("GMROI", "mean"),
            GMROI_Median=("GMROI", "median"),
            GMROI_Std=("GMROI", "std"),
            GMROI_Min=("GMROI", "min"),
            GMROI_Max=("GMROI", "max"),
            Total_Sales=("Net_Sales", "sum"),
        ).reset_index()
        brand_stats["GMROI_Range"] = brand_stats["GMROI_Max"] - brand_stats["GMROI_Min"]
        brand_stats["GMROI_CV"] = np.where(
            brand_stats["GMROI_Mean"] > 0,
            brand_stats["GMROI_Std"] / brand_stats["GMROI_Mean"] * 100, np.nan
        )
        brand_stats = brand_stats.sort_values("GMROI_CV", ascending=False)
    else:
        brand_stats = pd.DataFrame()

    return brand_shop, brand_stats


# ============================================================================
# FILTER HELPERS
# ============================================================================

def apply_filter(series, mode, selected):
    """Apply include/exclude filter. Empty selection = show all."""
    if not selected:
        return pd.Series(True, index=series.index)
    if mode == "Include":
        return series.isin(selected)
    else:
        return ~series.isin(selected)


def render_filters(sales, inventory, prefix):
    """Render 3-column filter controls. Returns filtered (sales, inventory)."""
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        cat_mode = st.radio("Category Filter Mode:",
                             ["Include", "Exclude"], horizontal=True,
                             key=f"{prefix}_cat_mode")
        all_cats = sorted(sales["Product Category"].dropna().unique())
        sel_cats = st.multiselect(
            f"{'Include' if cat_mode == 'Include' else 'Exclude'} Categories:",
            all_cats, default=[], key=f"{prefix}_cats")

    with col2:
        brand_mode = st.radio("Brand Filter Mode:",
                               ["Include", "Exclude"], horizontal=True,
                               key=f"{prefix}_brand_mode")
        all_brands = sorted(sales["Brand"].dropna().unique())
        sel_brands = st.multiselect(
            f"{'Include' if brand_mode == 'Include' else 'Exclude'} Brands:",
            all_brands, default=[], key=f"{prefix}_brands")

    with col3:
        shop_mode = st.radio("Shop Filter Mode:",
                              ["Include", "Exclude"], horizontal=True,
                              key=f"{prefix}_shop_mode")
        all_shops = sorted(sales["Shop"].dropna().unique())
        sel_shops = st.multiselect(
            f"{'Include' if shop_mode == 'Include' else 'Exclude'} Shops:",
            all_shops, default=[], key=f"{prefix}_shops")

    s_mask = (apply_filter(sales["Product Category"], cat_mode, sel_cats)
              & apply_filter(sales["Brand"], brand_mode, sel_brands)
              & apply_filter(sales["Shop"], shop_mode, sel_shops))

    i_mask = apply_filter(inventory["Product Category"], cat_mode, sel_cats)
    if "Brand" in inventory.columns:
        i_mask &= apply_filter(inventory["Brand"], brand_mode, sel_brands)
    i_mask &= apply_filter(inventory["Shop"], shop_mode, sel_shops)

    return sales[s_mask], inventory[i_mask]


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

COLUMN_CONFIG = {
    "Net_Sales": st.column_config.NumberColumn("Net Sales", format="$%,.0f"),
    "COGS": st.column_config.NumberColumn("COGS", format="$%,.0f"),
    "Gross_Margin": st.column_config.NumberColumn("Gross Margin", format="$%,.0f"),
    "Margin_Pct": st.column_config.NumberColumn("Margin %", format="%.1f%%"),
    "GMROI": st.column_config.NumberColumn("GMROI", format="%.2f"),
    "Inv_Turns": st.column_config.NumberColumn("Inv Turns", format="%.2f"),
    "Qty_Sold": st.column_config.NumberColumn("Qty Sold", format="%,.0f"),
    "Avg_Sell_Price": st.column_config.NumberColumn("Avg Sell Price", format="$%,.2f"),
    "Avg_Unit_Cost": st.column_config.NumberColumn("Avg Unit Cost", format="$%,.2f"),
    "Avg_Inv_Cost": st.column_config.NumberColumn("Avg Inv Cost", format="$%,.0f"),
    "Avg_Qty_On_Hand": st.column_config.NumberColumn("Avg Qty On Hand", format="%,.0f"),
    "Vendor_Credits": st.column_config.NumberColumn("Vendor Credits", format="$%,.0f"),
    "Transactions": st.column_config.NumberColumn("Transactions", format="%,.0f"),
    "Total_Sales": st.column_config.NumberColumn("Total Sales", format="$%,.0f"),
    "Shop_Count": st.column_config.NumberColumn("Shops", format="%d"),
    "GMROI_Mean": st.column_config.NumberColumn("GMROI Avg", format="%.2f"),
    "GMROI_Median": st.column_config.NumberColumn("GMROI Median", format="%.2f"),
    "GMROI_Std": st.column_config.NumberColumn("GMROI Std Dev", format="%.2f"),
    "GMROI_Min": st.column_config.NumberColumn("GMROI Min", format="%.2f"),
    "GMROI_Max": st.column_config.NumberColumn("GMROI Max", format="%.2f"),
    "GMROI_Range": st.column_config.NumberColumn("GMROI Range", format="%.2f"),
    "GMROI_CV": st.column_config.NumberColumn("CV %", format="%.1f%%"),
}


def show_table(df, display_cols, download_name):
    """Display a sortable dataframe with numeric formatting + download."""
    available = [c for c in display_cols if c in df.columns]
    show_df = df[available].copy()

    config = {}
    for col in available:
        if col in COLUMN_CONFIG:
            config[col] = COLUMN_CONFIG[col]

    st.dataframe(show_df, use_container_width=True, hide_index=True,
                  column_config=config)

    csv_buf = io.StringIO()
    df[available].to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_buf.getvalue(),
        file_name=download_name,
        mime="text/csv",
        key=f"dl_{download_name}"
    )


def network_metrics(df):
    """Display network-level summary metrics."""
    if df.empty:
        st.info("No data matches current filters.")
        return

    total_sales = df["Net_Sales"].sum()
    total_gm = df["Gross_Margin"].sum()
    total_cogs = df["COGS"].sum()
    total_avg_inv = df["Avg_Inv_Cost"].sum()
    gmroi = total_gm / total_avg_inv if total_avg_inv > 0 else 0
    margin_pct = total_gm / total_sales * 100 if total_sales > 0 else 0
    turns = total_cogs / total_avg_inv if total_avg_inv > 0 else 0
    credits = df["Vendor_Credits"].sum()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Net Sales", f"${total_sales:,.0f}")
    with col2:
        st.metric("Gross Margin", f"${total_gm:,.0f}")
    with col3:
        st.metric("Margin %", f"{margin_pct:.1f}%")
    with col4:
        st.metric("GMROI", f"{gmroi:.2f}")
    with col5:
        st.metric("Inv Turns", f"{turns:.2f}")
    with col6:
        st.metric("Vendor Credits", f"${credits:,.0f}")


# ============================================================================
# CHART BUILDERS
# ============================================================================

def build_trend_chart(monthly_data, group_col):
    """Build monthly GMROI trend line chart with rolling 3-month average."""
    if monthly_data.empty:
        return None

    # Only keep groups with meaningful data
    valid = monthly_data.dropna(subset=["GMROI"])
    if valid.empty:
        return None

    fig = px.line(
        valid, x="Month_str", y="GMROI", color=group_col,
        markers=True,
        labels={"Month_str": "Month", "GMROI": "GMROI"},
    )

    # Add rolling 3-month average as dashed lines
    for name in valid[group_col].unique():
        subset = valid[valid[group_col] == name].sort_values("Month_str")
        if len(subset) >= 3:
            rolling = subset["GMROI"].rolling(3, min_periods=3).mean()
            fig.add_trace(go.Scatter(
                x=subset["Month_str"],
                y=rolling,
                mode="lines",
                line=dict(dash="dash", width=1),
                name=f"{name} (3mo avg)",
                showlegend=False,
                opacity=0.5,
            ))

    # Reference lines
    fig.add_hline(y=1.0, line_dash="dot", line_color="red",
                   annotation_text="Break-even (1.0)")
    fig.add_hline(y=3.0, line_dash="dot", line_color="green",
                   annotation_text="Strong (3.0)")

    fig.update_layout(
        height=500,
        xaxis_title="Month",
        yaxis_title="GMROI",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        hovermode="x unified",
    )

    return fig


def build_scatter_chart(data, group_col):
    """
    Build Margin % vs Inventory Turns scatter plot with GMROI iso-lines.
    Each dot = a group (brand, category, etc.), sized by Net Sales.
    """
    if data.empty:
        return None

    plot_data = data.dropna(subset=["Margin_Pct", "Inv_Turns", "GMROI"]).copy()
    plot_data = plot_data[plot_data["Inv_Turns"] > 0]
    if plot_data.empty:
        return None

    # Quadrant thresholds from medians
    med_margin = plot_data["Margin_Pct"].median()
    med_turns = plot_data["Inv_Turns"].median()

    # Classify into quadrants
    def classify(row):
        high_m = row["Margin_Pct"] >= med_margin
        high_t = row["Inv_Turns"] >= med_turns
        if high_m and high_t:
            return "Star"
        elif high_m and not high_t:
            return "Sleeper"
        elif not high_m and high_t:
            return "Traffic Driver"
        else:
            return "Dog"

    plot_data["Quadrant"] = plot_data.apply(classify, axis=1)

    color_map = {
        "Star": "#2ecc71",
        "Sleeper": "#f39c12",
        "Traffic Driver": "#3498db",
        "Dog": "#e74c3c",
    }

    fig = px.scatter(
        plot_data,
        x="Inv_Turns", y="Margin_Pct",
        size="Net_Sales", size_max=50,
        color="Quadrant", color_discrete_map=color_map,
        hover_name=group_col,
        hover_data={
            "GMROI": ":.2f",
            "Net_Sales": ":$,.0f",
            "Margin_Pct": ":.1f",
            "Inv_Turns": ":.2f",
            "Quadrant": True,
        },
        labels={
            "Inv_Turns": "Inventory Turns",
            "Margin_Pct": "Gross Margin %",
        },
    )

    # GMROI iso-lines: GMROI = (Margin% / 100) * Turns
    # So Margin% = GMROI * 100 / Turns
    max_turns = min(plot_data["Inv_Turns"].max() * 1.2, 20)
    turns_range = np.linspace(0.1, max_turns, 200)

    for iso_val in GMROI_ISO_VALUES:
        margin_line = iso_val / turns_range * 100
        # Only show where margin is reasonable (0-100)
        mask = margin_line <= 100
        fig.add_trace(go.Scatter(
            x=turns_range[mask], y=margin_line[mask],
            mode="lines",
            line=dict(dash="dot", width=1, color="gray"),
            name=f"GMROI = {iso_val}",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Quadrant lines
    fig.add_hline(y=med_margin, line_dash="dash", line_color="gray",
                   opacity=0.4)
    fig.add_vline(x=med_turns, line_dash="dash", line_color="gray",
                   opacity=0.4)

    # Quadrant labels
    y_max = min(plot_data["Margin_Pct"].max() * 1.1, 100)
    fig.add_annotation(x=max_turns * 0.85, y=y_max * 0.95,
                        text="STARS", showarrow=False,
                        font=dict(size=14, color="#2ecc71"), opacity=0.5)
    fig.add_annotation(x=max_turns * 0.15, y=y_max * 0.95,
                        text="SLEEPERS", showarrow=False,
                        font=dict(size=14, color="#f39c12"), opacity=0.5)
    fig.add_annotation(x=max_turns * 0.85, y=med_margin * 0.3,
                        text="TRAFFIC DRIVERS", showarrow=False,
                        font=dict(size=14, color="#3498db"), opacity=0.5)
    fig.add_annotation(x=max_turns * 0.15, y=med_margin * 0.3,
                        text="DOGS", showarrow=False,
                        font=dict(size=14, color="#e74c3c"), opacity=0.5)

    fig.update_layout(
        height=600,
        xaxis_title="Inventory Turns (annualized)",
        yaxis_title="Gross Margin %",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )

    return fig


def build_variance_chart(brand_shop_data, brand_name):
    """Build a bar chart showing GMROI by shop for a specific brand."""
    subset = brand_shop_data[brand_shop_data["Brand"] == brand_name].copy()
    if subset.empty:
        return None

    subset = subset.sort_values("GMROI", ascending=True)

    fig = px.bar(
        subset, x="GMROI", y="Shop", orientation="h",
        color="GMROI",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        labels={"GMROI": "GMROI", "Shop": ""},
        hover_data={"Net_Sales": ":$,.0f", "Margin_Pct": ":.1f%%",
                     "Inv_Turns": ":.2f"},
    )

    fig.add_vline(x=1.0, line_dash="dot", line_color="red")
    fig.update_layout(
        height=max(300, len(subset) * 35),
        showlegend=False,
        coloraxis_showscale=False,
    )

    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

METRIC_COLS = ["Net_Sales", "COGS", "Gross_Margin", "Margin_Pct",
               "GMROI", "Inv_Turns", "Qty_Sold", "Transactions",
               "Avg_Sell_Price", "Avg_Unit_Cost", "Avg_Inv_Cost",
               "Avg_Qty_On_Hand", "Vendor_Credits"]


def main():
    st.title(f"📊 GMROI Dashboard v{VERSION}")
    st.markdown("Gross Margin Return on Inventory Investment - Haven 2025")

    # ── Load Data ──
    sales, inventory = load_and_prepare()

    if sales is None or inventory is None:
        st.error("Could not load data. Check that data/sales/ and "
                 "data/inventory/ contain CSV files.")
        return

    # ── Sidebar ──
    st.sidebar.header("📊 Settings")

    # Flower consolidation toggle
    combine_flower = st.sidebar.toggle(
        "Combine Flower categories",
        value=True,
        help="When ON, Indica/Sativa/Hybrid are grouped as 'Flower'. "
             "When OFF, they appear as separate categories."
    )

    if combine_flower:
        flower_map = {cat: "Flower" for cat in FLOWER_CATEGORIES}
        sales = sales.copy()
        sales["Product Category"] = sales["Product Category"].replace(flower_map)
        inventory = inventory.copy()
        inventory["Product Category"] = inventory["Product Category"].replace(
            flower_map)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Data Summary")
    st.sidebar.markdown(f"**Sales rows:** {len(sales):,}")
    st.sidebar.markdown(f"**Net Sales:** ${sales['Net Sales'].sum():,.0f}")
    credit_total = sales["Vendor_Pays"].sum()
    if credit_total > 0:
        st.sidebar.markdown(f"**Vendor Credits:** ${credit_total:,.0f}")
    st.sidebar.markdown(f"**Categories:** {sales['Product Category'].nunique()}")
    st.sidebar.markdown(f"**Brands:** {sales['Brand'].nunique()}")
    st.sidebar.markdown(f"**Shops:** {sales['Shop'].nunique()}")
    st.sidebar.markdown(f"**Date range:** {sales['Date'].min().date()} to "
                         f"{sales['Date'].max().date()}")
    st.sidebar.markdown("---")
    with st.sidebar.expander("📋 Version History"):
        st.markdown(f"""
**v{VERSION}** (2026-03-10)
- Flower combine/separate toggle
- Trends, Portfolio, Store Variance tabs
- True COGS everywhere, sortable columns
- Filters in main content area
        """)
    st.sidebar.markdown(f"**Version {VERSION}**")

    # ── Tabs ──
    tabs = st.tabs([
        "📊 By Category",
        "🏷️ By Brand",
        "📦 By Product",
        "🏪 By Shop",
        "📈 Trends",
        "🎯 Portfolio",
        "🔀 Store Variance",
    ])

    # ── TAB 1: By Category ──
    with tabs[0]:
        st.subheader("GMROI by Category")
        fsales, finv = render_filters(sales, inventory, "cat_tab")
        st.markdown("---")

        cat_data = compute_gmroi(fsales, finv, ["Product Category"])
        network_metrics(cat_data)

        if not cat_data.empty:
            st.markdown("---")
            show_table(cat_data, ["Product Category"] + METRIC_COLS,
                        "gmroi_by_category.csv")

            st.markdown("---")
            st.subheader("Drill Down: Brands within Category")
            cat_options = cat_data["Product Category"].tolist()
            selected_cat = st.selectbox("Select category:", cat_options,
                                         key="cat_drill")
            cat_s = fsales[fsales["Product Category"] == selected_cat]
            cat_i = finv[finv["Product Category"] == selected_cat]
            if len(cat_s) > 0:
                drill = compute_gmroi(cat_s, cat_i, ["Brand"])
                show_table(drill, ["Brand"] + METRIC_COLS,
                            f"gmroi_{selected_cat}_brands.csv")

    # ── TAB 2: By Brand ──
    with tabs[1]:
        st.subheader("GMROI by Brand")
        fsales, finv = render_filters(sales, inventory, "brand_tab")
        st.markdown("---")

        brand_data = compute_gmroi(fsales, finv, ["Brand"])
        network_metrics(brand_data)

        if not brand_data.empty:
            st.markdown("---")
            show_table(brand_data, ["Brand"] + METRIC_COLS,
                        "gmroi_by_brand.csv")

            st.markdown("---")
            st.subheader("Drill Down: Products within Brand")
            brand_options = brand_data["Brand"].tolist()
            selected_brand = st.selectbox("Select brand:", brand_options,
                                           key="brand_drill")
            b_s = fsales[fsales["Brand"] == selected_brand]
            b_i = finv
            if "Brand" in finv.columns:
                b_i = finv[finv["Brand"] == selected_brand]
            if len(b_s) > 0:
                drill = compute_gmroi(b_s, b_i,
                                       ["Product", "Product Category"])
                show_table(drill,
                            ["Product", "Product Category"] + METRIC_COLS,
                            f"gmroi_{selected_brand}_products.csv")

    # ── TAB 3: By Product ──
    with tabs[2]:
        st.subheader("GMROI by Product")
        fsales, finv = render_filters(sales, inventory, "prod_tab")
        st.markdown("---")

        product_data = compute_gmroi(
            fsales, finv, ["Product", "Brand", "Product Category"])
        network_metrics(product_data)

        if not product_data.empty:
            st.markdown("---")
            search = st.text_input(
                "Search products:", "",
                help="Filter by product name (case-insensitive)",
                key="product_search")
            if search:
                product_display = product_data[
                    product_data["Product"].str.contains(
                        search, case=False, na=False)]
            else:
                product_display = product_data

            show_table(product_display,
                        ["Product", "Brand", "Product Category"] + METRIC_COLS,
                        "gmroi_by_product.csv")

    # ── TAB 4: By Shop ──
    with tabs[3]:
        st.subheader("GMROI by Shop")
        fsales, finv = render_filters(sales, inventory, "shop_tab")
        st.markdown("---")

        shop_data = compute_gmroi(fsales, finv, ["Shop"])
        network_metrics(shop_data)

        if not shop_data.empty:
            st.markdown("---")
            show_table(shop_data, ["Shop"] + METRIC_COLS,
                        "gmroi_by_shop.csv")

            st.markdown("---")
            st.subheader("Drill Down: Categories within Shop")
            shop_options = shop_data["Shop"].tolist()
            selected_shop = st.selectbox("Select shop:", shop_options,
                                          key="shop_drill")
            sh_s = fsales[fsales["Shop"] == selected_shop]
            sh_i = finv[finv["Shop"] == selected_shop]
            if len(sh_s) > 0:
                drill = compute_gmroi(sh_s, sh_i, ["Product Category"])
                show_table(drill,
                            ["Product Category"] + METRIC_COLS,
                            f"gmroi_{selected_shop}_categories.csv")

    # ── TAB 5: Trends ──
    with tabs[4]:
        st.subheader("📈 GMROI Trends (Monthly)")
        st.markdown("Monthly GMROI with rolling 3-month average. "
                     "Dashed lines = 3-month rolling avg. "
                     "Red dotted = break-even (1.0). Green dotted = strong (3.0).")

        trend_view = st.radio("View by:", ["Category", "Shop"],
                               horizontal=True, key="trend_view")

        if trend_view == "Category":
            monthly = compute_monthly_gmroi(sales, inventory, "Product Category")
            fig = build_trend_chart(monthly, "Product Category")
        else:
            monthly = compute_monthly_gmroi(sales, inventory, "Shop")
            fig = build_trend_chart(monthly, "Shop")

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for trend analysis.")

        # Network-level monthly trend
        st.markdown("---")
        st.subheader("Network GMROI Trend")
        net_monthly = sales.groupby("Month").agg(
            Net_Sales=("Net Sales", "sum"),
            COGS=("COGS_Calc", "sum"),
        ).reset_index()
        net_monthly["Gross_Margin"] = net_monthly["Net_Sales"] - net_monthly["COGS"]

        inv_monthly = inventory.copy()
        inv_monthly["Month"] = inv_monthly["Date"].dt.to_period("M")
        inv_agg = inv_monthly.groupby("Month").agg(
            Avg_Inv=("Inventory Value", "mean"),
        ).reset_index()

        net_monthly = net_monthly.merge(inv_agg, on="Month", how="left")
        net_monthly["GMROI"] = np.where(
            net_monthly["Avg_Inv"] > 0,
            net_monthly["Gross_Margin"] / net_monthly["Avg_Inv"], np.nan
        )
        net_monthly["Month_str"] = net_monthly["Month"].astype(str)
        net_monthly["Rolling_3m"] = net_monthly["GMROI"].rolling(3, min_periods=3).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=net_monthly["Month_str"], y=net_monthly["GMROI"],
            mode="lines+markers", name="Monthly GMROI",
            line=dict(width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=net_monthly["Month_str"], y=net_monthly["Rolling_3m"],
            mode="lines", name="3-Month Rolling Avg",
            line=dict(dash="dash", width=2),
        ))
        fig2.add_hline(y=1.0, line_dash="dot", line_color="red",
                        annotation_text="Break-even")
        fig2.update_layout(
            height=400,
            xaxis_title="Month", yaxis_title="GMROI",
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 6: Portfolio (Margin vs Turns) ──
    with tabs[5]:
        st.subheader("🎯 Portfolio Analysis: Margin vs Turns")
        st.markdown("""
Scatter plot showing Gross Margin % vs Inventory Turns. Each bubble is sized
by Net Sales. Quadrant thresholds are set at the **median** of your data.
Gray dotted curves are GMROI iso-lines (constant GMROI = Margin% x Turns / 100).

| Quadrant | Meaning | Action |
|---|---|---|
| **Stars** (top-right) | High margin, high turns | Protect, never out of stock |
| **Sleepers** (top-left) | High margin, low turns | Investigate placement/allocation |
| **Traffic Drivers** (bottom-right) | Low margin, high turns | Manage tightly, nudge margin up |
| **Dogs** (bottom-left) | Low margin, low turns | Reduce or exit |
        """)

        port_view = st.radio("Analyze by:", ["Brand", "Category"],
                              horizontal=True, key="port_view")

        if port_view == "Brand":
            port_data = compute_gmroi(sales, inventory, ["Brand"])
            fig = build_scatter_chart(port_data, "Brand")
        else:
            port_data = compute_gmroi(sales, inventory, ["Product Category"])
            fig = build_scatter_chart(port_data, "Product Category")

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for portfolio analysis.")

        # Quadrant summary table
        if not port_data.empty:
            valid = port_data.dropna(subset=["Margin_Pct", "Inv_Turns"]).copy()
            valid = valid[valid["Inv_Turns"] > 0]
            med_m = valid["Margin_Pct"].median()
            med_t = valid["Inv_Turns"].median()

            def classify(row):
                hm = row["Margin_Pct"] >= med_m
                ht = row["Inv_Turns"] >= med_t
                if hm and ht: return "Star"
                elif hm: return "Sleeper"
                elif ht: return "Traffic Driver"
                else: return "Dog"

            valid["Quadrant"] = valid.apply(classify, axis=1)

            st.markdown("---")
            st.subheader("Quadrant Summary")
            col_name = "Brand" if port_view == "Brand" else "Product Category"
            show_table(valid,
                        [col_name, "Quadrant"] + METRIC_COLS,
                        f"portfolio_{port_view.lower()}.csv")

    # ── TAB 7: Store Variance ──
    with tabs[6]:
        st.subheader("🔀 Store Variance Analysis")
        st.markdown("""
Identifies brands whose GMROI **varies significantly across stores**.
High variance = assortment opportunity. A brand that's a Star at one store
and a Dog at another may need store-specific allocation.

**CV % (Coefficient of Variation)** = how spread out the GMROI is relative
to the average. Higher CV = more inconsistent performance across stores.
        """)

        brand_shop, brand_stats = compute_store_variance(sales, inventory)

        if not brand_stats.empty:
            # Filter to brands sold in multiple stores with meaningful volume
            min_sales = st.slider("Minimum total sales ($):", 0, 500000,
                                   50000, step=10000, key="var_min_sales")
            min_shops = st.slider("Minimum number of stores:", 2, 13,
                                   5, key="var_min_shops")

            filtered_stats = brand_stats[
                (brand_stats["Total_Sales"] >= min_sales) &
                (brand_stats["Shop_Count"] >= min_shops)
            ].copy()

            st.markdown(f"**{len(filtered_stats)}** brands meet criteria "
                         f"(${min_sales:,}+ sales, {min_shops}+ stores)")

            if not filtered_stats.empty:
                st.markdown("---")
                st.subheader("Brands by Store Variance (highest CV first)")
                var_cols = ["Brand", "Shop_Count", "Total_Sales",
                            "GMROI_Mean", "GMROI_Median", "GMROI_Min",
                            "GMROI_Max", "GMROI_Range", "GMROI_CV"]
                show_table(filtered_stats, var_cols,
                            "brand_store_variance.csv")

                # Drill into a specific brand
                st.markdown("---")
                st.subheader("Brand Detail: GMROI by Store")
                brand_options = filtered_stats["Brand"].tolist()
                if brand_options:
                    sel_brand = st.selectbox(
                        "Select brand to see store breakdown:",
                        brand_options, key="var_brand_drill")

                    fig = build_variance_chart(brand_shop, sel_brand)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Show the raw data too
                    brand_detail = brand_shop[
                        brand_shop["Brand"] == sel_brand
                    ].sort_values("GMROI", ascending=False)
                    show_table(brand_detail,
                                ["Shop", "Brand"] + METRIC_COLS,
                                f"gmroi_{sel_brand}_by_store.csv")
        else:
            st.info("Not enough data for store variance analysis.")


if __name__ == "__main__":
    main()
