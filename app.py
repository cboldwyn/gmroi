"""
GMROI Dashboard v2.3.0
Gross Margin Return on Inventory Investment analysis for Haven Cannabis

Provides interactive analysis of product performance across Brand, Category,
Product, and Shop dimensions with vendor credit integration, monthly trends,
margin-turnover portfolio analysis, store-level variance detection,
COGS adjustments for off-system credits, validation tab, and PDF reports.

CHANGELOG:
v2.3.0 (2026-03-31)
- Persistent data: Parquet cache for fast startup (replaces raw CSV reload)
- Timeframe selector: view data by year, quarter, or custom date range
- Data management sidebar: rebuild from CSVs, new file detection
- Vendor credits: glob pattern supports multiple yearly files

v2.2.0 (2026-03-19)
- COGS Adjustments: off-system credit memos (Stiiizy 30% vape for 2025,
  20% non-vape from Nov 2025). Date-aware config with sidebar overrides.
- Stiiizy excluded from vendor credits CSV (credits handled via COGS adjustments)
- NEW: Validation tab for cross-referencing data against external sources
- NEW: PDF report generation (one-pager per filtered view)
- Sidebar controls for Stiiizy non-vape adjustment (% or $ amount)

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
from datetime import datetime, date

from src.store import (build_from_csvs, load_from_parquet, load_metadata,
                       detect_new_files)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_RIGHT, TA_CENTER
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "2.3.0"

st.set_page_config(
    page_title=f"GMROI Dashboard v{VERSION}",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = "data"
SALES_DIR = os.path.join(DATA_DIR, "sales")
INVENTORY_DIR = os.path.join(DATA_DIR, "inventory")
CREDITS_PATTERN = os.path.join(DATA_DIR, "*Vendor Credits*.csv")
DISTRO_INV_PATH = os.path.join(DATA_DIR, "Daily Inventory Values - Source.csv")

EXCLUDE_CATEGORIES = [
    "Display", "Sample", "Promo", "Compassion", "Donation", "Non-Cannabis",
    "Boxes"
]

FLOWER_CATEGORIES = ["Indica", "Sativa", "Hybrid"]

# Brand consolidation: map variant names to canonical name
BRAND_CONSOLIDATION = {
    "Camino Gummies": "Camino",
}

# Off-system COGS adjustments (credit memos not in vendor credits CSV).
# Stiiizy credits come via off-system rebates, NOT through the promo credit system.
# Rates are date-aware to handle deal changes through 2025.
# From Feb 2026, Distru unit costs already reflect credits (no adjustment needed).
#
# Structure: brand -> list of periods with start/end dates and category rates.
# 'default' applies to categories not explicitly listed.
# Non-vape can be overridden via sidebar controls (% or $ amount from accounting).
COGS_ADJUSTMENTS = {
    'Stiiizy': [
        {
            'start': '2025-01-01', 'end': '2025-10-31',
            'categories': {'Vape': 0.30},
            'default': 0.0,  # non-vape: adjustable via sidebar
        },
        {
            'start': '2025-11-01', 'end': '2026-01-31',
            'categories': {'Vape': 0.30, 'Accessories': 0.30},
            'default': 0.20,  # non-vape 20% upfront from Nov 2025 deal
        },
        # Feb 2026+: no adjustment, Distru unit costs already include credits
    ],
}

# Brands to EXCLUDE from vendor credits CSV (handled via COGS_ADJUSTMENTS)
IGNORE_VENDOR_CREDITS_BRANDS = ['Stiiizy']

# Portfolio quadrant thresholds: set dynamically from median of data
# GMROI iso-line values to draw on scatter plot
GMROI_ISO_VALUES = [1.0, 2.0, 3.0, 5.0]


def apply_cogs_adjustments(sales, adjustments=None, nonvape_override=None):
    """
    Apply off-system COGS adjustments as % reduction on standard COGS.

    These are rebates received outside the promo credit system. They reduce
    True COGS without changing the base Unit Cost in the data.

    Args:
        sales: DataFrame with Brand, Product Category, Unit Cost, Quantity Sold, Date
        adjustments: COGS_ADJUSTMENTS config dict (defaults to module-level config)
        nonvape_override: dict with optional sidebar overrides, e.g.
            {'Stiiizy': {'mode': 'pct', 'value': 0.15}} or
            {'Stiiizy': {'mode': 'dollar', 'value': 60862, 'start': '...', 'end': '...'}}

    Returns:
        DataFrame with COGS_Adjustment column added
    """
    if adjustments is None:
        adjustments = COGS_ADJUSTMENTS

    sales = sales.copy()
    sales['COGS_Adjustment'] = 0.0

    if not adjustments:
        return sales

    cat_col = 'Product Category'
    adjustment_log = []

    for brand, periods in adjustments.items():
        brand_mask = sales['Brand'] == brand
        if brand_mask.sum() == 0:
            continue

        for period in periods:
            start = pd.Timestamp(period['start'])
            end = pd.Timestamp(period['end'])
            period_mask = brand_mask & (sales['Date'] >= start) & (sales['Date'] <= end)

            if period_mask.sum() == 0:
                continue

            standard_cogs = sales.loc[period_mask, 'Unit Cost'] * sales.loc[period_mask, 'Quantity Sold']
            category_pcts = period.get('categories', {})
            default_pct = period.get('default', 0)

            # Apply category-specific rates first
            for category, pct in category_pcts.items():
                cat_mask = period_mask & (sales[cat_col] == category)
                cat_count = cat_mask.sum()
                if cat_count > 0:
                    cat_cogs = sales.loc[cat_mask, 'Unit Cost'] * sales.loc[cat_mask, 'Quantity Sold']
                    adj_amount = cat_cogs * pct
                    sales.loc[cat_mask, 'COGS_Adjustment'] = adj_amount
                    adjustment_log.append(
                        f"{brand} / {category} ({period['start']} to {period['end']}): "
                        f"{cat_count:,} rows x {pct:.0%} = ${adj_amount.sum():,.0f}"
                    )

            # Apply default rate to remaining (not already adjusted in this period)
            if default_pct > 0:
                already_adjusted = sales['COGS_Adjustment'] > 0
                remaining = period_mask & ~already_adjusted
                remaining_count = remaining.sum()
                if remaining_count > 0:
                    rem_cogs = sales.loc[remaining, 'Unit Cost'] * sales.loc[remaining, 'Quantity Sold']
                    adj_amount = rem_cogs * default_pct
                    sales.loc[remaining, 'COGS_Adjustment'] = adj_amount
                    cats = ', '.join(sorted(sales.loc[remaining, cat_col].unique()))
                    adjustment_log.append(
                        f"{brand} / Other [{cats}] ({period['start']} to {period['end']}): "
                        f"{remaining_count:,} rows x {default_pct:.0%} = ${adj_amount.sum():,.0f}"
                    )

        # Apply non-vape sidebar override if provided
        if nonvape_override and brand in nonvape_override:
            override = nonvape_override[brand]
            # Find non-vape rows for this brand that haven't been adjusted
            # (or override existing non-vape adjustments)
            nonvape_mask = brand_mask & ~sales[cat_col].isin(
                [cat for p in periods for cat in p.get('categories', {}).keys()]
            )
            if override['mode'] == 'pct' and override['value'] > 0:
                pct = override['value']
                nv_cogs = sales.loc[nonvape_mask, 'Unit Cost'] * sales.loc[nonvape_mask, 'Quantity Sold']
                sales.loc[nonvape_mask, 'COGS_Adjustment'] = nv_cogs * pct
                adjustment_log.append(
                    f"{brand} / Non-Vape (sidebar override): "
                    f"{nonvape_mask.sum():,} rows x {pct:.0%} = ${(nv_cogs * pct).sum():,.0f}"
                )
            elif override['mode'] == 'dollar' and override['value'] > 0:
                total_dollar = override['value']
                # Distribute pro-rata by COGS weight
                nv_cogs = sales.loc[nonvape_mask, 'Unit Cost'] * sales.loc[nonvape_mask, 'Quantity Sold']
                total_nv_cogs = nv_cogs.sum()
                if total_nv_cogs > 0:
                    sales.loc[nonvape_mask, 'COGS_Adjustment'] = nv_cogs / total_nv_cogs * total_dollar
                    adjustment_log.append(
                        f"{brand} / Non-Vape (sidebar $ override): "
                        f"${total_dollar:,.0f} distributed across {nonvape_mask.sum():,} rows"
                    )

    # Store log for sidebar display
    st.session_state['cogs_adjustment_log'] = adjustment_log
    st.session_state['cogs_adjustment_total'] = sales['COGS_Adjustment'].sum()

    return sales


def format_currency(value):
    """Format a number as currency string."""
    if abs(value) >= 1_000_000:
        return f"${value:,.0f}"
    elif abs(value) >= 1_000:
        return f"${value:,.0f}"
    else:
        return f"${value:,.2f}"


def generate_gmroi_report_pdf(df, filter_info, title="GMROI Report"):
    """
    Generate a print-ready PDF one-pager for a GMROI filtered view.
    Returns PDF bytes, or None if reportlab not available.
    """
    if not HAS_REPORTLAB:
        return None

    # Compute summary metrics
    total_sales = df["Net_Sales"].sum()
    total_cogs = df["COGS"].sum()
    total_gm = df["Gross_Margin"].sum()
    total_avg_inv = df["Avg_Inv_Cost"].sum() if "Avg_Inv_Cost" in df.columns else 0
    gmroi = total_gm / total_avg_inv if total_avg_inv > 0 else 0
    margin_pct = total_gm / total_sales * 100 if total_sales > 0 else 0
    turns = total_cogs / total_avg_inv if total_avg_inv > 0 else 0
    credits = df["Vendor_Credits"].sum() if "Vendor_Credits" in df.columns else 0
    cogs_adj = df["COGS_Adj"].sum() if "COGS_Adj" in df.columns else 0

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                            topMargin=0.4*inch, bottomMargin=0.4*inch)

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=16, textColor=colors.HexColor('#2E7D32'),
                                  spaceAfter=2)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     fontSize=8, textColor=colors.gray, spaceAfter=6)

    # Build filter description
    filter_desc = filter_info if isinstance(filter_info, str) else "All Data"

    # Header
    header_data = [
        [Paragraph(f"<b>{title}</b>", title_style),
         '',
         Paragraph(f"<b>{len(df):,}</b> items",
                   ParagraphStyle('Right', parent=styles['Normal'],
                                  fontSize=12, alignment=TA_RIGHT))]
    ]
    header_table = Table(header_data, colWidths=[4*inch, 1.5*inch, 2*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (-1, 0), (-1, 0), 'RIGHT'),
    ]))
    elements.append(header_table)

    elements.append(Paragraph(
        f"{datetime.now().strftime('%B %d, %Y')} - Haven Cannabis - <i>{filter_desc}</i>",
        subtitle_style))
    elements.append(Spacer(1, 6))

    # Hero metrics
    hero_style = ParagraphStyle('Hero', parent=styles['Normal'], fontSize=22,
                                 alignment=TA_CENTER, textColor=colors.white, leading=24)
    hero_label_style = ParagraphStyle('HeroLabel', parent=styles['Normal'], fontSize=8,
                                       alignment=TA_CENTER,
                                       textColor=colors.HexColor('#E8F5E9'))

    hero_data = [
        [Paragraph("<b>GMROI</b>", hero_label_style),
         Paragraph("<b>MARGIN %</b>", hero_label_style),
         Paragraph("<b>GROSS MARGIN</b>", hero_label_style)],
        [Paragraph(f"<b>{gmroi:.2f}</b>", hero_style),
         Paragraph(f"<b>{margin_pct:.1f}%</b>", hero_style),
         Paragraph(f"<b>{format_currency(total_gm)}</b>", hero_style)],
    ]
    hero_table = Table(hero_data, colWidths=[2.5*inch]*3,
                       rowHeights=[0.22*inch, 0.38*inch])
    hero_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#2E7D32')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 0),
        ('TOPPADDING', (0, 1), (-1, 1), 2),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 6),
    ]))
    elements.append(hero_table)
    elements.append(Spacer(1, 6))

    # Detail metrics
    detail_label_style = ParagraphStyle('DetailLabel', parent=styles['Normal'],
                                         fontSize=7, alignment=TA_CENTER,
                                         textColor=colors.gray)
    detail_value_style = ParagraphStyle('DetailValue', parent=styles['Normal'],
                                         fontSize=11, alignment=TA_CENTER,
                                         fontName='Helvetica-Bold')

    detail_labels = ["NET SALES", "COGS", "INV TURNS", "AVG INV COST",
                     "VENDOR CREDITS", "COGS ADJ"]
    detail_values = [
        format_currency(total_sales), format_currency(total_cogs),
        f"{turns:.2f}", format_currency(total_avg_inv),
        format_currency(credits), format_currency(cogs_adj),
    ]

    detail_data = [
        [Paragraph(l, detail_label_style) for l in detail_labels],
        [Paragraph(v, detail_value_style) for v in detail_values],
    ]
    detail_table = Table(detail_data, colWidths=[1.25*inch]*6,
                          rowHeights=[0.2*inch, 0.3*inch])
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FAFAFA')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 4),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 10))

    # Data table
    th_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=8,
                               textColor=colors.white, fontName='Helvetica-Bold')
    td_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontSize=8)
    td_right = ParagraphStyle('TableCellRight', parent=styles['Normal'],
                               fontSize=8, alignment=TA_RIGHT)

    # Determine group columns (everything that's not a metric)
    metric_set = set(METRIC_COLS)
    group_cols = [c for c in df.columns if c not in metric_set]
    display_group_cols = group_cols[:2]  # Max 2 group columns in PDF

    table_headers = display_group_cols + ["GMROI", "Margin %", "Gross Margin",
                                          "Net Sales", "Turns", "Qty"]
    table_data = [[Paragraph(h, th_style) for h in table_headers]]

    sorted_df = df.sort_values("GMROI", ascending=False)
    max_rows = min(len(sorted_df), 40)

    for _, row in sorted_df.head(max_rows).iterrows():
        row_data = []
        for col in display_group_cols:
            val = str(row.get(col, ''))
            if len(val) > 30:
                val = val[:27] + '...'
            row_data.append(Paragraph(val, td_style))
        row_data.extend([
            Paragraph(f"{row.get('GMROI', 0):.2f}", td_right),
            Paragraph(f"{row.get('Margin_Pct', 0):.1f}%", td_right),
            Paragraph(format_currency(row.get('Gross_Margin', 0)), td_right),
            Paragraph(format_currency(row.get('Net_Sales', 0)), td_right),
            Paragraph(f"{row.get('Inv_Turns', 0):.1f}", td_right),
            Paragraph(f"{row.get('Qty_Sold', 0):,.0f}", td_right),
        ])
        table_data.append(row_data)

    # Column widths
    n_groups = len(display_group_cols)
    group_width = 1.8 if n_groups == 1 else 1.2
    metric_width = (7.5 - group_width * n_groups) / 6
    col_widths = [group_width * inch] * n_groups + [metric_width * inch] * 6

    product_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    product_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E7D32')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ('TOPPADDING', (0, 1), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#FAFAFA')]),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E0E0E0')),
        ('ALIGN', (n_groups, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(product_table)

    if len(sorted_df) > max_rows:
        elements.append(Spacer(1, 4))
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
                                        fontSize=7, textColor=colors.gray,
                                        alignment=TA_CENTER)
        elements.append(Paragraph(
            f"Showing top {max_rows} of {len(sorted_df)} items by GMROI",
            footer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def pdf_download_button(df, filter_desc, title, key):
    """Render a PDF download button if reportlab is available."""
    if not HAS_REPORTLAB:
        return
    pdf_bytes = generate_gmroi_report_pdf(df, filter_desc, title)
    if pdf_bytes:
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{title.lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
            key=key,
        )


# ============================================================================
# DATA LOADING & PREPARATION (one-time, cached)
# ============================================================================

@st.cache_resource(show_spinner="Loading data...")
def load_and_prepare():
    """Load data from Parquet cache (fast) or build from CSVs (first run).
    Uses cache_resource (not cache_data) to avoid pickle serialization overhead."""
    result = load_from_parquet()
    if result[0] is not None:
        return result
    # No cache yet: build from CSVs
    sales, inventory = build_from_csvs(
        SALES_DIR, INVENTORY_DIR, CREDITS_PATTERN, version=VERSION
    )
    metadata = load_metadata()
    return sales, inventory, metadata


def apply_adjustments_to_sales(sales, nonvape_override=None):
    """Apply COGS adjustments post-cache (uses session_state for logging)."""
    sales = apply_cogs_adjustments(sales, nonvape_override=nonvape_override)
    # Recalculate COGS_Calc with adjustments
    sales["COGS_Calc"] = (sales["Unit Cost"] * sales["Quantity Sold"]
                           - sales["Vendor_Pays"] - sales["COGS_Adjustment"]
                           + sales["Haven_Pays"])
    return sales


# ============================================================================
# COMPUTATION
# ============================================================================

def calc_annualize_factor(df, date_col="Date"):
    """Compute 365-day annualization factor from a date-filtered DataFrame.

    Every GMROI/Turns display must be annualized against the window it covers.
    Call this on the DataFrame that was actually used to compute the metric
    (not a parent/sidebar-scoped DataFrame)."""
    if df is None or len(df) == 0:
        return 1.0
    actual_days = (df[date_col].max() - df[date_col].min()).days + 1
    return 365 / actual_days if actual_days > 0 else 1.0


def compute_gmroi(sales, inventory, group_cols, annualize_factor=1.0):
    """Compute GMROI for given grouping columns. COGS includes vendor credits.
    annualize_factor: multiply GMROI and Turns to normalize to annual basis
    (e.g. 4.0 for a quarter, 12.0 for a month)."""
    if len(sales) == 0:
        return pd.DataFrame()

    sales_agg = sales.groupby(group_cols, dropna=False, observed=True).agg(
        Net_Sales=("Net Sales", "sum"),
        COGS=("COGS_Calc", "sum"),
        Qty_Sold=("Quantity Sold", "sum"),
        Transactions=("Trans No", "nunique"),
        Avg_Unit_Cost=("Unit Cost", "mean"),
        Vendor_Credits=("Vendor_Pays", "sum"),
        **({
            "COGS_Adj": ("COGS_Adjustment", "sum"),
        } if "COGS_Adjustment" in sales.columns else {}),
    ).reset_index()

    # Realized ASP: consistent with Net_Sales and Qty_Sold shown in the same row.
    sales_agg["Avg_Sell_Price"] = np.where(
        sales_agg["Qty_Sold"] > 0,
        sales_agg["Net_Sales"] / sales_agg["Qty_Sold"], 0)

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

    # Sum inventory per group per DAY (weekly files contain 7 daily snapshots)
    daily = inv.groupby(group_cols + ["Date"], dropna=False, observed=True).agg(
        Daily_Inv_Value=("Inventory Value", "sum"),
        Daily_Qty=("Quantity on Hand", "sum"),
    ).reset_index()

    avg_inv = daily.groupby(group_cols, dropna=False, observed=True).agg(
        Avg_Inv_Cost=("Daily_Inv_Value", "mean"),
        Avg_Qty_On_Hand=("Daily_Qty", "mean"),
    ).reset_index()

    merged = sales_agg.merge(avg_inv, on=group_cols, how="left")

    merged["GMROI"] = np.where(
        merged["Avg_Inv_Cost"] > 0,
        merged["Gross_Margin"] / merged["Avg_Inv_Cost"] * annualize_factor,
        np.nan
    )
    merged["Inv_Turns"] = np.where(
        merged["Avg_Inv_Cost"] > 0,
        merged["COGS"] / merged["Avg_Inv_Cost"] * annualize_factor, np.nan
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
    monthly_sales = sales.groupby([group_col, "Month"], dropna=False, observed=True).agg(
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
    # Sum per group per day first, then average across days within each month
    daily_inv = inv.groupby([group_col, "Month", "Date"], dropna=False, observed=True).agg(
        Daily_Inv=("Inventory Value", "sum"),
    ).reset_index()
    monthly_inv = daily_inv.groupby([group_col, "Month"], dropna=False, observed=True).agg(
        Avg_Inv_Cost=("Daily_Inv", "mean"),
    ).reset_index()

    merged = monthly_sales.merge(monthly_inv, on=[group_col, "Month"], how="left")
    # Annualize monthly GMROI (x12) so break-even at 1.0 is meaningful
    merged["GMROI"] = np.where(
        merged["Avg_Inv_Cost"] > 0,
        merged["Gross_Margin"] / merged["Avg_Inv_Cost"] * 12, np.nan
    )

    # Convert Period to string for plotting
    merged["Month_str"] = merged["Month"].astype(str)

    return merged


def compute_store_variance(sales, inventory, annualize_factor=1.0):
    """
    Compute GMROI by Brand x Shop and by Product x Shop.
    Returns DataFrames with variance metrics to identify assortment opportunities.
    """
    if len(sales) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Brand x Shop
    brand_shop = compute_gmroi(sales, inventory, ["Brand", "Shop"],
                               annualize_factor)

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


def compute_share_metrics(sales, group_cols, annualize_factor=1.0,
                          inventory=None):
    """Compute sales metrics, shares, and optionally GMROI/Turns."""
    if len(sales) == 0:
        return pd.DataFrame()
    agg = sales.groupby(group_cols, dropna=False, observed=True).agg(
        Units_Sold=("Quantity Sold", "sum"),
        Revenue=("Net Sales", "sum"),
        COGS=("COGS_Calc", "sum"),
        Avg_Cost=("Unit Cost", "mean"),
    ).reset_index()
    # Realized ASP: consistent with Revenue and Units_Sold shown in the same row.
    agg["Avg_Sell"] = np.where(agg["Units_Sold"] > 0,
                                agg["Revenue"] / agg["Units_Sold"], 0)
    agg["Gross_Profit"] = agg["Revenue"] - agg["COGS"]
    total_units = agg["Units_Sold"].sum()
    total_rev = agg["Revenue"].sum()
    total_gp = agg["Gross_Profit"].sum()
    agg["Units_Share"] = np.where(total_units > 0,
                                   agg["Units_Sold"] / total_units * 100, 0)
    agg["Revenue_Share"] = np.where(total_rev > 0,
                                     agg["Revenue"] / total_rev * 100, 0)
    agg["GP_Share"] = np.where(total_gp > 0,
                                agg["Gross_Profit"] / total_gp * 100, 0)
    agg["Margin_Pct"] = np.where(agg["Revenue"] > 0,
                                  agg["Gross_Profit"] / agg["Revenue"] * 100, 0)

    # GMROI and Turns from inventory (when available)
    if inventory is not None and len(inventory) > 0:
        inv = inventory.copy()
        if "Product Name" in inv.columns:
            inv = inv.rename(columns={"Product Name": "Product"})
        # Only group by columns that exist in inventory
        inv_group = [c for c in group_cols if c in inv.columns]
        if inv_group:
            daily = inv.groupby(inv_group + ["Date"], dropna=False,
                                observed=True).agg(
                Daily_Inv=("Inventory Value", "sum")).reset_index()
            avg_inv = daily.groupby(inv_group, dropna=False,
                                    observed=True).agg(
                Avg_Inv_Cost=("Daily_Inv", "mean")).reset_index()
            agg = agg.merge(avg_inv, on=inv_group, how="left")
            agg["GMROI"] = np.where(
                agg["Avg_Inv_Cost"] > 0,
                agg["Gross_Profit"] / agg["Avg_Inv_Cost"] * annualize_factor,
                np.nan)
            agg["Inv_Turns"] = np.where(
                agg["Avg_Inv_Cost"] > 0,
                agg["COGS"] / agg["Avg_Inv_Cost"] * annualize_factor,
                np.nan)

    return agg.sort_values("Units_Sold", ascending=False)


def build_store_matrix(sales, col_col, value_col="Quantity Sold"):
    """Pivot: rows = Shop, columns = col_col, values = sum of value_col."""
    pivot = sales.pivot_table(
        index="Shop", columns=col_col, values=value_col,
        aggfunc="sum", fill_value=0, observed=True
    )
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)
    return pivot


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
    "COGS_Adj": st.column_config.NumberColumn("COGS Adj", format="$%,.0f"),
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
    "Units_Share": st.column_config.NumberColumn("Units Share %", format="%.1f%%"),
    "Revenue_Share": st.column_config.NumberColumn("Revenue Share %", format="%.1f%%"),
    "Profile_Template": st.column_config.TextColumn("Profile Template"),
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


def network_metrics(df, annualize_factor=1.0, inventory=None):
    """Display network-level summary metrics. GMROI and Turns are annualized.
    When inventory is provided, compute avg inventory from raw data (correct).
    Otherwise falls back to summing group averages (approximate)."""
    if df.empty:
        st.info("No data matches current filters.")
        return

    total_sales = df["Net_Sales"].sum()
    total_gm = df["Gross_Margin"].sum()
    total_cogs = df["COGS"].sum()
    if inventory is not None and len(inventory) > 0:
        daily_total = inventory.groupby("Date", observed=True).agg(
            Daily_Total=("Inventory Value", "sum")).reset_index()
        total_avg_inv = daily_total["Daily_Total"].mean()
    else:
        total_avg_inv = df["Avg_Inv_Cost"].sum()
    gmroi = (total_gm / total_avg_inv * annualize_factor
             if total_avg_inv > 0 else 0)
    margin_pct = total_gm / total_sales * 100 if total_sales > 0 else 0
    turns = (total_cogs / total_avg_inv * annualize_factor
             if total_avg_inv > 0 else 0)
    credits = df["Vendor_Credits"].sum()
    cogs_adj = df["COGS_Adj"].sum() if "COGS_Adj" in df.columns else 0

    ann_label = " (Ann.)" if annualize_factor != 1.0 else ""
    cols = st.columns(7 if cogs_adj > 0 else 6)
    with cols[0]:
        st.metric("Net Sales", f"${total_sales:,.0f}")
    with cols[1]:
        st.metric("Gross Margin", f"${total_gm:,.0f}")
    with cols[2]:
        st.metric("Margin %", f"{margin_pct:.1f}%")
    with cols[3]:
        st.metric(f"GMROI{ann_label}", f"{gmroi:.2f}")
    with cols[4]:
        st.metric(f"Inv Turns{ann_label}", f"{turns:.2f}")
    with cols[5]:
        st.metric("Vendor Credits", f"${credits:,.0f}")
    if cogs_adj > 0:
        with cols[6]:
            st.metric("COGS Adj", f"${cogs_adj:,.0f}")


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
    # Filter out outliers: require minimum $5k sales and margin between -100% and 100%
    plot_data = plot_data[plot_data["Net_Sales"] >= 5000]
    plot_data = plot_data[(plot_data["Margin_Pct"] >= -100) & (plot_data["Margin_Pct"] <= 100)]
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
               "Avg_Qty_On_Hand", "Vendor_Credits", "COGS_Adj"]


def build_timeframe_presets(sales):
    """Generate timeframe presets from the data's actual date range."""
    min_date = sales["Date"].min().date()
    max_date = sales["Date"].max().date()
    presets = {"All Data": (None, None)}

    # Walk years present in data
    for year in range(min_date.year, max_date.year + 1):
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        if year_start <= max_date and year_end >= min_date:
            presets[f"{year} Full Year"] = (year_start, year_end)

        # Quarters within each year
        for q in range(1, 5):
            q_start = date(year, (q - 1) * 3 + 1, 1)
            # Last day of quarter: Q1=3/31, Q2=6/30, Q3=9/30, Q4=12/31
            q_end_month = q * 3
            if q_end_month == 6 or q_end_month == 9:
                q_end = date(year, q_end_month, 30)
            else:
                q_end = date(year, q_end_month, 31)
            # Only include quarters that overlap with actual data
            if q_start <= max_date and q_end >= min_date:
                presets[f"Q{q} {year}"] = (q_start, q_end)

    presets["Custom Range"] = "custom"
    return presets


def main():
    st.title(f"📊 GMROI Dashboard v{VERSION}")

    # ── Load Data ──
    sales, inventory, metadata = load_and_prepare()

    if sales is None or inventory is None:
        st.error("Could not load data. Check that data/sales/ and "
                 "data/inventory/ contain CSV files.")
        return

    # ── Sidebar: Timeframe ──
    st.sidebar.header("Timeframe")
    presets = build_timeframe_presets(sales)
    preset_names = list(presets.keys())
    selected_tf = st.sidebar.selectbox(
        "Period", preset_names, key="timeframe_select",
        label_visibility="collapsed"
    )

    start_date = None
    end_date = None
    timeframe_label = selected_tf

    if selected_tf == "Custom Range":
        data_min = sales["Date"].min().date()
        data_max = sales["Date"].max().date()
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input("From", value=data_min, min_value=data_min,
                                      max_value=data_max, key="tf_start")
        end_date = col2.date_input("To", value=data_max, min_value=data_min,
                                    max_value=data_max, key="tf_end")
        timeframe_label = f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"
    elif selected_tf != "All Data":
        start_date, end_date = presets[selected_tf]

    # Apply timeframe filter
    if start_date and end_date:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        sales = sales[(sales["Date"] >= start_ts) & (sales["Date"] <= end_ts)].copy()
        inventory = inventory[(inventory["Date"] >= start_ts)
                              & (inventory["Date"] <= end_ts)].copy()
        if len(sales) == 0:
            st.warning(f"No sales data in the selected period ({timeframe_label}).")
            return

    # Annualization factor: normalize GMROI and Turns to a 365-day basis
    # so that Q1, Q2, YTD, and full-year periods are directly comparable.
    annualize_factor = calc_annualize_factor(sales)

    if selected_tf == "All Data":
        st.markdown("Gross Margin Return on Inventory Investment - Haven")
    else:
        st.markdown(f"Gross Margin Return on Inventory Investment - Haven - {timeframe_label}")

    # ── Sidebar: Data Management ──
    st.sidebar.markdown("---")
    with st.sidebar.expander("Data Management"):
        if metadata:
            built = metadata.get("build_timestamp", "unknown")
            if built != "unknown":
                built = built[:19].replace("T", " ")
            st.caption(f"Last built: {built}")
            dr = metadata.get("sales_date_range", [])
            if dr:
                st.caption(f"Sales: {dr[0]} to {dr[1]}")
            st.caption(f"Sales rows: {metadata.get('sales_row_count', '?'):,}")
            st.caption(f"Inventory rows: {metadata.get('inventory_row_count', '?'):,}")

            new_sales, new_inv = detect_new_files(SALES_DIR, INVENTORY_DIR, metadata)
            if new_sales or new_inv:
                st.info(f"{len(new_sales)} new sales file(s), "
                        f"{len(new_inv)} new inventory file(s) detected.")

        if st.button("Rebuild Data from CSVs", type="secondary"):
            with st.spinner("Rebuilding from CSVs..."):
                build_from_csvs(SALES_DIR, INVENTORY_DIR, CREDITS_PATTERN,
                                version=VERSION)
            st.cache_data.clear()
            st.rerun()

    # ── Sidebar: Settings ──
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Settings")

    # Flower consolidation toggle
    combine_flower = st.sidebar.toggle(
        "Combine Flower categories",
        value=True,
        help="When ON, Indica/Sativa/Hybrid are grouped as 'Flower'. "
             "When OFF, they appear as separate categories."
    )

    # Category exclusion toggles
    exclude_giveaways = st.sidebar.toggle(
        "Exclude giveaways & apparel",
        value=False,
        help="Remove 'Accessories (under $5)' (giveaway items) and Apparel (staff t-shirts) from analysis."
    )

    # Apply category exclusions
    extra_excludes = []
    if exclude_giveaways:
        extra_excludes.extend(["Accessories (under $5)", "Apparel"])
    if extra_excludes:
        sales = sales.copy()
        sales = sales[~sales["Product Category"].isin(extra_excludes)]
        inventory = inventory.copy()
        inventory = inventory[~inventory["Product Category"].isin(extra_excludes)]

    if combine_flower:
        flower_map = {cat: "Flower" for cat in FLOWER_CATEGORIES}
        if not extra_excludes:
            sales = sales.copy()
            inventory = inventory.copy()
        sales["Product Category"] = sales["Product Category"].replace(flower_map)
        inventory["Product Category"] = inventory["Product Category"].replace(flower_map)

    # Stiiizy non-vape adjustment controls
    st.sidebar.markdown("---")
    st.sidebar.header("Stiiizy Non-Vape Adjustment")
    st.sidebar.caption(
        "Vape gets 30% COGS reduction for 2025 (auto-applied). "
        "Non-vape credits varied by period. Override here with % or $ from accounting."
    )
    nv_mode = st.sidebar.radio(
        "Non-vape adjustment mode:",
        ["None", "Percentage", "Dollar Amount"],
        horizontal=True,
        key="stiiizy_nv_mode",
    )
    nonvape_override = None
    if nv_mode == "Percentage":
        nv_pct = st.sidebar.number_input(
            "Non-vape credit %:", min_value=0.0, max_value=50.0,
            value=0.0, step=1.0, key="stiiizy_nv_pct",
            help="Applied to all non-vape Stiiizy sales COGS"
        )
        if nv_pct > 0:
            nonvape_override = {'Stiiizy': {'mode': 'pct', 'value': nv_pct / 100}}
    elif nv_mode == "Dollar Amount":
        nv_dollar = st.sidebar.number_input(
            "Total non-vape credit ($):", min_value=0.0,
            value=0.0, step=1000.0, key="stiiizy_nv_dollar",
            help="Total credit distributed pro-rata across non-vape Stiiizy COGS"
        )
        if nv_dollar > 0:
            nonvape_override = {'Stiiizy': {'mode': 'dollar', 'value': nv_dollar}}

    # Apply COGS adjustments (post-cache, uses session_state for logging)
    sales = apply_adjustments_to_sales(sales, nonvape_override=nonvape_override)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Data Summary")
    st.sidebar.markdown(f"**Sales rows:** {len(sales):,}")
    st.sidebar.markdown(f"**Net Sales:** ${sales['Net Sales'].sum():,.0f}")
    credit_total = sales["Vendor_Pays"].sum()
    if credit_total > 0:
        st.sidebar.markdown(f"**Vendor Credits:** ${credit_total:,.0f}")
    cogs_adj_total = st.session_state.get('cogs_adjustment_total', 0)
    if cogs_adj_total > 0:
        st.sidebar.markdown(f"**COGS Adjustments:** ${cogs_adj_total:,.0f}")
        with st.sidebar.expander("COGS Adjustment Breakdown"):
            for line in st.session_state.get('cogs_adjustment_log', []):
                st.caption(line)
    st.sidebar.markdown(f"**Categories:** {sales['Product Category'].nunique()}")
    st.sidebar.markdown(f"**Brands:** {sales['Brand'].nunique()}")
    st.sidebar.markdown(f"**Shops:** {sales['Shop'].nunique()}")
    st.sidebar.markdown(f"**Date range:** {sales['Date'].min().date()} to "
                         f"{sales['Date'].max().date()}")
    st.sidebar.markdown("---")
    with st.sidebar.expander("📋 Version History"):
        st.markdown(f"""
**v{VERSION}** (2026-03-31)
- Persistent data (Parquet cache)
- Timeframe selector (year, quarter, custom)
- Data management sidebar

**v2.2.0** (2026-03-19)
- COGS Adjustments (Stiiizy off-system credits)
- Validation tab, PDF reports

**v2.1.0** (2026-03-10)
- Flower combine/separate toggle
- Trends, Portfolio, Store Variance tabs
- True COGS everywhere, sortable columns
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
        "✅ Validation",
        "🏆 Product Performance",
    ])

    # ── TAB 1: By Category ──
    with tabs[0]:
        st.subheader("GMROI by Category")
        fsales, finv = render_filters(sales, inventory, "cat_tab")
        st.markdown("---")

        cat_data = compute_gmroi(fsales, finv, ["Product Category"], annualize_factor)
        network_metrics(cat_data, annualize_factor, inventory=finv)

        if not cat_data.empty:
            st.markdown("---")
            show_table(cat_data, ["Product Category"] + METRIC_COLS,
                        "gmroi_by_category.csv")
            pdf_download_button(cat_data, "All Categories",
                                "GMROI by Category", "pdf_cat")

            st.markdown("---")
            st.subheader("Drill Down: Brands within Category")
            cat_options = cat_data["Product Category"].tolist()
            selected_cat = st.selectbox("Select category:", cat_options,
                                         key="cat_drill")
            cat_s = fsales[fsales["Product Category"] == selected_cat]
            cat_i = finv[finv["Product Category"] == selected_cat]
            if len(cat_s) > 0:
                drill = compute_gmroi(cat_s, cat_i, ["Brand"], annualize_factor)
                show_table(drill, ["Brand"] + METRIC_COLS,
                            f"gmroi_{selected_cat}_brands.csv")

    # ── TAB 2: By Brand ──
    with tabs[1]:
        st.subheader("GMROI by Brand")
        fsales, finv = render_filters(sales, inventory, "brand_tab")
        st.markdown("---")

        brand_data = compute_gmroi(fsales, finv, ["Brand"], annualize_factor)
        network_metrics(brand_data, annualize_factor, inventory=finv)

        if not brand_data.empty:
            st.markdown("---")
            show_table(brand_data, ["Brand"] + METRIC_COLS,
                        "gmroi_by_brand.csv")
            pdf_download_button(brand_data, "All Brands",
                                "GMROI by Brand", "pdf_brand")

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
                                       ["Product", "Product Category"],
                                       annualize_factor)
                show_table(drill,
                            ["Product", "Product Category"] + METRIC_COLS,
                            f"gmroi_{selected_brand}_products.csv")

    # ── TAB 3: By Product ──
    with tabs[2]:
        st.subheader("GMROI by Product")
        fsales, finv = render_filters(sales, inventory, "prod_tab")
        st.markdown("---")

        product_data = compute_gmroi(
            fsales, finv, ["Product", "Brand", "Product Category"],
            annualize_factor)
        network_metrics(product_data, annualize_factor, inventory=finv)

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

        shop_data = compute_gmroi(fsales, finv, ["Shop"], annualize_factor)
        network_metrics(shop_data, annualize_factor, inventory=finv)

        if not shop_data.empty:
            st.markdown("---")
            show_table(shop_data, ["Shop"] + METRIC_COLS,
                        "gmroi_by_shop.csv")
            pdf_download_button(shop_data, "All Shops",
                                "GMROI by Shop", "pdf_shop")

            st.markdown("---")
            st.subheader("Drill Down: Categories within Shop")
            shop_options = shop_data["Shop"].tolist()
            selected_shop = st.selectbox("Select shop:", shop_options,
                                          key="shop_drill")
            sh_s = fsales[fsales["Shop"] == selected_shop]
            sh_i = finv[finv["Shop"] == selected_shop]
            if len(sh_s) > 0:
                drill = compute_gmroi(sh_s, sh_i, ["Product Category"], annualize_factor)
                show_table(drill,
                            ["Product Category"] + METRIC_COLS,
                            f"gmroi_{selected_shop}_categories.csv")

    # ── TAB 5: Trends ──
    with tabs[4]:
        st.subheader("📈 GMROI Trends (Monthly, Annualized)")
        st.markdown("Monthly GMROI annualized (x12) with rolling 3-month average. "
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
        st.subheader("Network GMROI Trend (Annualized)")
        st.caption(
            "Monthly GMROI annualized (x12) so break-even at 1.0 is comparable to the full-year metric. "
            "Retail = store inventory only. System-Wide = stores + Distro warehouse."
        )
        net_monthly = sales.groupby("Month").agg(
            Net_Sales=("Net Sales", "sum"),
            COGS=("COGS_Calc", "sum"),
        ).reset_index()
        net_monthly["Gross_Margin"] = net_monthly["Net_Sales"] - net_monthly["COGS"]

        inv_monthly = inventory.copy()
        inv_monthly["Month"] = inv_monthly["Date"].dt.to_period("M")
        # Sum inventory per day first (each snapshot = total across all products),
        # then average across days within each month
        daily_totals = inv_monthly.groupby(["Month", "Date"]).agg(
            Daily_Total=("Inventory Value", "sum"),
        ).reset_index()
        inv_agg = daily_totals.groupby("Month").agg(
            Avg_Inv=("Daily_Total", "mean"),
        ).reset_index()

        net_monthly = net_monthly.merge(inv_agg, on="Month", how="left")
        # Annualize monthly GMROI (multiply by 12) so break-even at 1.0 is meaningful
        net_monthly["GMROI_Retail"] = np.where(
            net_monthly["Avg_Inv"] > 0,
            net_monthly["Gross_Margin"] / net_monthly["Avg_Inv"] * 12, np.nan
        )
        net_monthly["Month_str"] = net_monthly["Month"].astype(str)
        net_monthly["Rolling_3m_Retail"] = net_monthly["GMROI_Retail"].rolling(
            3, min_periods=3).mean()

        # Load Distro warehouse inventory for system-wide view
        distro_inv = None
        if os.path.exists(DISTRO_INV_PATH):
            try:
                distro_raw = pd.read_csv(DISTRO_INV_PATH, encoding="utf-8-sig")
                distro_raw["Date"] = pd.to_datetime(distro_raw["Date"], errors="coerce")
                distro_raw["Distro"] = (distro_raw["Distro"].astype(str)
                                        .str.replace(r'[\$,]', '', regex=True))
                distro_raw["Distro"] = pd.to_numeric(distro_raw["Distro"],
                                                      errors="coerce")
                distro_raw = distro_raw.dropna(subset=["Date", "Distro"])
                if start_date and end_date:
                    distro_raw = distro_raw[
                        (distro_raw["Date"] >= pd.Timestamp(start_date))
                        & (distro_raw["Date"] <= pd.Timestamp(end_date))
                    ]
                distro_raw["Month"] = distro_raw["Date"].dt.to_period("M")
                distro_inv = distro_raw.groupby("Month").agg(
                    Distro_Avg_Inv=("Distro", "mean"),
                ).reset_index()
            except Exception:
                distro_inv = None

        if distro_inv is not None and not distro_inv.empty:
            net_monthly = net_monthly.merge(distro_inv, on="Month", how="left")
            net_monthly["Distro_Avg_Inv"] = net_monthly["Distro_Avg_Inv"].fillna(0)
            net_monthly["System_Avg_Inv"] = (net_monthly["Avg_Inv"]
                                              + net_monthly["Distro_Avg_Inv"])
            net_monthly["GMROI_System"] = np.where(
                net_monthly["System_Avg_Inv"] > 0,
                net_monthly["Gross_Margin"] / net_monthly["System_Avg_Inv"] * 12, np.nan
            )
            net_monthly["Rolling_3m_System"] = net_monthly["GMROI_System"].rolling(
                3, min_periods=3).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=net_monthly["Month_str"], y=net_monthly["GMROI_Retail"],
            mode="lines+markers", name="Retail GMROI",
            line=dict(width=2, color="#2ecc71"),
        ))
        fig2.add_trace(go.Scatter(
            x=net_monthly["Month_str"], y=net_monthly["Rolling_3m_Retail"],
            mode="lines", name="Retail 3mo Avg",
            line=dict(dash="dash", width=1, color="#2ecc71"),
        ))

        if "GMROI_System" in net_monthly.columns:
            fig2.add_trace(go.Scatter(
                x=net_monthly["Month_str"], y=net_monthly["GMROI_System"],
                mode="lines+markers", name="System-Wide GMROI",
                line=dict(width=2, color="#3498db"),
            ))
            fig2.add_trace(go.Scatter(
                x=net_monthly["Month_str"], y=net_monthly["Rolling_3m_System"],
                mode="lines", name="System 3mo Avg",
                line=dict(dash="dash", width=1, color="#3498db"),
            ))

        fig2.add_hline(y=1.0, line_dash="dot", line_color="red",
                        annotation_text="Break-even")
        fig2.update_layout(
            height=400,
            xaxis_title="Month", yaxis_title="GMROI (Annualized)",
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Show the underlying data
        if "GMROI_System" in net_monthly.columns:
            with st.expander("Monthly Data"):
                trend_display = net_monthly[["Month_str", "Net_Sales", "Gross_Margin",
                                              "Avg_Inv", "GMROI_Retail",
                                              "Distro_Avg_Inv", "System_Avg_Inv",
                                              "GMROI_System"]].copy()
                trend_display.columns = ["Month", "Net Sales", "Gross Margin",
                                          "Retail Inv", "Retail GMROI",
                                          "Distro Inv", "System Inv",
                                          "System GMROI"]
                config = {
                    "Net Sales": st.column_config.NumberColumn(format="$%,.0f"),
                    "Gross Margin": st.column_config.NumberColumn(format="$%,.0f"),
                    "Retail Inv": st.column_config.NumberColumn(format="$%,.0f"),
                    "Retail GMROI": st.column_config.NumberColumn(format="%.2f"),
                    "Distro Inv": st.column_config.NumberColumn(format="$%,.0f"),
                    "System Inv": st.column_config.NumberColumn(format="$%,.0f"),
                    "System GMROI": st.column_config.NumberColumn(format="%.2f"),
                }
                st.dataframe(trend_display, use_container_width=True,
                              hide_index=True, column_config=config)

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
            port_data = compute_gmroi(sales, inventory, ["Brand"], annualize_factor)
            fig = build_scatter_chart(port_data, "Brand")
        else:
            port_data = compute_gmroi(sales, inventory, ["Product Category"], annualize_factor)
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

        brand_shop, brand_stats = compute_store_variance(sales, inventory,
                                                              annualize_factor)

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

    # ── TAB 8: Validation ──
    with tabs[7]:
        st.subheader("✅ Data Validation")
        st.markdown("""
Cross-reference tool data against external sources (accounting, Headset, Blaze, Distru)
to build confidence before using these numbers in presentations or reports.
        """)

        val_view = st.radio(
            "View:", ["Vendor Credits by Brand", "COGS Adjustments",
                      "Net Sales by Brand", "Net Sales by Store",
                      "Inventory Spot Check", "Data Freshness"],
            horizontal=True, key="val_view"
        )

        if val_view == "Vendor Credits by Brand":
            st.markdown("### Vendor Credits by Brand")
            st.caption("Compare these totals against what accounting billed to each vendor.")
            vc = sales.groupby("Brand").agg(
                Vendor_Credits=("Vendor_Pays", "sum"),
                Net_Sales=("Net Sales", "sum"),
                Transactions=("Trans No", "nunique"),
            ).reset_index()
            vc = vc[vc["Vendor_Credits"] > 0].sort_values("Vendor_Credits", ascending=False)
            vc["Credit_Pct_of_Sales"] = np.where(
                vc["Net_Sales"] > 0,
                vc["Vendor_Credits"] / vc["Net_Sales"] * 100, 0
            )
            config = {
                "Vendor_Credits": st.column_config.NumberColumn("Vendor Credits", format="$%,.0f"),
                "Net_Sales": st.column_config.NumberColumn("Net Sales", format="$%,.0f"),
                "Transactions": st.column_config.NumberColumn("Transactions", format="%,.0f"),
                "Credit_Pct_of_Sales": st.column_config.NumberColumn("Credits % of Sales", format="%.1f%%"),
            }
            st.dataframe(vc, use_container_width=True, hide_index=True,
                          column_config=config)
            st.markdown(f"**Total Vendor Credits:** ${vc['Vendor_Credits'].sum():,.0f}")

            # Download
            csv_buf = io.StringIO()
            vc.to_csv(csv_buf, index=False)
            st.download_button("📥 Download CSV", csv_buf.getvalue(),
                                "vendor_credits_by_brand.csv", "text/csv",
                                key="dl_val_vc")

        elif val_view == "COGS Adjustments":
            st.markdown("### COGS Adjustments by Brand")
            st.caption("Off-system credit memos applied outside the vendor credits CSV.")

            if "COGS_Adjustment" in sales.columns and sales["COGS_Adjustment"].sum() > 0:
                adj_sales = sales[sales["COGS_Adjustment"] > 0].copy()
                adj_sales["_Std_COGS"] = adj_sales["Unit Cost"] * adj_sales["Quantity Sold"]
                ca = adj_sales.groupby(
                    ["Brand", "Product Category"]
                ).agg(
                    COGS_Adjustment=("COGS_Adjustment", "sum"),
                    Standard_COGS=("_Std_COGS", "sum"),
                    Net_Sales=("Net Sales", "sum"),
                    Rows=("Trans No", "count"),
                ).reset_index()
                ca["Effective_Rate"] = np.where(
                    ca["Standard_COGS"] > 0,
                    ca["COGS_Adjustment"] / ca["Standard_COGS"] * 100, 0
                )
                ca = ca.sort_values("COGS_Adjustment", ascending=False)

                config = {
                    "COGS_Adjustment": st.column_config.NumberColumn("COGS Adj", format="$%,.0f"),
                    "Standard_COGS": st.column_config.NumberColumn("Std COGS", format="$%,.0f"),
                    "Net_Sales": st.column_config.NumberColumn("Net Sales", format="$%,.0f"),
                    "Rows": st.column_config.NumberColumn("Rows", format="%,.0f"),
                    "Effective_Rate": st.column_config.NumberColumn("Eff. Rate %", format="%.1f%%"),
                }
                st.dataframe(ca, use_container_width=True, hide_index=True,
                              column_config=config)
                st.markdown(f"**Total COGS Adjustments:** ${ca['COGS_Adjustment'].sum():,.0f}")

                # Show configured adjustments
                with st.expander("Configured Adjustment Rates"):
                    for brand, periods in COGS_ADJUSTMENTS.items():
                        st.markdown(f"**{brand}:**")
                        for p in periods:
                            cats = ", ".join(f"{k}: {v:.0%}" for k, v in p.get('categories', {}).items())
                            default = p.get('default', 0)
                            st.caption(
                                f"  {p['start']} to {p['end']}: {cats}"
                                f"{f', Default: {default:.0%}' if default > 0 else ''}"
                            )
                    if IGNORE_VENDOR_CREDITS_BRANDS:
                        st.caption(f"Brands excluded from vendor credits CSV: "
                                   f"{', '.join(IGNORE_VENDOR_CREDITS_BRANDS)}")

                # Show log from apply_cogs_adjustments
                if st.session_state.get('cogs_adjustment_log'):
                    with st.expander("Adjustment Application Log"):
                        for line in st.session_state['cogs_adjustment_log']:
                            st.caption(line)
            else:
                st.info("No COGS adjustments applied. Configure in COGS_ADJUSTMENTS "
                        "or use sidebar Stiiizy controls.")

        elif val_view == "Net Sales by Brand":
            st.markdown("### Net Sales by Brand (Top 30)")
            st.caption("Compare against Headset or Blaze brand reports.")
            brand_sales = sales.groupby("Brand").agg(
                Net_Sales=("Net Sales", "sum"),
                Qty_Sold=("Quantity Sold", "sum"),
                Transactions=("Trans No", "nunique"),
                Avg_Sell_Price=("Effective Retail Price", "mean"),
            ).reset_index().sort_values("Net_Sales", ascending=False).head(30)

            config = {
                "Net_Sales": st.column_config.NumberColumn("Net Sales", format="$%,.0f"),
                "Qty_Sold": st.column_config.NumberColumn("Qty Sold", format="%,.0f"),
                "Transactions": st.column_config.NumberColumn("Transactions", format="%,.0f"),
                "Avg_Sell_Price": st.column_config.NumberColumn("Avg Sell Price", format="$%,.2f"),
            }
            st.dataframe(brand_sales, use_container_width=True, hide_index=True,
                          column_config=config)
            st.markdown(f"**Total (top 30):** ${brand_sales['Net_Sales'].sum():,.0f} | "
                        f"**All brands:** ${sales['Net Sales'].sum():,.0f}")

            csv_buf = io.StringIO()
            brand_sales.to_csv(csv_buf, index=False)
            st.download_button("📥 Download CSV", csv_buf.getvalue(),
                                "net_sales_by_brand.csv", "text/csv",
                                key="dl_val_brand")

        elif val_view == "Net Sales by Store":
            st.markdown("### Net Sales by Store")
            st.caption("Compare against Blaze store-level reports.")
            store_sales = sales.groupby("Shop").agg(
                Net_Sales=("Net Sales", "sum"),
                Qty_Sold=("Quantity Sold", "sum"),
                Transactions=("Trans No", "nunique"),
                Brands=("Brand", "nunique"),
                Products=("Product", "nunique"),
            ).reset_index().sort_values("Net_Sales", ascending=False)

            config = {
                "Net_Sales": st.column_config.NumberColumn("Net Sales", format="$%,.0f"),
                "Qty_Sold": st.column_config.NumberColumn("Qty Sold", format="%,.0f"),
                "Transactions": st.column_config.NumberColumn("Transactions", format="%,.0f"),
                "Brands": st.column_config.NumberColumn("Brands", format="%d"),
                "Products": st.column_config.NumberColumn("Products", format="%d"),
            }
            st.dataframe(store_sales, use_container_width=True, hide_index=True,
                          column_config=config)
            st.markdown(f"**Total:** ${store_sales['Net_Sales'].sum():,.0f}")

            csv_buf = io.StringIO()
            store_sales.to_csv(csv_buf, index=False)
            st.download_button("📥 Download CSV", csv_buf.getvalue(),
                                "net_sales_by_store.csv", "text/csv",
                                key="dl_val_store")

        elif val_view == "Inventory Spot Check":
            st.markdown("### Inventory Spot Check")
            st.caption("Pick a snapshot date and filter to compare against Distru.")

            inv_dates = sorted(inventory["Date"].unique())
            if inv_dates:
                selected_date = st.selectbox(
                    "Snapshot date:",
                    [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime')
                     else str(d) for d in inv_dates],
                    index=len(inv_dates) - 1,
                    key="val_inv_date"
                )
                selected_dt = pd.Timestamp(selected_date)

                spot_inv = inventory[inventory["Date"] == selected_dt]

                col1, col2 = st.columns(2)
                with col1:
                    spot_brand = st.selectbox(
                        "Filter by brand (optional):",
                        ["All"] + sorted(spot_inv["Brand"].dropna().unique().tolist()),
                        key="val_inv_brand"
                    )
                with col2:
                    spot_shop = st.selectbox(
                        "Filter by store (optional):",
                        ["All"] + sorted(spot_inv["Shop"].dropna().unique().tolist()),
                        key="val_inv_shop"
                    )

                if spot_brand != "All":
                    spot_inv = spot_inv[spot_inv["Brand"] == spot_brand]
                if spot_shop != "All":
                    spot_inv = spot_inv[spot_inv["Shop"] == spot_shop]

                if len(spot_inv) > 0:
                    total_inv_value = spot_inv["Inventory Value"].sum()
                    total_qty = spot_inv["Quantity on Hand"].sum()
                    st.metric("Total Inventory Value", f"${total_inv_value:,.0f}")
                    st.metric("Total Qty on Hand", f"{total_qty:,.0f}")

                    spot_agg = spot_inv.groupby(["Shop", "Brand", "Product Category"]).agg(
                        Inventory_Value=("Inventory Value", "sum"),
                        Qty_on_Hand=("Quantity on Hand", "sum"),
                        SKU_Count=("Product Name", "nunique"),
                    ).reset_index().sort_values("Inventory_Value", ascending=False)

                    config = {
                        "Inventory_Value": st.column_config.NumberColumn("Inv Value", format="$%,.0f"),
                        "Qty_on_Hand": st.column_config.NumberColumn("Qty", format="%,.0f"),
                        "SKU_Count": st.column_config.NumberColumn("SKUs", format="%d"),
                    }
                    st.dataframe(spot_agg, use_container_width=True, hide_index=True,
                                  column_config=config)
                else:
                    st.info("No inventory data for selected filters.")
            else:
                st.info("No inventory snapshots loaded.")

        elif val_view == "Data Freshness":
            st.markdown("### Data Freshness")
            st.caption("Check that data files are current and complete.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Sales Data**")
                sales_files = sorted(glob.glob(os.path.join(SALES_DIR, "*.csv")))
                st.markdown(f"- Files: {len(sales_files)}")
                if sales_files:
                    st.markdown(f"- Oldest: `{os.path.basename(sales_files[0])}`")
                    st.markdown(f"- Newest: `{os.path.basename(sales_files[-1])}`")
                st.markdown(f"- Date range: {sales['Date'].min().date()} to {sales['Date'].max().date()}")
                st.markdown(f"- Total rows: {len(sales):,}")

            with col2:
                st.markdown("**Inventory Data**")
                inv_files = sorted(glob.glob(os.path.join(INVENTORY_DIR, "*.csv")))
                st.markdown(f"- Files: {len(inv_files)}")
                if inv_files:
                    st.markdown(f"- Oldest: `{os.path.basename(inv_files[0])}`")
                    st.markdown(f"- Newest: `{os.path.basename(inv_files[-1])}`")
                st.markdown(f"- Date range: {inventory['Date'].min().date()} to {inventory['Date'].max().date()}")
                st.markdown(f"- Snapshot dates: {inventory['Date'].nunique()}")

            st.markdown("---")
            st.markdown("**Vendor Credits**")
            import pathlib
            credits_files = sorted(glob.glob(CREDITS_PATTERN))
            if credits_files:
                for cf in credits_files:
                    cred_size = pathlib.Path(cf).stat().st_size / (1024 * 1024)
                    st.markdown(f"- File: `{os.path.basename(cf)}` ({cred_size:.1f} MB)")
                st.markdown(f"- Brands excluded: {', '.join(IGNORE_VENDOR_CREDITS_BRANDS) if IGNORE_VENDOR_CREDITS_BRANDS else 'None'}")
            else:
                st.warning("No vendor credits files found.")


    # ── TAB 9: Product Performance ──
    PP_CONFIG = {
        "Units_Sold": st.column_config.NumberColumn("Units Sold", format="%,.0f"),
        "Revenue": st.column_config.NumberColumn("Revenue", format="$%,.0f"),
        "Gross_Profit": st.column_config.NumberColumn("Gross Profit", format="$%,.0f"),
        "COGS": st.column_config.NumberColumn("COGS", format="$%,.0f"),
        "Margin_Pct": st.column_config.NumberColumn("Margin %", format="%.1f%%"),
        "Avg_Sell": st.column_config.NumberColumn("Avg Sell", format="$%,.2f"),
        "Avg_Cost": st.column_config.NumberColumn("Avg Cost", format="$%,.2f"),
        "Units_Share": st.column_config.NumberColumn("Units Share %", format="%.1f%%"),
        "Revenue_Share": st.column_config.NumberColumn("Rev Share %", format="%.1f%%"),
        "GP_Share": st.column_config.NumberColumn("GP Share %", format="%.1f%%"),
        "GMROI": st.column_config.NumberColumn("GMROI", format="%.2f"),
        "Inv_Turns": st.column_config.NumberColumn("Inv Turns", format="%.2f"),
        "Avg_Inv_Cost": st.column_config.NumberColumn("Avg Inv Cost", format="$%,.0f"),
    }

    with tabs[8]:
        st.subheader("Product Performance")

        has_templates = ("Profile_Template" in sales.columns
                         and (sales["Profile_Template"] != "Unmatched").any())

        pp_mode = st.radio(
            "Analysis mode:",
            ["SKU Type Comparison", "Within-Brand SKU Review",
             "Data Quality"],
            horizontal=True, key="pp_mode")

        if pp_mode == "SKU Type Comparison":
            st.caption(
                "Compare brands selling the same product type. "
                "Select Profile Templates to define the comparison group."
            )
            if not has_templates:
                st.warning("No Profile Template data. Rebuild from CSVs after "
                           "adding data/catalog/profile_templates.csv.")
            else:
                matched_sales = sales[sales["Profile_Template"] != "Unmatched"]

                # Category filter
                all_cats = sorted(matched_sales["Product Category"].dropna()
                                  .unique().tolist())
                sel_cat = st.selectbox("Category:", ["All"] + all_cats,
                                        key="pp_xb_cat")
                if sel_cat != "All":
                    matched_sales = matched_sales[
                        matched_sales["Product Category"] == sel_cat]

                # Template search + multiselect
                all_templates = sorted(
                    matched_sales["Profile_Template"].dropna().unique().tolist())
                search = st.text_input("Search templates:", "",
                                        key="pp_xb_search")
                if search:
                    filtered_templates = [t for t in all_templates
                                          if search.lower() in str(t).lower()]
                else:
                    filtered_templates = all_templates

                sel_templates = st.multiselect(
                    "Select templates to compare:",
                    filtered_templates, key="pp_xb_templates")

                if sel_templates:
                    # Review window
                    col_w1, col_w2 = st.columns(2)
                    review_days = col_w1.number_input(
                        "Review window (days):", 30, 365, 90, 30,
                        key="pp_xb_days")
                    anchor = col_w2.radio(
                        "Time anchor:", ["Last N days", "From first sale"],
                        horizontal=True, key="pp_xb_anchor")

                    # Filter to selected templates
                    comp_sales = matched_sales[
                        matched_sales["Profile_Template"].isin(sel_templates)]

                    # Apply time window
                    if anchor == "Last N days":
                        cutoff = comp_sales["Date"].max() - pd.Timedelta(
                            days=review_days)
                    else:
                        cutoff = comp_sales["Date"].min()
                    end_dt = cutoff + pd.Timedelta(days=review_days)
                    comp_sales = comp_sales[
                        (comp_sales["Date"] >= cutoff)
                        & (comp_sales["Date"] <= end_dt)]

                    # Filter inventory to same window + matching products
                    # Map Profile_Template onto inventory so groupby works
                    comp_products = comp_sales["Product"].unique()
                    inv_name_col = ("Product Name" if "Product Name"
                                    in inventory.columns else "Product")
                    comp_inv = inventory[
                        (inventory["Date"] >= cutoff)
                        & (inventory["Date"] <= end_dt)
                        & (inventory[inv_name_col].isin(comp_products))].copy()
                    prod_tmpl_map = (comp_sales[["Product", "Profile_Template"]]
                                    .drop_duplicates()
                                    .set_index("Product")["Profile_Template"])
                    comp_inv["Profile_Template"] = (
                        comp_inv[inv_name_col].map(prod_tmpl_map))

                    if len(comp_sales) == 0:
                        st.warning("No sales in selected window.")
                    else:
                        # Annualize against the review window, not the sidebar timeframe
                        comp_ann_factor = calc_annualize_factor(comp_sales)
                        st.markdown("---")

                        # ── Share charts ──
                        brand_agg = compute_share_metrics(
                            comp_sales, ["Brand", "Profile_Template"],
                            comp_ann_factor, inventory=comp_inv)

                        ch1, ch2 = st.columns(2)
                        with ch1:
                            fig = px.pie(brand_agg, values="Units_Sold",
                                         names="Brand", hole=0.4,
                                         title="Units Sold Share")
                            fig.update_layout(height=320, margin=dict(t=40, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                        with ch2:
                            fig = px.pie(brand_agg, values="Gross_Profit",
                                         names="Brand", hole=0.4,
                                         title="Gross Profit Share")
                            fig.update_layout(height=320, margin=dict(t=40, b=0))
                            st.plotly_chart(fig, use_container_width=True)

                        # ── GMROI + Turns bar chart ──
                        if "GMROI" in brand_agg.columns:
                            gmroi_data = brand_agg.dropna(
                                subset=["GMROI", "Inv_Turns"])
                            if not gmroi_data.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    y=gmroi_data["Brand"].astype(str),
                                    x=gmroi_data["GMROI"],
                                    name="GMROI", orientation='h'))
                                fig.add_trace(go.Bar(
                                    y=gmroi_data["Brand"].astype(str),
                                    x=gmroi_data["Inv_Turns"],
                                    name="Inv Turns", orientation='h'))
                                fig.update_layout(
                                    barmode='group', title="GMROI and Inv Turns by Brand",
                                    xaxis_title="", yaxis_title="",
                                    height=max(280, len(gmroi_data) * 50))
                                st.plotly_chart(fig, use_container_width=True)

                        # ── Brand summary table ──
                        st.markdown("### Brand Summary")
                        brand_cols = [
                            "Brand", "Profile_Template", "Units_Sold",
                            "Revenue", "Gross_Profit", "Margin_Pct",
                            "Avg_Inv_Cost", "GMROI", "Inv_Turns",
                            "Avg_Sell", "Avg_Cost",
                            "Units_Share", "GP_Share"]
                        avail = [c for c in brand_cols
                                 if c in brand_agg.columns]
                        st.dataframe(brand_agg[avail],
                                      use_container_width=True,
                                      hide_index=True,
                                      column_config=PP_CONFIG)

                        # ── Store breakdown: units vs % share ──
                        st.markdown("### Store Breakdown")
                        pivot = build_store_matrix(
                            comp_sales, "Brand", "Quantity Sold")
                        pct = pivot.drop(columns=["Total"]).div(
                            pivot["Total"], axis=0) * 100

                        store_view = st.radio(
                            "View:", ["Units", "% of Store Total"],
                            horizontal=True, key="pp_xb_store_view")

                        if store_view == "% of Store Total":
                            # Heatmap
                            fig = px.imshow(
                                pct.values,
                                x=[str(c) for c in pct.columns],
                                y=[str(i) for i in pct.index],
                                color_continuous_scale="YlOrRd",
                                labels=dict(color="% Share"),
                                title="Market Share by Store (%)",
                                text_auto=".1f")
                            fig.update_layout(
                                height=max(350, len(pct) * 40))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.dataframe(pivot, use_container_width=True)
                            # Stacked bar
                            brands_in_data = [c for c in pivot.columns
                                              if c != "Total"]
                            if brands_in_data:
                                fig = go.Figure()
                                for brand in brands_in_data:
                                    fig.add_trace(go.Bar(
                                        y=pivot.index.astype(str),
                                        x=pivot[brand],
                                        name=str(brand),
                                        orientation='h'))
                                fig.update_layout(
                                    barmode='stack',
                                    title="Units Sold by Store",
                                    xaxis_title="Units Sold",
                                    yaxis_title="",
                                    height=max(350, len(pivot) * 35))
                                st.plotly_chart(fig, use_container_width=True)

                        # ── Individual product drill-down ──
                        st.markdown("### Individual Products")
                        sel_drill_brand = st.selectbox(
                            "Drill into brand:",
                            sorted(comp_sales["Brand"].unique().tolist()),
                            key="pp_xb_drill")
                        drill_sales = comp_sales[
                            comp_sales["Brand"] == sel_drill_brand]
                        drill_products = drill_sales["Product"].unique()
                        drill_inv = comp_inv[
                            comp_inv[inv_name_col].isin(drill_products)
                        ] if len(comp_inv) > 0 else comp_inv
                        prod_agg = compute_share_metrics(
                            drill_sales, ["Product"], comp_ann_factor,
                            inventory=drill_inv)
                        prod_cols = [
                            "Product", "Units_Sold", "Revenue",
                            "Gross_Profit", "Margin_Pct",
                            "Avg_Inv_Cost", "GMROI", "Inv_Turns",
                            "Avg_Sell", "Avg_Cost", "Units_Share"]
                        avail_p = [c for c in prod_cols
                                   if c in prod_agg.columns]
                        st.dataframe(prod_agg[avail_p],
                                      use_container_width=True,
                                      hide_index=True,
                                      column_config=PP_CONFIG)
                else:
                    st.info("Select one or more templates above to run the "
                            "comparison.")

        elif pp_mode == "Within-Brand SKU Review":
            st.caption(
                "Compare SKU types within a single brand. "
                "Use when a product has no direct competitor."
            )
            if not has_templates:
                st.warning("No Profile Template data.")
            else:
                matched_sales = sales[sales["Profile_Template"] != "Unmatched"]
                all_brands = sorted(
                    matched_sales["Brand"].dropna().unique().tolist())
                sel_brand = st.selectbox("Brand:", all_brands,
                                          key="pp_sku_brand")
                brand_sales = matched_sales[
                    matched_sales["Brand"] == sel_brand]

                # Review window
                col_w1, col_w2 = st.columns(2)
                review_days = col_w1.number_input(
                    "Review window (days):", 30, 365, 90, 30,
                    key="pp_sku_days")
                anchor = col_w2.radio(
                    "Time anchor:", ["Last N days", "From first sale"],
                    horizontal=True, key="pp_sku_anchor")

                if anchor == "Last N days":
                    cutoff = brand_sales["Date"].max() - pd.Timedelta(
                        days=review_days)
                else:
                    cutoff = brand_sales["Date"].min()
                end_dt = cutoff + pd.Timedelta(days=review_days)
                brand_sales = brand_sales[
                    (brand_sales["Date"] >= cutoff)
                    & (brand_sales["Date"] <= end_dt)]

                # Filter inventory for this brand's products + window
                # Map Profile_Template onto inventory so groupby works
                brand_products = brand_sales["Product"].unique()
                inv_name_col = ("Product Name" if "Product Name"
                                in inventory.columns else "Product")
                brand_inv = inventory[
                    (inventory["Date"] >= cutoff)
                    & (inventory["Date"] <= end_dt)
                    & (inventory[inv_name_col].isin(brand_products))].copy()
                prod_tmpl_map = (brand_sales[["Product", "Profile_Template"]]
                                .drop_duplicates()
                                .set_index("Product")["Profile_Template"])
                brand_inv["Profile_Template"] = (
                    brand_inv[inv_name_col].map(prod_tmpl_map))

                if len(brand_sales) == 0:
                    st.warning("No sales in selected window.")
                else:
                    # Annualize against the review window, not the sidebar timeframe
                    brand_ann_factor = calc_annualize_factor(brand_sales)
                    st.markdown("---")

                    # ── Template summary with charts ──
                    tmpl_agg = compute_share_metrics(
                        brand_sales, ["Profile_Template"], brand_ann_factor,
                        inventory=brand_inv)

                    # Share donuts
                    if len(tmpl_agg) > 1:
                        ch1, ch2 = st.columns(2)
                        with ch1:
                            fig = px.pie(tmpl_agg, values="Units_Sold",
                                         names="Profile_Template", hole=0.4,
                                         title="Units Sold Share")
                            fig.update_layout(height=320,
                                              margin=dict(t=40, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                        with ch2:
                            fig = px.pie(tmpl_agg, values="Gross_Profit",
                                         names="Profile_Template", hole=0.4,
                                         title="Gross Profit Share")
                            fig.update_layout(height=320,
                                              margin=dict(t=40, b=0))
                            st.plotly_chart(fig, use_container_width=True)

                    # GMROI + Turns bar
                    if "GMROI" in tmpl_agg.columns:
                        gdata = tmpl_agg.dropna(subset=["GMROI", "Inv_Turns"])
                        if not gdata.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                y=gdata["Profile_Template"].astype(str),
                                x=gdata["GMROI"],
                                name="GMROI", orientation='h'))
                            fig.add_trace(go.Bar(
                                y=gdata["Profile_Template"].astype(str),
                                x=gdata["Inv_Turns"],
                                name="Inv Turns", orientation='h'))
                            fig.update_layout(
                                barmode='group',
                                title=f"{sel_brand}: GMROI and Inv Turns",
                                height=max(280, len(gdata) * 50))
                            st.plotly_chart(fig, use_container_width=True)

                    # Summary table
                    st.markdown("### SKU Type Summary")
                    tmpl_cols = [
                        "Profile_Template", "Units_Sold", "Revenue",
                        "Gross_Profit", "Margin_Pct",
                        "Avg_Inv_Cost", "GMROI", "Inv_Turns",
                        "Avg_Sell", "Avg_Cost",
                        "Units_Share", "Revenue_Share"]
                    avail_t = [c for c in tmpl_cols
                               if c in tmpl_agg.columns]
                    st.dataframe(tmpl_agg[avail_t],
                                  use_container_width=True,
                                  hide_index=True,
                                  column_config=PP_CONFIG)

                    # ── Store breakdown: units vs % share ──
                    st.markdown("### Store Breakdown")
                    pivot = build_store_matrix(
                        brand_sales, "Profile_Template", "Quantity Sold")
                    pct = pivot.drop(columns=["Total"]).div(
                        pivot["Total"], axis=0) * 100

                    store_view = st.radio(
                        "View:", ["Units", "% of Store Total"],
                        horizontal=True, key="pp_sku_store_view")

                    if store_view == "% of Store Total":
                        fig = px.imshow(
                            pct.values,
                            x=[str(c) for c in pct.columns],
                            y=[str(i) for i in pct.index],
                            color_continuous_scale="YlOrRd",
                            labels=dict(color="% Share"),
                            title="SKU Type Share by Store (%)",
                            text_auto=".1f")
                        fig.update_layout(
                            height=max(350, len(pct) * 40))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(pivot, use_container_width=True)

                    # ── Weekly trend ──
                    st.markdown("### Weekly Trend")
                    weekly = (brand_sales
                              .groupby([pd.Grouper(key="Date", freq="W"),
                                        "Profile_Template"],
                                       observed=True)
                              .agg(Units_Sold=("Quantity Sold", "sum"))
                              .reset_index())
                    if not weekly.empty:
                        fig = px.line(
                            weekly, x="Date", y="Units_Sold",
                            color="Profile_Template",
                            labels={"Profile_Template": "SKU Type",
                                    "Units_Sold": "Weekly Units Sold"},
                            title=f"{sel_brand}: Weekly Units Sold by SKU Type")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

        elif pp_mode == "Data Quality":
            st.caption("Audit matching, unmatched products, and vendor credits.")

            if "Profile_Template" not in sales.columns:
                st.warning("No Profile Template data.")
            else:
                # Match rate summary
                total = len(sales)
                matched = (sales["Profile_Template"] != "Unmatched").sum()
                unmatched_n = total - matched
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Rows", f"{total:,}")
                c2.metric("Matched", f"{matched:,} ({matched/total*100:.1f}%)")
                c3.metric("Unmatched", f"{unmatched_n:,} ({unmatched_n/total*100:.1f}%)")

                st.markdown("---")

                # Unmatched by brand
                st.markdown("### Unmatched Products by Brand")
                unmatched = sales[sales["Profile_Template"] == "Unmatched"]
                brand_counts = (unmatched.groupby("Brand", observed=True)
                                .agg(Rows=("Product", "size"),
                                     Products=("Product", "nunique"),
                                     Revenue=("Net Sales", "sum"))
                                .reset_index()
                                .sort_values("Rows", ascending=False))
                brand_counts_cfg = {
                    "Rows": st.column_config.NumberColumn("Rows", format="%,.0f"),
                    "Products": st.column_config.NumberColumn("Products", format="%d"),
                    "Revenue": st.column_config.NumberColumn("Revenue", format="$%,.0f"),
                }
                st.dataframe(brand_counts.head(30), use_container_width=True,
                              hide_index=True, column_config=brand_counts_cfg)

                # Drill into unmatched products
                st.markdown("### Unmatched Product Detail")
                um_brand = st.selectbox(
                    "Filter by brand:",
                    ["All"] + sorted([b for b in unmatched["Brand"].unique().tolist()
                                      if pd.notna(b)]),
                    key="pp_dq_brand")
                um_view = unmatched if um_brand == "All" else unmatched[
                    unmatched["Brand"] == um_brand]
                um_detail = (um_view.groupby(
                    ["Brand", "Product", "Product Category"], observed=True)
                    .agg(Units=("Quantity Sold", "sum"),
                         Revenue=("Net Sales", "sum"))
                    .reset_index()
                    .sort_values("Revenue", ascending=False))
                st.dataframe(um_detail.head(50), use_container_width=True,
                              hide_index=True, column_config={
                    "Units": st.column_config.NumberColumn(format="%,.0f"),
                    "Revenue": st.column_config.NumberColumn(format="$%,.0f"),
                })

                # Vendor Credits audit
                st.markdown("---")
                st.markdown("### Vendor Credits Audit")
                has_credits = ("Vendor_Pays" in sales.columns
                               and sales["Vendor_Pays"].sum() > 0)
                if has_credits:
                    credit_sales = sales[sales["Vendor_Pays"] > 0]
                    total_credits = credit_sales["Vendor_Pays"].sum()
                    total_haven = credit_sales["Haven_Pays"].sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Vendor Credits",
                              f"${total_credits:,.0f}")
                    c2.metric("Haven Pays", f"${total_haven:,.0f}")
                    c3.metric("Rows with Credits",
                              f"{len(credit_sales):,}")

                    # Credits by brand
                    credit_by_brand = (credit_sales
                        .groupby("Brand", observed=True)
                        .agg(Vendor_Credits=("Vendor_Pays", "sum"),
                             Haven_Pays=("Haven_Pays", "sum"),
                             Rows=("Product", "size"),
                             Net_Sales=("Net Sales", "sum"))
                        .reset_index()
                        .sort_values("Vendor_Credits", ascending=False))
                    credit_by_brand["Credit_Pct"] = (
                        credit_by_brand["Vendor_Credits"]
                        / credit_by_brand["Net_Sales"] * 100)
                    st.dataframe(credit_by_brand.head(20),
                                  use_container_width=True,
                                  hide_index=True, column_config={
                        "Vendor_Credits": st.column_config.NumberColumn(
                            "Vendor Credits", format="$%,.0f"),
                        "Haven_Pays": st.column_config.NumberColumn(
                            "Haven Pays", format="$%,.0f"),
                        "Rows": st.column_config.NumberColumn(format="%,.0f"),
                        "Net_Sales": st.column_config.NumberColumn(
                            "Net Sales", format="$%,.0f"),
                        "Credit_Pct": st.column_config.NumberColumn(
                            "Credit %", format="%.1f%%"),
                    })

                    # Check: credits on unmatched products
                    um_credits = credit_sales[
                        credit_sales["Profile_Template"] == "Unmatched"]
                    if len(um_credits) > 0:
                        st.warning(
                            f"**{len(um_credits):,} credit rows "
                            f"(${um_credits['Vendor_Pays'].sum():,.0f}) "
                            f"are on unmatched products.** These won't "
                            f"appear in Product Performance comparisons.")
                else:
                    st.info("No vendor credit data in current dataset.")


if __name__ == "__main__":
    main()
