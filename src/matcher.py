"""
Product-to-Profile-Template matching engine.

Subset copy of the matching logic from price-checker (C:\\Users\\Charles\\github\\price-checker).
Matches sales product names to catalog Profile Templates (SKU Types).

Strategies tried in priority order:
1. Exact match (case-insensitive)
2. Placeholder pattern match (STRAIN, COLOR, FLAVOR, SIZE, VARIANT)
3. Wildcard regex match
4. Single-entry brand auto-match
5. Brand+Category auto-match
6. Advanced category-specific matching (Flower, Preroll, Vape, Extract)
"""

import re
import pandas as pd


# Brands that require exact or placeholder matching only (no auto-matching).
# These have multiple products per template that differ only by name, not structure.
EXACT_PRODUCT_MATCH_BRANDS = {
    'Blazy Susan', 'Camino', 'Crave', 'Daily Dose', "Dr. Norm's", 'Good Tide',
    'Happy Fruit', 'High Gorgeous', 'Kiva', 'Lost Farm', 'Made From Dirt',
    'Papa & Barkley', 'Sip Elixirs', 'St. Ides', "Uncle Arnie's", 'Vet CBD',
    'Wyld', 'Yummi Karma', "Not Your Father's",
}


# ---------------------------------------------------------------------------
# Helper: weight, pack size, keyword extraction
# ---------------------------------------------------------------------------

def extract_weight(item_text):
    """Extract weight from product name (e.g., 'Blue Dream 3.5g' -> '3.5g')."""
    if pd.isna(item_text):
        return None
    s = str(item_text).strip()

    # End-of-string patterns first (most reliable)
    end_patterns = [
        r'(\d+\.?\d*mg)$',
        r'(\d+\.\d+g)$',
        r'(\.\d+g)$',
        r'(\d+\.?\d*g)$',
        r'(\d+\.?\d*\s?oz?)$',
        r'(1/[248]\s?oz?)$',
    ]
    for pat in end_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).lower().replace(' ', '')

    # Anywhere patterns (fallback)
    any_patterns = [
        r'(\d+\.?\d*mg)',
        r'(\d+\.\d+g)',
        r'(\.\d+g)',
        r'(\d+\.?\d*g)',
        r'(\d+\.?\d*\s?oz?)',
        r'(1/[248]\s?oz?)',
    ]
    for pat in any_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).lower().replace(' ', '')
    return None


def extract_pack_size(item_text):
    """Extract pack size (e.g., '5pk 2.5g' -> '5pk')."""
    if pd.isna(item_text):
        return None
    s = str(item_text).strip()

    # Pack before weight
    for pat in [r'(\d+pk)\s+\d+\.?\d*g', r'(\d+pk)\s+\d+\s?oz',
                r'(\d+pk)\s+1/[248]\s?oz']:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    # Weight before pack
    for pat in [r'\d+\.?\d*g\s*(\d+pk)', r'\d+\s?oz\s*(\d+pk)',
                r'1/[248]\s?oz\s*(\d+pk)']:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    # Standalone
    m = re.search(r'(\d+pk)', s, re.IGNORECASE)
    return m.group(1).lower() if m else None


def extract_keywords(item_text, category):
    """Extract category-specific keywords from a product/template name."""
    if pd.isna(item_text) or pd.isna(category):
        return []
    s = str(item_text).lower()
    cat = str(category).lower()

    # Normalize flower subcategories
    if 'flower' in cat:
        cat = 'flower'

    if cat == 'vape':
        kws = ['originals', 'ascnd', 'dna', 'exotics', 'disposable',
               'live resin', 'reload', 'rtu', 'curepen', 'curebar',
               'liquid diamonds', 'melted diamonds', 'aio', 'lipstick']
        return [k for k in kws if k in s]

    if cat == 'flower':
        kws = ['top shelf', 'headstash', 'exotic', 'premium',
               'private reserve', 'reserve', 'smalls']
        return [k for k in kws if k in s]

    if cat == 'extract':
        found = []
        if 'live rosin' in s:
            found.append('live rosin')
        elif 'live resin' in s:
            found.append('live resin')
        elif 'hash rosin' in s:
            found.append('hash rosin')
        elif 'rosin' in s:
            found.append('rosin')
        elif 'resin' in s:
            found.append('resin')
        tier = re.search(r'tier\s*([1-4])', s)
        if tier:
            found.append(f"tier {tier.group(1)}")
        for mod in ['cold cure', 'fresh press', 'curated', 'hte blend']:
            if mod in s:
                found.append(mod)
        for con in ['diamonds', 'budder', 'badder', 'sauce', 'sugar', 'jam']:
            if con in s:
                found.append(con)
        for pt in ['rso', 'syringe']:
            if pt in s:
                found.append(pt)
        return found

    if cat == 'preroll':
        found = []
        for pt in ['blunts', 'preroll', 'prerolls', 'joints', 'mini', 'cannon']:
            if pt in s:
                found.append(pt)
        if 'infused' in s:
            found.append('infused')
        return found

    return []


# ---------------------------------------------------------------------------
# Pattern matching functions
# ---------------------------------------------------------------------------

def _match_placeholder(product_name, template_name):
    """Check if product matches a template with STRAIN/COLOR/FLAVOR/SIZE/VARIANT."""
    if pd.isna(product_name) or pd.isna(template_name):
        return False

    placeholders = ['STRAIN', 'COLOR', 'FLAVOR', 'SIZE', 'VARIANT']
    template_upper = str(template_name).upper()
    if not any(ph in template_upper for ph in placeholders):
        return False

    product_upper = str(product_name).upper()

    # Turn Up/Turn Down pattern
    if "TURN UP/TURN DOWN" in template_upper and "TURN" in product_upper:
        if "TURN UP" in product_upper:
            template_upper = template_upper.replace("TURN UP/TURN DOWN", "TURN UP")
        elif "TURN DOWN" in product_upper:
            template_upper = template_upper.replace("TURN UP/TURN DOWN", "TURN DOWN")

    # Stiiizy bag pattern
    if 'STIIIZY' in product_upper and 'STIIIZY' in template_upper and 'STRAIN' in template_upper:
        bag_types = ['BLACK BAG', 'WHITE BAG', 'BLUE BAG', 'GOLD BAG',
                     'SILVER BAG', 'PURPLE BAG']
        for bag in bag_types:
            if bag in product_upper and bag in template_upper:
                prod_after = product_upper.split('STIIIZY -')[-1].strip()
                tmpl_after = template_upper.split('STIIIZY -')[-1].strip()
                if tmpl_after.startswith('STRAIN ' + bag):
                    pw = re.search(r'(\d+\.?\d*G)', prod_after)
                    tw = re.search(r'(\d+\.?\d*G)', tmpl_after)
                    if pw and tw and pw.group(1) == tw.group(1) and bag in prod_after:
                        return True

    # Standard placeholder matching
    for ph in placeholders:
        if ph in template_upper:
            parts = template_upper.split(ph)
            if len(parts) != 2:
                continue
            prefix, suffix = parts
            if product_upper.startswith(prefix) and product_upper.endswith(suffix):
                val = product_upper[len(prefix):-len(suffix) if suffix else len(product_upper)]
                if val and 0 < len(val.strip()) < 50:
                    return True
    return False


def _match_wildcard(product_name, template_name,
                    wildcards=('COLOR', 'STRAIN', 'FLAVOR')):
    """Regex wildcard matching. Returns (matched: bool, extracted: dict)."""
    if pd.isna(product_name) or pd.isna(template_name):
        return False, {}

    item_str = str(product_name).strip()
    tmpl_str = str(template_name).strip()

    positions = {w: tmpl_str.find(w) for w in wildcards if w in tmpl_str}
    if not positions:
        return False, {}

    pattern = re.escape(tmpl_str)
    for w in wildcards:
        esc = re.escape(w)
        if esc in pattern:
            pattern = pattern.replace(esc, r'([^,]+?)', 1)

    m = re.match(pattern + r'\s*$', item_str, re.IGNORECASE)
    if m:
        ordered = sorted(positions.items(), key=lambda x: x[1])
        extracted = {}
        for i, (w, _) in enumerate(ordered, 1):
            if i <= len(m.groups()):
                extracted[w] = m.group(i).strip()
        return True, extracted
    return False, {}


# ---------------------------------------------------------------------------
# Category-specific advanced matchers
# ---------------------------------------------------------------------------

def _match_flower(product, templates, category):
    """Weight + quality-tier keyword matching for flower."""
    current = list(templates)
    steps = []
    weight = extract_weight(product)
    if weight:
        filtered = [t for t in current if extract_weight(t) == weight]
        if filtered:
            current = filtered
            steps.append(f"weight: {weight}")

    if len(current) > 1:
        prod_kws = extract_keywords(product, category)
        if prod_kws:
            scores = []
            for t in current:
                t_kws = extract_keywords(t, category)
                score = sum(1 for k in prod_kws if k in t_kws)
                scores.append((t, score, len(t_kws)))
            best = max(s[1] for s in scores)
            if best > 0:
                top = [(t, sc, tk) for t, sc, tk in scores if sc == best]
                if len(top) == 1:
                    current = [top[0][0]]
                    steps.append(f"keywords: {', '.join(prod_kws)}")
                else:
                    min_kw = min(tk for _, _, tk in top)
                    final = [t for t, _, tk in top if tk == min_kw]
                    if len(final) == 1:
                        current = final
                        steps.append(f"keywords: {', '.join(prod_kws)} (tiebreaker)")

    return (current[0], steps) if len(current) == 1 else (None, [])


def _match_preroll(product, templates, category):
    """Infused -> pack size -> weight -> type keywords for prerolls."""
    has_infused = 'infused' in str(product).lower()
    filtered = [t for t in templates
                if ('infused' in str(t).lower()) == has_infused]
    current = filtered if filtered else list(templates)
    steps = []
    if filtered:
        steps.append(f"infused: {'yes' if has_infused else 'no'}")

    # Pack size
    pack = extract_pack_size(product)
    if pack and len(current) > 1:
        matched = [t for t in current if extract_pack_size(t) == pack]
        if matched:
            current = matched
            steps.append(f"pack: {pack}")

    # Weight
    weight = extract_weight(product)
    if weight and len(current) > 1:
        matched = [t for t in current if extract_weight(t) == weight]
        if matched:
            current = matched
            steps.append(f"weight: {weight}")

    # No pack fallback
    if not pack and len(current) > 1:
        no_pack = [t for t in current if not extract_pack_size(t)]
        if len(no_pack) == 1:
            current = no_pack
            steps.append("no pack (fallback)")

    # Type keywords (excluding 'infused')
    if len(current) > 1:
        prod_kws = [k for k in extract_keywords(product, category) if k != 'infused']
        if prod_kws:
            scores = []
            for t in current:
                t_kws = [k for k in extract_keywords(t, category) if k != 'infused']
                score = sum(1 for k in prod_kws if k in t_kws)
                scores.append((t, score, len(t_kws)))
            best = max(s[1] for s in scores)
            if best > 0:
                top = [(t, sc, tk) for t, sc, tk in scores if sc == best]
                if len(top) == 1:
                    current = [top[0][0]]
                    steps.append(f"type: {', '.join(prod_kws)}")
                else:
                    min_kw = min(tk for _, _, tk in top)
                    final = [t for t, _, tk in top if tk == min_kw]
                    if len(final) == 1:
                        current = final
                        steps.append(f"type: {', '.join(prod_kws)} (tiebreaker)")

    return (current[0], steps) if len(current) == 1 else (None, [])


def _match_vape_extract(product, templates, category):
    """Weight + keywords for vape and extract products."""
    current = list(templates)
    steps = []

    weight = extract_weight(product)
    if weight and len(current) > 1:
        matched = [t for t in current if extract_weight(t) == weight]
        if matched:
            current = matched
            steps.append(f"weight: {weight}")

    prod_kws = extract_keywords(product, category)
    if prod_kws and len(current) > 1:
        scores = []
        for t in current:
            t_kws = extract_keywords(t, category)
            score = sum(1 for k in prod_kws if k in t_kws)
            scores.append((t, score, len(t_kws)))
        best = max(s[1] for s in scores)
        if best > 0:
            top = [(t, sc, tk) for t, sc, tk in scores if sc == best]
            if len(top) == 1:
                current = [top[0][0]]
                steps.append(f"keywords: {', '.join(prod_kws)}")
            else:
                min_kw = min(tk for _, _, tk in top)
                final = [t for t, _, tk in top if tk == min_kw]
                if len(final) == 1:
                    current = final
                    steps.append(f"keywords: {', '.join(prod_kws)} (tiebreaker)")
    elif not prod_kws and len(current) > 1:
        no_kw = [t for t in current if not extract_keywords(t, category)]
        if len(no_kw) == 1:
            current = no_kw
            steps.append("no keywords (fallback)")

    return (current[0], steps) if len(current) == 1 else (None, [])


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _normalize_category(cat):
    """Normalize 'Flower (Indica)' etc. to base category for catalog lookup."""
    if pd.isna(cat):
        return 'Unknown'
    c = str(cat).strip()
    if c.startswith('Flower'):
        return 'Flower'
    return c


def match_products_to_templates(products_df, catalog_df):
    """
    Match unique products to Profile Templates.

    Args:
        products_df: DataFrame with [Product, Brand, Product Category]
        catalog_df: DataFrame with [Brand, Profile Template, Category]

    Returns:
        dict mapping (Product, Brand) -> Profile Template (or 'Unmatched')
    """
    # Build lookup maps
    brand_map = {}          # brand -> [templates]
    brand_cat_map = {}      # "brand|category" -> [templates]

    for _, row in catalog_df.iterrows():
        brand = row['Brand']
        template = row['Profile Template']
        category = row['Category']
        if pd.isna(brand) or pd.isna(template):
            continue
        brand_map.setdefault(brand, [])
        if template not in brand_map[brand]:
            brand_map[brand].append(template)
        key = f"{brand}|{category}"
        brand_cat_map.setdefault(key, [])
        if template not in brand_cat_map[key]:
            brand_cat_map[key].append(template)

    # Pre-compute single-entry lookups
    single_brand = {b: ts[0] for b, ts in brand_map.items()
                    if len(ts) == 1 and b not in EXACT_PRODUCT_MATCH_BRANDS}
    single_brand_cat = {k: ts[0] for k, ts in brand_cat_map.items()
                        if len(ts) == 1
                        and k.split('|')[0] not in EXACT_PRODUCT_MATCH_BRANDS}

    result = {}

    for _, row in products_df.iterrows():
        product = row['Product']
        brand = row['Brand']
        raw_cat = row['Product Category']
        cat = _normalize_category(raw_cat)

        if pd.isna(brand) or pd.isna(product):
            result[(product, brand)] = 'Unmatched'
            continue

        matched = None

        # Strategy 1: Exact match
        if brand in brand_map:
            for tmpl in brand_map[brand]:
                if str(product).lower() == str(tmpl).lower():
                    matched = tmpl
                    break

        # Strategy 2: Placeholder pattern (category-aware)
        if not matched and brand in brand_map:
            bck = f"{brand}|{cat}"
            templates = brand_cat_map.get(bck, [])
            candidates = []
            for tmpl in templates:
                if _match_placeholder(product, tmpl):
                    prod_words = set(str(product).lower().split())
                    tmpl_words = set(str(tmpl).lower().split())
                    phs = {'strain', 'color', 'flavor', 'size', 'variant'}
                    score = len((tmpl_words - phs) & prod_words)
                    candidates.append((tmpl, score))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                matched = candidates[0][0]

        # Strategy 3: Wildcard regex (category-aware)
        if not matched and brand in brand_map:
            bck = f"{brand}|{cat}"
            templates = brand_cat_map.get(bck, [])
            candidates = []
            for tmpl in templates:
                ok, _ = _match_wildcard(product, tmpl)
                if ok:
                    prod_words = set(str(product).lower().split())
                    tmpl_words = set(str(tmpl).lower().split())
                    phs = {'strain', 'color', 'flavor', 'size', 'variant'}
                    score = len((tmpl_words - phs) & prod_words)
                    candidates.append((tmpl, score))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                matched = candidates[0][0]

        skip_auto = brand in EXACT_PRODUCT_MATCH_BRANDS

        # Strategy 4: Single-entry brand auto-match
        if not matched and not skip_auto and brand in single_brand:
            matched = single_brand[brand]

        # Strategy 5: Brand+Category auto-match
        if not matched and not skip_auto:
            bck = f"{brand}|{cat}"
            if bck in single_brand_cat:
                matched = single_brand_cat[bck]

        # Strategy 6: Advanced category-specific matching
        if not matched and cat in ('Flower', 'Preroll', 'Vape', 'Extract'):
            bck = f"{brand}|{cat}"
            templates = brand_cat_map.get(bck, [])
            if len(templates) > 1:
                if cat == 'Flower':
                    matched, _ = _match_flower(product, templates, cat)
                elif cat == 'Preroll':
                    matched, _ = _match_preroll(product, templates, cat)
                elif cat in ('Vape', 'Extract'):
                    matched, _ = _match_vape_extract(product, templates, cat)

        if not matched:
            matched = _auto_template(product, brand, cat)

        result[(product, brand)] = matched if matched else 'Unmatched'

    return result


def _auto_template(product, brand, category):
    """Generate a synthetic template for unmatched products.

    Groups by Brand + normalized category + weight so unmatched products
    still roll up meaningfully in Product Performance views.
    E.g. "Coastal Cowboys - Blueberry Kush 28g" -> "Coastal Cowboys - Flower 28g"
    """
    if pd.isna(brand) or pd.isna(product):
        return None

    weight = extract_weight(product)
    pack = extract_pack_size(product)

    parts = [str(brand), "-", category]
    if pack:
        parts.append(pack)
    if weight:
        parts.append(weight)

    return " ".join(parts)
