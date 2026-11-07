"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.

New in this version:
- rotated_coords(): latitude/longitude measured along tilted axes, so tree models can
  follow diagonal neighbourhood boundaries with one split instead of a staircase.
- gbm_features(): the feature set used by GBMBlend in models.py. Categorical columns
  are returned as text; the model converts them itself.

The earlier feature functions (off_market_features, feature_list_mlp, ...) are unchanged.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


# --------------------------------------------------------------------------------------
# Rotated coordinates
# --------------------------------------------------------------------------------------
CENTER_LAT, CENTER_LON = 43.65, -79.38   # rough middle of the data
LON_SCALE = 0.72                         # cos(43.7 deg): makes a degree of longitude
                                         # the same length as a degree of latitude


def rotated_coords(lat, lon, angles=(15, 30, 45, 60, 75)):
    """Project each home onto axes tilted by each angle.

    A tree splits on one column at a time ("longitude < -79.4"), so every boundary it
    can draw is vertical or horizontal. A boundary that runs diagonally (a highway,
    a ravine, the edge of a pricey area) gets approximated by a staircase of many
    small splits. With a column measured along a tilted axis, the same diagonal
    boundary is a single split on that column.

    Each angle gives two axes (a and a + 90 degrees), so the default angles cover
    every 15 degrees.
    """
    x = (lon - CENTER_LON) * LON_SCALE   # east-west
    y = lat - CENTER_LAT                 # north-south
    out = pd.DataFrame(index=lat.index)
    for a in angles:
        t = np.radians(a)
        out[f"rot{a}_u"] = x * np.cos(t) + y * np.sin(t)
        out[f"rot{a}_v"] = -x * np.sin(t) + y * np.cos(t)
    return out


# --------------------------------------------------------------------------------------
# Features for the gradient-boosted models (GBMBlend in models.py)
# --------------------------------------------------------------------------------------
CATEGORICAL = ["house_type_name", "house_style", "municipality", "community"]


def gbm_features(homes, track):
    """Features for GBMBlend. track is "off_market" or "list_aware".

    Categorical columns stay as text. LightGBM, CatBoost and sklearn each get them
    converted to their own categorical format inside models.py.
    """
    X = pd.DataFrame(index=homes.index)
    for col in CATEGORICAL:
        X[col] = homes[col].astype("string")

    sqft = square_feet(homes["house_area"])
    X["square_feet"] = sqft.where((sqft >= 150) & (sqft <= 15000))   # impossible -> missing
    X["area_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(int)
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking_total"] = homes["parking_total"]
    X["garage"] = homes["garage"]
    X["lot_front"] = homes["lot_front"]          # empty for condos: fine for trees
    X["lot_depth"] = homes["lot_depth"]
    X["lot_area"] = homes["lot_front"] * homes["lot_depth"]

    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X = X.join(rotated_coords(homes["latitude"], homes["longitude"]))

    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44

    if track == "list_aware":
        X["log_list_price"] = np.log(homes["list_price"])
        X["list_price_per_sqft"] = homes["list_price"] / X["square_feet"]
    return X


# --------------------------------------------------------------------------------------
# Remarks: keyword flags (no fitting, so they can live here)
# --------------------------------------------------------------------------------------
# Condition tiers, best to worst, plus a few features the other columns don't record.
# Patterns are matched case-insensitively. Add phrases as you read more remarks.
REMARK_PATTERNS = {
    "rm_renovated": r"fully (?:redone|renovated|updated)|completely (?:redone|renovated)"
                    r"|redone (?:from )?top to bottom|(?:newly|recently|beautifully) renovated"
                    r"|turn-?key|quality finishes|nothing left to do",
    "rm_updated": r"\bupdated\b|refreshed|\bnewer\b|freshly painted|fresh paint|upgraded"
                  r"|modern light fixtures|new countertops",
    "rm_original": r"original finishes|finishes are original|clean and cared for|well[- ]cared[- ]for"
                   r"|well[- ]maintained|sound and liveable|original (?:kitchen|light fixtures)",
    "rm_needs_work": r"needs? (?:paint|work|updating|updates|significant)|requires updating"
                     r"|cosmetic overhaul|\btired\b|\bdated\b|worn flooring|older (?:appliances|windows)"
                     r"|could be updated|diamond in the rough|bring your contractor|sweat equity",
    "rm_as_is": r"\bas[- ]is\b|significant repair|end of (?:their|its) life|overgrown"
                r"|waterproofing|renovation project",
    "rm_motivated": r"motivated|priced to sell|must sell",
    "rm_den": r"\bden\b",
    "rm_bsmt_finished": r"finished basement|renovated (?:basement|bsmt)|basement apartment",
    "rm_bsmt_unfinished": r"unfinished basement",
    "rm_side_entrance": r"separate (?:side )?entrance",
    "rm_view": r"unobstructed view",
    "rm_locker": r"\blocker\b",
    "rm_amenities": r"concierge|\bgym\b|party room|guest suite|rooftop terrace",
}


def remark_flags(remarks):
    """One 0/1 column per pattern above, a rough condition score (+2 renovated to
    -2 as-is) and the word count."""
    text = remarks.astype("string").fillna("")
    out = pd.DataFrame(index=remarks.index)
    for name, pattern in REMARK_PATTERNS.items():
        out[name] = text.str.contains(pattern, case=False, regex=True).astype(int)
    out["rm_condition"] = (2 * out["rm_renovated"] + out["rm_updated"]
                           - out["rm_needs_work"] - 2 * out["rm_as_is"]).clip(-2, 2)
    out["rm_words"] = text.str.split().str.len().fillna(0)
    return out


def gbm_features_off_market(homes):
    return gbm_features(homes, "off_market")


def gbm_features_list_aware(homes):
    return gbm_features(homes, "list_aware")


# --------------------------------------------------------------------------------------
# Earlier feature functions (unchanged)
# --------------------------------------------------------------------------------------
def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X


def get_preprocessor_mlp(track):
    """
    Creates a ColumnTransformer that applies StandardScaler to numerical features
    and OneHotEncoder to categorical features without distorting binary scales.
    """
    numeric_features = ['house_area', 'bedrooms', 'washrooms', 'parking_total', 'latitude', 'longitude', 'months']
    categorical_features = ['house_type_name', 'house_style']

    if track == 'list_aware':
        numeric_features.append('list_price')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop=None, sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def get_preprocessor_random_forest(track):
    numeric_features = ['house_area', 'bedrooms', 'washrooms', 'parking_total', 'latitude', 'longitude', 'months']
    categorical_features = ['house_type_name', 'house_style']

    if track == 'list_aware':
        numeric_features.append('list_price')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def feature_list_mlp(homes):
    """"Features to fit MLP"""
    X = pd.DataFrame(index=homes.index)
    X['list_price'] = homes['list_price']
    X['house_type_name'] = homes['house_type_name']
    X['house_style'] = homes['house_style']
    X['house_area'] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking_total"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    return X


def feature_list_mlp_off_market(homes):
    """"Features to fit MLP"""
    X = pd.DataFrame(index=homes.index)
    X['house_type_name'] = homes['house_type_name']
    X['house_style'] = homes['house_style']
    X['house_area'] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking_total"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    return X
