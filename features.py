"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


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



from sklearn.base import BaseEstimator, TransformerMixin
class GroupedConditionalImputer(BaseEstimator, TransformerMixin):

  def __init__(self, cond_col, category_col, target_cols, strategy='median'):
    self.cond_col = cond_col
    self.category_col = category_col
    self.target_cols = target_cols
    self.strategy = strategy

  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      X = pd.DataFrame(X)

    # Validate the strategy parameter
    if self.strategy not in ['median', 'mean']:
      raise ValueError("Strategy must be either 'median' or 'mean'")

    # Convert to list inside fit/transform to respect cloning rules
    target_cols_list = (
        [self.target_cols]
        if isinstance(self.target_cols, str)
        else list(self.target_cols)
    )

    # Dictionaries to store calculated stats per column and category
    self.group_stats_ = {}
    self.fallback_stats_ = {}

    # Filter where colA > 0 to isolate the rows used for summary calculations
    mask_gt_zero = X[self.cond_col] > 0
    df_gt_zero = X[mask_gt_zero]

    for col in self.target_cols:
      # 1. Compute summary statistic grouped by the category column
      if self.strategy == 'median':
        stats_series = df_gt_zero.groupby(self.category_col)[col].median()
        fallback_val = df_gt_zero[col].median()
      else:
        stats_series = df_gt_zero.groupby(self.category_col)[col].mean()
        fallback_val = df_gt_zero[col].mean()

      # Store values (convert mapping to dict for reliable lookups)
      self.group_stats_[col] = stats_series.to_dict()
      # Global backup value in case a totally unseen category shows up in transform()
      self.fallback_stats_[col] = (
          fallback_val if not pd.isna(fallback_val) else 0.0
      )

    return self

  def transform(self, X):
    X = X.copy()
    if not isinstance(X, pd.DataFrame):
      X = pd.DataFrame(X)

    mask_zero = X[self.cond_col] == 0
    mask_gt_zero = X[self.cond_col] > 0

    for col in self.target_cols:
      # --- Condition 1: colA == 0 -> Impute with 0 ---
      X.loc[mask_zero, col] = X.loc[mask_zero, col].fillna(0)

      # --- Condition 2: colA > 0 -> Impute with Category-specific Stat ---
      # Create a custom fill-series aligned with X's index using our mapped categories
      category_mapping = X.loc[mask_gt_zero, self.category_col].map(
          self.group_stats_[col]
      )
      # Apply fallback for safety
      category_mapping = category_mapping.fillna(self.fallback_stats_[col])

      # Apply only to the missing parts inside the targeted subset
      X.loc[mask_gt_zero, col] = X.loc[mask_gt_zero, col].fillna(
          category_mapping
        )

    return X


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
def get_preprocessor_mlp(track):
    """
    Creates a ColumnTransformer that applies StandardScaler to numerical features
    and OneHotEncoder to categorical features without distorting binary scales.
    """
    numeric_features = ['house_area', 'bedrooms', 'washrooms', 'parking_total', 'latitude', 'longitude', 'months']
    categorical_features = ['house_type_name', 'house_style', 'municipality', 'community']

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
    categorical_features = ['house_type_name', 'house_style', 'municipality', 'community']
    imputed_features = ['parking_total', 'house_type_name', 'lot_front', 'lot_depth']

    if track == 'list_aware':
        numeric_features.append('list_price')

    preprocessor = ColumnTransformer(
        transformers=[
            # ('num', StandardScaler(), numeric_features),
            # ('cond', GroupedConditionalImputer(cond_col="parking_total", category_col="house_type_name", target_cols=['lot_front', 'lot_depth'], strategy='median'), imputed_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def get_preprocessor_imputed(track):
    numeric_features = ['house_area', 'bedrooms', 'washrooms', 'parking_total', 'latitude', 'longitude', 'months']
    categorical_features = ['house_type_name', 'house_style', 'municipality', 'community']
    imputed_features = ['parking_total', 'house_type_name', 'lot_front', 'lot_depth']

    if track == 'list_aware':
        numeric_features.append('list_price')

    preprocessing_pipeline = Pipeline(
        steps=[
            ('cond', GroupedConditionalImputer(cond_col="parking_total", category_col="house_type_name", target_cols=['lot_front', 'lot_depth'], strategy='median')),
            ('col_trans', ColumnTransformer(
                transformers=[
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
                ],
                remainder='passthrough'
            ))
        ]
    )
    # preprocessing_pipeline.set_output(transform="pandas")
    return preprocessing_pipeline

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
def get_preprocessor_nlp(track):
    nltk_stop_words = list(stopwords.words("english"))

    numeric_features = ['house_area', 'bedrooms', 'washrooms', 'parking_total', 'latitude', 'longitude', 'months']
    categorical_features = ['house_type_name', 'house_style', 'municipality', 'community']
    imputed_features = ['parking_total', 'house_type_name', 'lot_front', 'lot_depth']

    if track == 'list_aware':
        numeric_features.append('list_price')

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(stop_words=nltk_stop_words, lowercase=True), "remarks"),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ],
        remainder='passthrough'
    )
    # preprocessing_pipeline.set_output(transform="pandas")
    return preprocessor

def feature_list_mlp_nlp_off_market(homes):
    X = feature_list_mlp_off_market(homes)
    X['remarks'] = homes['remarks']
    return X

def feature_list_mlp_nlp(homes):
    X = feature_list_mlp(homes)
    X['remarks'] = homes['remarks']
    return X

def feature_list_mlp(homes):
    """"Features to fit MLP"""
    X = feature_list_mlp_off_market(homes)
    X['list_price'] = homes['list_price']

    return X

def feature_list_mlp_off_market(homes):
    """"Features to fit MLP"""
    X = pd.DataFrame(index=homes.index)
    # X['list_price'] = homes['list_price']
    # X['days_on_market'] = homes['days_on_market']
    # X[new_house_type_cols] = encoded_house_type
    X['house_type_name'] = homes['house_type_name']
    X['house_style'] = homes['house_style']
    X['house_area'] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking_total"] = homes["parking_total"]
    # X["lot_front"] = homes["lot_front"]
    # X["lot_depth"] = homes["lot_depth"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X["municipality"] = homes["municipality"]
    X["community"] = homes["community"]
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44

    return X
