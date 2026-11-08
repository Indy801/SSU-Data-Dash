"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track.

New in this version:
- clean_sales():      automatic cleaning of the TRAINING sales (rules + residual outliers)
- recency_weights():  recent sales count more than old ones (half-life in months)
- RemarksScorer:      TF-IDF + ridge on the remarks, stacked out-of-fold as one feature
- GBMBlend:           LightGBM + CatBoost blended in log space, with all of the above.
                      Needs:  pip install lightgbm catboost
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# import features
import john.features as features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        # return ListPriceBaseline()
        # return RandomForestRegressorModel(features.feature_list_mlp, features.get_preprocessor_random_forest(track))
        return GBMBlend(track, features.gbm_features_list_aware)
    # return NearestSalesBaseline(k=10)
    # return RandomForestRegressorModel(features.feature_list_mlp_off_market, features.get_preprocessor_random_forest(track))
    return GBMBlend(track, features.gbm_features_off_market)


# ======================================================================================
# 1. Automatic cleaning of the training sales
# ======================================================================================
def clean_sales(train, track, z_cut=4.0, verbose=False):
    """Remove training sales that look like data errors. Never applied to homes you predict.

    Step 1, rules: duplicates, missing or tiny prices, negative days on market and,
    for the list-aware track, sold/list ratios outside 0.5-2.0.

    Step 2, residual outliers: a quick model predicts each sale's log price from its
    property, location and date, using 5-fold out-of-fold predictions (each sale is
    predicted by a model that never saw it). Sales whose miss is more than z_cut robust
    standard deviations from typical are dropped. This catches what rules can't, such
    as a correctly-formatted price that is wrong for that home.
    """
    df = train.copy()
    n_start = len(df)

    # ---- Step 1: rules --------------------------------------------------------------
    df = df.sort_values("date_sold").drop_duplicates("ml_num", keep="last")
    df = df[df["sold_price"].notna() & (df["sold_price"] > 50_000) & df["date_sold"].notna()]
    if "days_on_market" in df:
        df = df[~(df["days_on_market"] < 0)]
    if track == "list_aware":
        ratio = df["sold_price"] / df["list_price"]
        df = df[ratio.between(0.5, 2.0)]
    n_rules = len(df)

    # ---- Step 2: residual outliers --------------------------------------------------
    # Property, location and date only (no list price), so this is valid for both tracks.
    X = features.gbm_features(df, "off_market")
    for col in features.CATEGORICAL:
        X[col] = X[col].astype("category")
    y = np.log(df["sold_price"].to_numpy())
    quick = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                          categorical_features="from_dtype", random_state=0)
    # Random folds are fine here: this only flags bad rows inside the training data,
    # and the training data already ends before the homes being predicted.
    pred = cross_val_predict(quick, X, y, cv=KFold(5, shuffle=True, random_state=0))
    resid = y - pred
    centre = np.median(resid)
    spread = 1.4826 * np.median(np.abs(resid - centre))       # robust standard deviation
    keep = np.abs(resid - centre) <= z_cut * spread
    df = df[keep]

    if verbose:
        print(f"clean_sales[{track}]: {n_start} -> {n_rules} after rules -> {len(df)} after "
              f"residual filter (dropped {n_start - len(df)}, {100 * (n_start - len(df)) / n_start:.2f}%)")
    return df


# ======================================================================================
# 2. Recency sample weights
# ======================================================================================
def recency_weights(dates, half_life_months=12):
    """Weight each sale by 0.5 ** (age / half_life): a sale half_life months older than
    the newest sale counts half as much. None gives every sale the same weight.
    Weights are rescaled to average 1 so tree settings like min_child_weight keep meaning."""
    if half_life_months is None:
        return np.ones(len(dates))
    age_months = (dates.max() - dates).dt.days.to_numpy() / 30.44
    w = 0.5 ** (age_months / half_life_months)
    return w / w.mean()


# ======================================================================================
# 3. Remarks: TF-IDF + ridge, stacked as one feature
# ======================================================================================
def _oof_predict(X, y, seed=0, folds=5):
    """Out-of-fold predictions from a quick tree model: each row is predicted by a
    model that never saw it."""
    quick = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                          categorical_features="from_dtype", random_state=seed)
    return cross_val_predict(quick, X, y, cv=KFold(folds, shuffle=True, random_state=seed))


class RemarksScorer:
    """Turns each listing description into one number: how much more or less (in log
    price) the words suggest the home is worth than its property facts alone.

    TF-IDF counts words and 1-3 word phrases ("fully redone", "offered as-is", "den"),
    then a ridge regression learns a weight for each one. On training rows the scores
    are out-of-fold, so the trees never see a score that was fitted to that row's own
    price. Without that, the score would partly memorise the answer and the trees would
    trust it far more than they should.
    """

    def __init__(self, alpha=3.0, ngram_range=(1, 3), min_df=5, max_features=300_000,
                 folds=5, seed=0):
        self.alpha, self.ngram_range, self.min_df = alpha, ngram_range, min_df
        self.max_features, self.folds, self.seed = max_features, folds, seed

    @staticmethod
    def _texts(remarks):
        return remarks.astype("string").fillna("").astype(str)

    def fit_transform(self, remarks, target, weights=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        self.vec_ = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                                    ngram_range=self.ngram_range, min_df=self.min_df,
                                    max_features=self.max_features, dtype=np.float32)
        M = self.vec_.fit_transform(self._texts(remarks))
        target = np.asarray(target, dtype=float)
        w = np.ones(len(target)) if weights is None else np.asarray(weights)

        oof = np.zeros(len(target))
        for tr, va in KFold(self.folds, shuffle=True, random_state=self.seed).split(M):
            ridge = Ridge(alpha=self.alpha).fit(M[tr], target[tr], sample_weight=w[tr])
            oof[va] = ridge.predict(M[va])
        self.ridge_ = Ridge(alpha=self.alpha).fit(M, target, sample_weight=w)
        return oof

    def transform(self, remarks):
        return self.ridge_.predict(self.vec_.transform(self._texts(remarks)))

    def top_terms(self, n=15):
        """Phrases with the largest positive and negative weights. Use them to rank
        phrases, not as a % effect: TF-IDF values are scaled per listing."""
        terms = self.vec_.get_feature_names_out()
        order = np.argsort(self.ridge_.coef_)
        pick = np.r_[order[::-1][:n], order[:n]]
        return pd.DataFrame({"phrase": terms[pick], "weight": self.ridge_.coef_[pick].round(3),
                             "direction": ["up"] * n + ["down"] * n})


# ======================================================================================
# 4. LightGBM + CatBoost blend
# ======================================================================================
def _as_category(X, cats):
    """Text columns -> pandas category with the categories seen in training.
    Unseen categories at prediction time become missing, which the trees handle."""
    X = X.copy()
    for col, levels in cats.items():
        X[col] = pd.Categorical(X[col].astype("string"), categories=levels)
    return X


class _LGBMLearner:
    OBJECTIVES = {"l2": "regression", "l1": "regression_l1", "huber": "huber"}

    def __init__(self, objective, params, seed):
        self.objective, self.params, self.seed = objective, params or {}, seed

    def fit(self, X, y, w, cats):
        import lightgbm as lgb
        p = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=30,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 objective=self.OBJECTIVES[self.objective],
                 random_state=self.seed, n_jobs=-1, verbose=-1,
                 device_type='gpu')
        if self.objective == "huber":
            p["alpha"] = 0.1          # Huber threshold in log units: misses beyond ~10% count linearly
        p.update(self.params)
        self.model = lgb.LGBMRegressor(**p).fit(_as_category(X, cats), y, sample_weight=w)
        return self

    def predict(self, X, cats):
        return self.model.predict(_as_category(X, cats))


class _CatBoostLearner:
    LOSSES = {"l2": "RMSE", "l1": "MAE", "huber": "Huber:delta=0.1"}

    def __init__(self, objective, params, seed):
        self.objective, self.params, self.seed = objective, params or {}, seed

    @staticmethod
    def _prep(X, cats):
        X = X.copy()
        for col in cats:   # CatBoost wants plain text with no missing values
            X[col] = X[col].astype("string").fillna("NA").astype(object)
        return X

    def fit(self, X, y, w, cats):
        from catboost import CatBoostRegressor
        p = dict(iterations=1500, learning_rate=0.06, depth=6, l2_leaf_reg=3.0,
                 loss_function=self.LOSSES[self.objective], random_seed=self.seed,
                 verbose=0, thread_count=-1, allow_writing_files=False, task_type="GPU")
        p.update(self.params)
        self.model = CatBoostRegressor(**p)
        self.model.fit(self._prep(X, cats), y, sample_weight=w, cat_features=list(cats))
        return self

    def predict(self, X, cats):
        return self.model.predict(self._prep(X, cats))


class _HGBLearner:
    """sklearn's histogram GBM: no extra install, handy as a fallback or third member."""
    LOSSES = {"l2": "squared_error", "l1": "absolute_error", "huber": "absolute_error"}

    def __init__(self, objective, params, seed):
        self.objective, self.params, self.seed = objective, params or {}, seed

    def fit(self, X, y, w, cats):
        p = dict(max_iter=1000, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=30,
                 l2_regularization=1.0, loss=self.LOSSES[self.objective],
                 categorical_features="from_dtype", random_state=self.seed)
        p.update(self.params)
        self.model = HistGradientBoostingRegressor(**p).fit(_as_category(X, cats), y, sample_weight=w)
        return self

    def predict(self, X, cats):
        return self.model.predict(_as_category(X, cats))


class GBMBlend:
    """Gradient-boosted trees, one or several, blended in log space.

    Target:
      off_market  ->  log(sold_price)
      list_aware  ->  log(sold_price / list_price); the prediction is list_price * exp(pred)
    Both make errors relative (a 5% miss counts the same on any home), which matches MdAPE.

    learners:   any of "lgbm", "catboost", "hgb"
    weights:    blend weights per learner, e.g. {"lgbm": 0.6, "catboost": 0.4}; default equal
    half_life:  recency weighting in months (None = off)
    objective:  "l2", "l1" or "huber" (l1 aims at the median, which is what MdAPE scores)
    clean:      run clean_sales() on the training data first
    text:       how to use remarks: None, "flags" (keyword columns), "tfidf" (one stacked
                RemarksScorer score) or "both"
    text_alpha: ridge strength for the TF-IDF score (higher = smoother)
    """

    LEARNERS = {"lgbm": _LGBMLearner, "catboost": _CatBoostLearner, "hgb": _HGBLearner}
    TEXT_OPTIONS = (None, "flags", "tfidf", "both")

    def __init__(self, track, make_features, learners=("lgbm", "catboost"), weights=None,
                 half_life=12, objective="l2", clean=True, ratio_target=None,
                 text="both", text_alpha=3.0,
                 lgbm_params=None, catboost_params=None, hgb_params=None, seed=0, verbose=False):
        if text not in self.TEXT_OPTIONS:
            raise ValueError(f"text must be one of {self.TEXT_OPTIONS}")
        self.text, self.text_alpha = text, text_alpha
        self.track = track
        self.make_features = make_features
        self.learners = tuple(learners)
        self.weights = weights
        self.half_life = half_life
        self.objective = objective
        self.clean = clean
        self.ratio_target = (track == "list_aware") if ratio_target is None else ratio_target
        self.params = {"lgbm": lgbm_params, "catboost": catboost_params, "hgb": hgb_params}
        self.seed = seed
        self.verbose = verbose

    def fit(self, train):
        if self.clean:
            train = clean_sales(train, self.track, verbose=self.verbose)
        if self.ratio_target:
            train = train[train["list_price"] > 0]

        X = self.make_features(train)
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.cats_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in cat_cols}

        y = np.log(train["sold_price"].to_numpy())
        if self.ratio_target:
            y = y - np.log(train["list_price"].to_numpy())
        w = recency_weights(train["date_sold"], self.half_life)
        X = X.join(self._text_features(train, X, y, w))

        self.models_ = {}
        for name in self.learners:
            learner = self.LEARNERS[name](self.objective, self.params.get(name), self.seed)
            self.models_[name] = learner.fit(X, y, w, self.cats_)
        return self

    def _text_features(self, homes, X_tab=None, y=None, w=None):
        """Remarks -> keyword flags and/or the TF-IDF score. With y given (training) the
        scorer is fitted and returns out-of-fold scores; otherwise the fitted one is used."""
        out = pd.DataFrame(index=homes.index)
        if not self.text:
            return out
        remarks = homes["remarks"] if "remarks" in homes else pd.Series("", index=homes.index)
        if self.text in ("flags", "both"):
            out = out.join(features.remark_flags(remarks))
        if self.text in ("tfidf", "both"):
            if y is not None:
                # Target = what the property facts alone miss (out of fold), so the text
                # model spends its effort on condition and finishes, not on repeating
                # bedrooms and community names that are already columns.
                base = _oof_predict(_as_category(X_tab, self.cats_), y, self.seed)
                self.scorer_ = RemarksScorer(alpha=self.text_alpha, seed=self.seed)
                out["text_score"] = self.scorer_.fit_transform(remarks, y - base, w)
            else:
                out["text_score"] = self.scorer_.transform(remarks)
        return out

    def predict_components(self, homes):
        """Each learner's predicted sold price, before blending."""
        X = self.make_features(homes)
        X = X.join(self._text_features(homes))
        base = np.log(homes["list_price"].to_numpy()) if self.ratio_target else 0.0
        return {name: np.exp(m.predict(X, self.cats_) + base) for name, m in self.models_.items()}

    def blend(self, components):
        """Weighted geometric mean (= weighted average of log prices)."""
        w = self.weights or {name: 1.0 for name in components}
        total = sum(w[name] for name in components)
        log_price = sum(w[name] * np.log(p) for name, p in components.items()) / total
        return np.exp(log_price)

    def predict(self, homes):
        return self.blend(self.predict_components(homes))


# ======================================================================================
# Earlier models (unchanged)
# ======================================================================================
class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """Gradient-boosted trees (sklearn) on log(price), using a features.py function."""

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


class MLPModel:
    """MLP neural network on the features in feature_list_mlp."""

    def __init__(self, make_features, preprocessor, hidden_layer_sizes=(64, 32), alpha=0.0001,
                 max_iter=500, random_state=255):
        self.make_features = make_features
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('mlp', MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                                 solver='adam', alpha=alpha, max_iter=max_iter,
                                 random_state=random_state))
        ])

    def fit(self, train):
        self.pipeline.fit(self.make_features(train), train["sold_price"])
        return self

    def predict(self, homes):
        return self.pipeline.predict(self.make_features(homes))


class RandomForestRegressorModel:
    """Random Forest on the features in feature_list_mlp."""

    def __init__(self, make_features, preprocessor, n_estimators=100, random_state=42):
        self.make_features = make_features
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
        ])

    def fit(self, train):
        self.pipeline.fit(self.make_features(train), train["sold_price"])
        return self

    def predict(self, homes):
        return self.pipeline.predict(self.make_features(homes))
