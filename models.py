"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from john.models import GBMBlend
from john.features import gbm_features

import features
from functools import partial


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        # return ListPriceBaseline()
        # return FeatureModel(features.list_aware_features)   # try this instead
        # return MLPModel(features.feature_list_mlp, features.get_preprocessor_mlp(track))
        # return RandomForestRegressorModel(features.feature_list_mlp, features.get_preprocessor_random_forest(track))
        # return XGBModel(features.feature_list_mlp, features.get_preprocessor_random_forest(track), learning_rate=0.1, n_estimators=500, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0)
        # return LGBModel(features.feature_list_mlp, features.get_preprocessor_random_forest(track), learning_rate=0.1, n_estimators=500, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0)
        return GBMBlend("list_aware", partial(gbm_features, track="list_aware"), learners=("lgbm",), objective='l1', text_alpha=3.0, 
            lgbm_params={'num_leaves': 63, 'min_child_samples': 10, 'learning_rate': 0.05, 'n_estimators': 1500, 'colsample_bytree': 0.5}
        )
    # return NearestSalesBaseline(k=10)
    # return FeatureModel(features.off_market_features)       # try this instead
    # return MLPModel(features.feature_list_mlp_off_market, features.get_preprocessor_mlp(track))
    # return RandomForestRegressorModel(features.feature_list_mlp_off_market, features.get_preprocessor_random_forest(track))
    # return XGBModel(features.feature_list_mlp_off_market, features.get_preprocessor_random_forest(track), learning_rate=0.2, n_estimators=500, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0)
    # return LGBModel(features.feature_list_mlp_off_market, features.get_preprocessor_random_forest(track), learning_rate=0.2, n_estimators=500, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0)
    return GBMBlend("off_market", partial(gbm_features, track="off_market"), learners=("lgbm",), objective='l1', text_alpha=3.0, 
            lgbm_params={'num_leaves': 63, 'min_child_samples': 10, 'learning_rate': 0.05, 'n_estimators': 1500, 'colsample_bytree': 0.5}
        )

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
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

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
    def __init__(self, make_features, preprocessor, hidden_layer_sizes=(64, 32), alpha=0.0001, max_iter=500, random_state=255):
        self.make_features = make_features

        # 1. Fetch the preprocessor blueprint from features.py
        # preprocessor = get_preprocessor()
        
        # 2. Build the inner scikit-learn Pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes, 
                activation='relu', 
                solver='adam', 
                alpha=alpha,
                max_iter=max_iter, 
                random_state=random_state
            ))
        ])

    def fit(self, train):
        self.pipeline.fit(self.make_features(train), train["sold_price"])
        return self

    def predict(self, homes):
        return self.pipeline.predict(self.make_features(homes))

from sklearn.base import BaseEstimator, RegressorMixin
class RandomForestRegressorModel(BaseEstimator, RegressorMixin):
    """Random Forest on the features in feature_list_mlp."""
    def __init__(self, make_features, preprocessor, n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=42):
        self.make_features = make_features

        # 1. Fetch the preprocessor blueprint from features.py
        # preprocessor = get_preprocessor()

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = max_depth
        self.preprocessor = preprocessor
        self.random_state = random_state
        
        # 2. Build the inner scikit-learn Pipeline
        # self.pipeline = Pipeline(steps=[
        #     ('preprocessor', preprocessor),
        #     ('regressor', RandomForestRegressor(
        #             n_estimators=n_estimators, 
        #             max_depth=max_depth, 
        #             min_samples_leaf=min_samples_leaf,
        #             random_state=random_state
        #         )
        #     )
        # ])
    
    # def fit(self, train):
    #     self.pipeline.fit(self.make_features(train), train["sold_price"])
    #     return self

    def fit(self, x, y=None, **fit_params):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                    n_estimators= self.n_estimators, 
                    max_depth= self.max_depth, 
                    min_samples_leaf= self.min_samples_leaf,
                    random_state= self.random_state
                )
            )
        ])

        self.pipeline.fit(self.make_features(x), y, **fit_params)
        return self

    def predict(self, x):
        return self.pipeline.predict(self.make_features(x))

from joblib import Memory
import tempfile
cachedir = tempfile.mkdtemp()
memory = Memory(location=cachedir, verbose=0)
class XGBModel(BaseEstimator, RegressorMixin):
    """Random Forest on the features in feature_list_mlp."""
    def __init__(self, make_features, preprocessor, 
        learning_rate=0.01,
        n_estimators=100,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.0,
        column_sample_percent=1.0,
        min_split_gain=0.0,
        random_state=42):
        self.make_features = make_features
        self.random_state = random_state
        self.preprocessor = preprocessor

        self.learning_rate = learning_rate
        self.max_iter = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.column_sample_percent = column_sample_percent
        self.min_split_gain = min_split_gain
        
    # def fit(self, x, y=None, **fit_params):
    #     self.pipeline = Pipeline(steps=[
    #         ('preprocessor', self.preprocessor),
    #         ('regressor', XGBRegressor(
    #                 tree_method="hist",                     # Uses the histogram approach
    #                 device="cuda",                          # Forces GPU execution
                    
    #                 learning_rate=self.learning_rate,       # Same name
    #                 n_estimators=self.max_iter,             # max_iter maps to n_estimators
    #                 max_leaves=self.max_leaf_nodes,         # max_leaf_nodes maps to max_leaves
    #                 min_child_weight=self.min_samples_leaf, # min_samples_leaf maps closely to min_child_weight
    #                 reg_lambda=self.l2_regularization       # l2_regularization maps to reg_lambda
    #             )
    #         )
    #     ], memory=memory)

    #     self.pipeline.fit(self.make_features(x), np.log(y), **fit_params)
    #     return self

    def fit(self, x):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(
                    tree_method="hist",                     # Uses the histogram approach
                    device="cuda",                          # Forces GPU execution
                    
                    learning_rate=self.learning_rate,       # Same name
                    n_estimators=self.n_estimators,             # max_iter maps to n_estimators
                    max_leaves=self.max_leaf_nodes,         # max_leaf_nodes maps to max_leaves
                    min_child_weight=self.min_samples_leaf, # min_samples_leaf maps closely to min_child_weight
                    reg_lambda=self.l2_regularization,       # l2_regularization maps to reg_lambda
                    colsample_bytree=self.column_sample_percent,
                    gamma=self.min_split_gain
                )
            )
        ], memory=memory)

        self.pipeline.fit(self.make_features(x), np.log(x['sold_price']))
        return self

    def predict(self, x):
        return np.exp(self.pipeline.predict(self.make_features(x)))

from lightgbm import LGBMRegressor
class LGBModel(BaseEstimator, RegressorMixin):
    """Random Forest on the features in feature_list_mlp."""
    def __init__(self, make_features, preprocessor, 
        learning_rate=0.01,
        n_estimators=100,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.0,
        column_sample_percent=1.0,
        min_split_gain=0.0,
        random_state=42):
        self.make_features = make_features
        self.random_state = random_state
        self.preprocessor = preprocessor

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.column_sample_percent = column_sample_percent
        self.min_split_gain = min_split_gain
        
    # def fit(self, x, y=None, **fit_params):
    #     self.pipeline = Pipeline(steps=[
    #         ('preprocessor', self.preprocessor),
    #         ('regressor', XGBRegressor(
    #                 device_type="gpu",
                    
                    # learning_rate=self.learning_rate,       # Same name
                    # n_estimators=self.n_estimators,             # max_iter maps to n_estimators
                    # num_leaves=self.max_leaf_nodes,         # max_leaf_nodes maps to max_leaves
                    # min_child_samples=self.min_samples_leaf, # min_samples_leaf maps closely to min_child_weight
                    # reg_lambda=self.l2_regularization,       # l2_regularization maps to reg_lambda
                    # colsample_bytree=1.0,
                    # min_split_gain=0.0
    #             )
    #         )
    #     ], memory=memory)

    #     self.pipeline.fit(self.make_features(x), np.log(y), **fit_params)
    #     return self

    def fit(self, x):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', LGBMRegressor(
                    device_type="gpu",
                    
                    learning_rate=self.learning_rate,       # Same name
                    n_estimators=self.n_estimators,             # max_iter maps to n_estimators
                    num_leaves=self.max_leaf_nodes,         # max_leaf_nodes maps to max_leaves
                    min_child_samples=self.min_samples_leaf, # min_samples_leaf maps closely to min_child_weight
                    reg_lambda=self.l2_regularization,       # l2_regularization maps to reg_lambda
                    colsample_bytree=self.column_sample_percent,
                    min_split_gain=self.min_split_gain
                )
            )
        ], memory=memory)

        self.pipeline.fit(self.make_features(x), np.log(x['sold_price']))
        return self

    def predict(self, x):
        return np.exp(self.pipeline.predict(self.make_features(x)))
