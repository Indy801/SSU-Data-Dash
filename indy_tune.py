import data
import features
import models
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import os
import time


class LastNMonthsTimeSeriesSplit:
    def __init__(self, n_validation_months=6, max_train_months=None):
        """
        n_validation_months: Number of months at the end of the series to use as sequential test sets
        """
        self.n_validation_months = n_validation_months
        self.max_train_months = max_train_months

    def split(self, X, y=None, groups=None):
        """
        Using the 'groups' argument to safely pass dates ensures parallel 
        jobs (n_jobs > 1) map indices correctly.
        """
        if groups is None:
            raise ValueError("You must pass the date column/series to the 'groups' parameter of grid_search.fit()")
            
        # Convert groups to datetime safely
        dates = pd.to_datetime(groups)
        
        # 1. Identify unique Year-Months present in the dataset and sort them
        year_months = sorted(dates.dt.to_period('M').unique())
        
        # 2. Extract the newest N months earmarked for evaluation
        test_months = year_months[-self.n_validation_months:]
        
        # Convert date series to Period for alignment matching
        data_periods = dates.dt.to_period('M')
        
        # Use underlying integer positions independent of the dataframe's index
        indices = np.arange(len(X))
        
        # 3. Generate splits
        for current_test_month in test_months:
            test_mask = (data_periods == current_test_month).values
            if self.max_train_months is not None:
                start_train_month = current_test_month - self.max_train_months
                train_mask = ((data_periods < current_test_month) & (data_periods >= start_train_month)).values
            else:
                train_mask = (data_periods < current_test_month).values
            
            test_idx = indices[test_mask]
            train_idx = indices[train_mask]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_validation_months


def tune():
    start_time = time.time()
    listing = data.load("listings")
    listing_sorted = listing.sort_values('date_listed').reset_index(drop=True)

    X = listing_sorted.drop(columns='sold_price')
    X_om = listing_sorted.drop(columns=['sold_price', 'list_price'])
    Y = listing_sorted['sold_price']

    # Extract the alignment series to pass to 'groups'
    dates_vector = listing_sorted['date_listed']

    custom_cv = LastNMonthsTimeSeriesSplit(n_validation_months=1, max_train_months=6)

    # param_grid = {
    #     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
    #     'n_estimators': [100, 500, 1000, 1500],
    #     'max_leaf_nodes': [15, 31, 63, 127],
    #     'min_samples_leaf': [10, 20, 50],
    #     'l2_regularization': [0.0, 0.1, 1.0],
    #     'column_sample_percent': [0.6, 0.8, 1.0],
    #     'min_split_gain': [0.0, 0.1, 0.5],
    # }

    param_grid = {
        'learning_rate': [0.1, 0.2],
        'n_estimators': [300, 600, 1000],
        'max_leaf_nodes': [31, 63, 127],
        'min_samples_leaf': [25, 50],
        'l2_regularization': [0.8, 1.0, 1.2],
        'column_sample_percent': [0.6, 1],
        'min_split_gain': [0.0, 0.1],
    }

    grid_search_la = GridSearchCV(
        estimator=models.XGBModel(features.feature_list_mlp_nlp, features.get_preprocessor_nlp('list_aware'), random_state=42),
        param_grid=param_grid,
        cv=custom_cv,               # <--- Inject custom validation strategy here
        scoring='neg_mean_squared_error',
        n_jobs=10,
        verbose=1
    )

    grid_search_om = GridSearchCV(
        estimator=models.XGBModel(features.feature_list_mlp_nlp_off_market, features.get_preprocessor_nlp('off_market'), random_state=42),
        param_grid=param_grid,
        cv=custom_cv,               # <--- Inject custom validation strategy here
        scoring='neg_mean_squared_error',
        n_jobs=10,
        verbose=1
    )

    grid_search_la.fit(X, Y, groups=dates_vector)
    results_df_list_aware = pd.DataFrame(grid_search_la.cv_results_)

    print("Fitting list aware completed")

    grid_search_om.fit(X_om, Y, groups=dates_vector)
    results_df_off_market = pd.DataFrame(grid_search_om.cv_results_)

    print("Fitting off market completed")

    # Convert the negative MSE score into positive metrics
    results_df_list_aware['mean_mse'] = -results_df_list_aware['mean_test_score']
    results_df_list_aware['mean_rmse'] = np.sqrt(results_df_list_aware['mean_mse'])  # RMSE is in the original target units ($)
    results_df_off_market['mean_mse'] = -results_df_off_market['mean_test_score']
    results_df_off_market['mean_rmse'] = np.sqrt(results_df_off_market['mean_mse'])  # RMSE is in the original target units ($)

    results_df_list_aware_sorted = results_df_list_aware.sort_values(by='rank_test_score')
    results_df_off_market_sorted = results_df_off_market.sort_values(by='rank_test_score')

    
    # 4. Generate a unique, timestamped filename to prevent overwriting past runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_la_filename = f"grid_search_list_aware_results_{timestamp}.csv"
    output_om_filename = f"grid_search_off_market_results_{timestamp}.csv"

    # 5. Export to CSV
    results_df_list_aware_sorted.to_csv(output_la_filename, index=False)
    results_df_off_market_sorted.to_csv(output_om_filename, index=False)

    print(f"✅ Grid search results successfully saved to: {os.path.abspath(output_la_filename)}")
    print(f"✅ Grid search results successfully saved to: {os.path.abspath(output_om_filename)}")

    # 2. Capture the end time and calculate the duration
    end_time = time.time()
    total_duration_seconds = end_time - start_time
    
    # 3. Format into a clean, human-readable layout
    minutes, seconds = divmod(total_duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    print("\n==========================================")
    print(f"⏱️  TOTAL RUNTIME: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("==========================================")

    print("\n==========================================")
    print("Best Param list aware: ")
    print(results_df_list_aware_sorted['params'].iloc[0])
    print("Best Param off market: ")
    print(results_df_off_market_sorted['params'].iloc[0])
    print("==========================================")
tune()
