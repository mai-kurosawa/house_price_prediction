import os
import pandas as pd
import argparse
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


def get_args():
    parser = argparse.ArgumentParser('house price prediction')
    parser.add_argument('--data', required=True, help='dir path to read the train/test datasets')
    parser.add_argument('--pred', required=True, help='prediction file path')
    return parser.parse_args()


def load_data(dir_path, model_features, numerical_features, model_target):
    train_path = os.path.join(dir_path, 'train.csv')
    test_path = os.path.join(dir_path, 'test.csv')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    #Drop outliers
    dropindex_totalbsmtsq = train_df[train_df[numerical_features[3]] > 5499].index
    train_df.drop(dropindex_totalbsmtsq, inplace=True)
    x_full_train, y_full_train = train_df[model_features].to_numpy(), train_df[model_target].to_numpy()
    x_test = test_df[model_features].to_numpy()

    return x_full_train, y_full_train, x_test, test_df[['Id']]


def build_pipeline(numerical_features, categorical_features):
    numerical_col_len = len(numerical_features)
    numerical_cols = list(range(numerical_col_len))
    categorical_cols = list(range(numerical_col_len, numerical_col_len + len(categorical_features)))
    
    # Preprocess the numerical features
    numerical_processor = Pipeline([
        ('num_imputer', SimpleImputer(strategy='mean')),
        ('num_scaler', MinMaxScaler())
    ])

    # Preprocess the categorical features
    cateogrical_processor = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine all data preprocessors from above
    data_processor = ColumnTransformer([
        ('numerical_processing', numerical_processor, numerical_cols),
        ('categorical_processing', cateogrical_processor, categorical_cols)
    ])

    pipeline = Pipeline([
        ('data_processing', data_processor),
        ('model', XGBRegressor())
     ])

    return pipeline


def search_hyperparameter(pipeline):
    param_grid = {
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_depth': [10, 20, 30, 40, 50],
        'model__learning_rate': [1, 0.1, 0.01],
        'model__reg_alpha': [1, 0.1, 0.01]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, verbose=1, n_jobs=-1)

    return grid_search


def main(args):
    #Selected Features
    numerical_features = ['OverallQual', 'GrLivArea', 'GarageArea',
        'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
    categorical_features = ['MSZoning']
    model_features = numerical_features + categorical_features
    model_target = 'SalePrice'
    
    x_full_train, y_full_train, x_test, pred_df = \
        load_data(args.data, model_features, numerical_features, model_target)

    ###BUILD PIPELINE###
    pipeline = build_pipeline(numerical_features, categorical_features)

    ### SEARCH HYPERPARAMETER###
    grid_search = search_hyperparameter(pipeline)
    grid_search.fit(x_full_train, y_full_train)

    print(grid_search.best_params_)
    print(grid_search.best_score_)

    classifier = grid_search.best_estimator_
    classifier.fit(x_full_train, y_full_train)

    ###Test Classifier###
    test_predictions = classifier.predict(x_test)
    pred_df['SalePrice'] = test_predictions
    output_file_path = args.pred
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.pred, index=False)


if __name__ == '__main__':
    main(get_args())
