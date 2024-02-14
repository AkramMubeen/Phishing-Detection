import os
import sys
import pandas as pd
import numpy as np
from src.Phishing.logger import logging
from src.Phishing.exception import CustomException
from dataclasses import dataclass
from src.Phishing.utils.utils import save_object
from src.Phishing.utils.utils import evaluate_model

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'XGBoost': XGBClassifier(),
                'CatBoost': CatBoostClassifier()
            }

            # Define hyperparameters search spaces for GridSearchCV
            search_spaces = {
                'XGBoost': {'learning_rate': [1.0], 'max_depth': [10], 'n_estimators': [50]},
                'CatBoost': {'learning_rate': [1.0], 'depth': [10], 'n_estimators': [50]}
            }
            # search_spaces = {
            #     'XGBoost': {'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [1, 5, 10, 20], 'n_estimators': [50, 100, 500, 1000]},
            #     'CatBoost': {'learning_rate': [0.01, 0.1, 1.0], 'depth': [1, 5, 10, 20], 'n_estimators': [50, 100, 500, 1000]}
            # }
            logging.info('Parameters Initialized')

            # Perform grid search using GridSearchCV
            for model_name, model in models.items():
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=search_spaces[model_name],
                    cv=5,  # number of folds for cross-validation
                    n_jobs=-1  # number of CPU cores to use
                )
                grid_search.fit(X_train, y_train)
                models[model_name] = grid_search.best_estimator_

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get best model from model report
            best_model_name = max(model_report, key=lambda x: model_report[x]['accuracy'])
            best_model_scores = model_report[best_model_name]
            print("Best Model Name:", best_model_name)
            print("Best Model Scores:", best_model_scores)
            # Extract the accuracy score
            best_model_score = best_model_scores['accuracy']
            print("Best Model Accuracy Score:", best_model_score)
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
