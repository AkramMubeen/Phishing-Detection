# Import necessary modules
import os
import sys
import pandas as pd
from src.Phishing.logger import logging
from src.Phishing.exception import CustomException
from src.Phishing.components.data_ingestion import DataIngestion
from src.Phishing.components.data_transformation import DataTransformation
from src.Phishing.components.model_trainer import ModelTrainer
from src.Phishing.components.model_evaluation import ModelEvaluation

# Define the TrainingPipeline class
class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            # Initialize DataIngestion object
            data_ingestion = DataIngestion()
            # Start data ingestion
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path, test_data_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, train_data_path, test_data_path):
        try:
            # Initialize DataTransformation object
            data_transformation = DataTransformation()
            # Start data transformation
            train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
            return train_arr, test_arr
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):
        try:
            # Initialize ModelTrainer object
            model_trainer = ModelTrainer()
            # Start model training
            model_trainer.initiate_model_training(train_arr, test_arr)
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self, train_arr, test_arr):
        try:
            # Initialize ModelEvaluation object
            model_evaluation = ModelEvaluation()
            # Start model evaluation
            model_evaluation.initiate_model_evaluation(train_arr, test_arr)
        except Exception as e:
            raise CustomException(e, sys)

    def start_training_pipeline(self):
        try:
            # Start data ingestion
            train_data_path, test_data_path = self.start_data_ingestion()
            # Start data transformation
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            # Start model training
            self.start_model_training(train_arr, test_arr)
            # Start model evaluation
            self.start_model_evaluation(train_arr, test_arr)
        except Exception as e:
            raise CustomException(e, sys)

# Main block
if __name__ == "__main__":
    try:
        # Initialize the TrainingPipeline object
        pipeline = TrainingPipeline()
        # Start the training pipeline
        pipeline.start_training_pipeline()
    except CustomException as e:
        logging.error(f"CustomException occurred: {e}")
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
