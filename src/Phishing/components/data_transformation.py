import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.Phishing.exception import CustomException
from src.Phishing.logger import logging

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import yeojohnson

from src.Phishing.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    target_enc_obj_file_path = os.path.join('artifacts', 'target_enc.pkl')

def print_unique_values_in_all_columns(df):
    for column in df.columns:
        unique_values = df[column].unique()[:20]  # Limit to 20 values
        print(f"{column}: {list(unique_values)}")

def print_unique_values(column_name, df):
    unique_values = df[column_name].unique()[:20]  # Limit to 20 values
    print(f"{column_name}: {list(unique_values)}")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = [
                'qty_dot_domain', 'qty_slash_directory', 'qty_hyphen_file',
                'qty_percent_file', 'qty_hyphen_params', 'qty_underline_params',
                'qty_slash_params', 'qty_hashtag_params', 'email_in_url',
                'domain_in_ip', 'server_client_domain', 'tld_present_params',
                'domain_spf', 'tls_ssl_certificate', 'url_google_index',
                'domain_google_index', 'url_shortened'
            ]
            numerical_cols = [
                'time_response', 'time_domain_activation', 'time_domain_expiration',
                'length_url', 'domain_length', 'directory_length', 'file_length', 'params_length'
            ]
            
            logging.info('Pipeline Initiated')
            
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("onehotencoder", OneHotEncoder(handle_unknown='ignore'))])
            
        
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            return preprocessor

        except Exception as e:
            logging.error("Exception occurred during Data Transformation")
            raise CustomException(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            logging.info(print_unique_values_in_all_columns(train_df))
            
            preprocessing_obj = self.get_data_transformation()
            high_ct = ['ttl_hostname', 'asn_ip']
            target_column_name = 'phishing'
            
            X_train = train_df.drop(columns=target_column_name, axis=1)
            y_train = train_df[target_column_name]
            
            X_test = test_df.drop(columns=target_column_name, axis=1)
            y_test = test_df[target_column_name]
            
            numerical_cols = [
                'time_response', 'time_domain_activation', 'time_domain_expiration',
                'length_url', 'domain_length', 'directory_length', 'file_length', 'params_length'
            ]
            
            for col in numerical_cols:
                X_train[col], _ = yeojohnson(X_train[col])
                X_test[col], _ = yeojohnson(X_test[col])
            logging.info("Applied Power Transformation.")

            # Fit and transform the training set
            input_feature_train_arr = preprocessing_obj.fit_transform(X_train)
            
            # Transform the test set
            input_feature_test_arr = preprocessing_obj.transform(X_test)
            
            # Encoding high cardinality features
            encoder = TargetEncoder()
            X_train_encoded = encoder.fit_transform(X_train[high_ct], y_train)
            X_test_encoded = encoder.transform(X_test[high_ct])
            
            # Create DataFrames from transformed arrays
            X_train = pd.DataFrame(input_feature_train_arr, columns=preprocessing_obj.get_feature_names_out())
            X_test = pd.DataFrame(input_feature_test_arr, columns=preprocessing_obj.get_feature_names_out())
            
            # Add encoded high cardinality features to DataFrames
            X_train[high_ct] = X_train_encoded
            X_test[high_ct] = X_test_encoded
            
            logging.info("Applied preprocessing object on training and testing datasets.")
            
            # Convert to numpy arrays for compatibility
            train_arr = np.hstack((X_train.values, y_train.values.reshape(-1, 1)))
            test_arr = np.hstack((X_test.values, y_test.values.reshape(-1, 1)))

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing pickle file saved.")
            save_object(
                file_path=self.data_transformation_config.target_enc_obj_file_path,
                obj=encoder
            )
            logging.info("Target Encoder pickle file saved.")
            return train_arr, test_arr

        except Exception as e:
            logging.error("Exception occurred during Data Transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Specify paths to train and test data
        train_data_path = "artifacts/train.csv"
        test_data_path = "artifacts/test.csv"

        # Initialize DataTransformation object
        data_transformation = DataTransformation()

        # Perform data transformation
        train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

        # Print or further process the transformed data if needed
        print("Train array shape:", train_arr.shape)
        print("Test array shape:", test_arr.shape)

    except CustomException as e:
        logging.error(f"CustomException occurred: {e}")
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
