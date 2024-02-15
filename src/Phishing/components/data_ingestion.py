import sys
import shutil
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path  # Import Path
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr
from pandas.api.types import is_numeric_dtype
import boto3

from src.Phishing.logger import logging
from src.Phishing.exception import CustomException

class DataIngestionConfig:
    raw_data_dir = "artifacts"
    raw_data_file = "raw.csv"
    raw_data_path = os.path.join(raw_data_dir, raw_data_file)
    train_data_path = os.path.join(raw_data_dir, "train.csv")
    test_data_path = os.path.join(raw_data_dir, "test.csv")
    bucket_name = 'phishingdataset1'
    file_key = 'dynamodb.csv'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def cleanup_artifacts_folder(self):
        raw_csv_path = self.ingestion_config.raw_data_path
        if os.path.exists(raw_csv_path):
            os.remove(raw_csv_path)
            logging.info("Removed existing raw.csv file.")
    
    def create_empty_raw_csv(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.ingestion_config.raw_data_dir, exist_ok=True)
        
        # Create an empty raw.csv file if it doesn't exist
        raw_data_file_path = Path(self.ingestion_config.raw_data_path)
        if not raw_data_file_path.exists():
            raw_data_file_path.touch()  # Create an empty file
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        
        try:
            # Cleanup artifacts folder if exists
            self.cleanup_artifacts_folder()
            
            # Initialize the S3 client
            s3 = boto3.client('s3')
            
            # Create an empty raw.csv file
            self.create_empty_raw_csv()

            # Download the file
            s3.download_file(self.ingestion_config.bucket_name, self.ingestion_config.file_key, self.ingestion_config.raw_data_path)
            logging.info("Downloaded the dataset from S3")

            # Read the CSV file
            data = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Read dataset as a DataFrame")
            
            # Save the raw dataset
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Saved the raw dataset in artifact folder")
            
            # Preprocessing steps
                        
            # Drop 'id' column
            data = data.drop('id', axis=1)
            
            # Define columns for further processing
            categorical_columns = ['email_in_url','domain_in_ip','server_client_domain','tld_present_params','domain_spf','tls_ssl_certificate','url_google_index','domain_google_index','url_shortened']
            time_length = ['time_response','time_domain_activation','time_domain_expiration','length_url','domain_length','directory_length','file_length','params_length']
            high_ct = ['ttl_hostname','asn_ip']
            cols = categorical_columns + time_length
            
            # Calculate correlation matrix
            correlation_matrix = data.corr()
            mean_correlation = correlation_matrix.unstack().mean()
            median_correlation = correlation_matrix.unstack().median()
            logging.info(f"Mean Correlation: {mean_correlation:.4f}")
            logging.info(f"Median Correlation: {median_correlation:.4f}")

            # Remove highly correlated features
            copy_df = data.drop(cols,axis=1)
            significant_columns = []
            for col in copy_df.columns[:-1]:  
                if is_numeric_dtype(copy_df[col]):
                    correlation, pvalue = pearsonr(copy_df[col], copy_df['phishing'])
                    if pvalue <= 0.05 and np.abs(correlation) > mean_correlation:
                        significant_columns.append(col)
            copy_df = copy_df[significant_columns + ['phishing']]
            
            correlation_matrix = copy_df.corr()
            correlation_threshold = 0.8
            highly_correlated_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                        feature_i = correlation_matrix.columns[i]
                        feature_j = correlation_matrix.columns[j]
                        highly_correlated_pairs.append((feature_i, feature_j, correlation_matrix.iloc[i, j]))
            for pair in highly_correlated_pairs:
                logging.info(f"Correlation between {pair[0]} and {pair[1]}: {pair[2]:.4f}")
            features_to_remove = set()
            for feature_i, feature_j, correlation_value in highly_correlated_pairs:
                if abs(correlation_matrix.loc['phishing', feature_i]) > abs(correlation_matrix.loc['phishing', feature_j]):
                    features_to_remove.add(feature_j)
                else:
                    features_to_remove.add(feature_i)
            copy_df.drop(features_to_remove, axis=1, inplace=True)
            
            # Perform Isolation Forest outlier detection
            outlier_threshold = 0.05
            df1 = data[list(copy_df.columns[:-1]) + time_length]
            for column in df1.columns:
                X = df1[column].values.reshape(-1, 1)
                isolation_forest = IsolationForest(contamination=outlier_threshold, random_state=42)
                isolation_forest.fit(X)
                outliers = isolation_forest.predict(X)
                median_value = df1[column].median()
                df1.loc[outliers == -1, column] = median_value
            
            df_selected = data[categorical_columns + high_ct + ['phishing']]
            df_final = pd.concat([df1, df_selected], axis=1)
            
            # Perform train-test split
            train_data, test_data = train_test_split(df_final, test_size=0.30)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("Data ingestion part completed")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
            
        except Exception as e:
            logging.info("Exception occurred during data ingestion stage")
            raise CustomException(e,sys)

if __name__=="__main__":
    # Instantiate DataIngestion class
    data_ingestion = DataIngestion()
    
    # Call the method to initiate data ingestion
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    
    # Print paths of train and test data
    print("Train Data Path:", train_data_path)
    print("Test Data Path:", test_data_path)




