import os
import sys
import numpy as np
import pandas as pd
from src.Phishing.exception import CustomException
from src.Phishing.logger import logging
from src.Phishing.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            encoder_path = os.path.join("artifacts", "target_enc.pkl")

            preprocessor = load_object(preprocessor_path)
            targetencoder = load_object(encoder_path)
            model = load_object(model_path)
            
            # Specify columns used during training
            numerical_cols = [
                'time_response', 'time_domain_activation', 'time_domain_expiration',
                'length_url', 'domain_length', 'directory_length', 'file_length', 'params_length'
            ]
            categorical_cols = [
                'qty_dot_domain', 'qty_slash_directory', 'qty_hyphen_file',
                'qty_percent_file', 'qty_hyphen_params', 'qty_underline_params',
                'qty_slash_params', 'qty_hashtag_params', 'email_in_url',
                'domain_in_ip', 'server_client_domain', 'tld_present_params',
                'domain_spf', 'tls_ssl_certificate', 'url_google_index',
                'domain_google_index', 'url_shortened'
            ]
            
            # Apply target encoding to high cardinality features
            high_ct = ['ttl_hostname', 'asn_ip']
            encoded_data = targetencoder.transform(features[high_ct])

            # Apply preprocessing to numerical and categorical features
            numerical_data = features[numerical_cols]
            categorical_data = features[categorical_cols]
            scaled_numerical_data = preprocessor.transform(numerical_data)
            encoded_categorical_data = preprocessor.transform(categorical_data)
            
            # Combine transformed numerical and categorical features
            scaled_data = np.hstack((scaled_numerical_data, encoded_categorical_data, encoded_data))
            
            # Make predictions using the model
            pred = model.predict(scaled_data)

            return pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 qty_dot_domain: int,
                 qty_slash_directory: int,
                 qty_hyphen_file: int,
                 qty_percent_file: int,
                 qty_hyphen_params: int,
                 qty_underline_params: int,
                 qty_slash_params: int,
                 qty_hashtag_params: int,
                 email_in_url: int,
                 domain_in_ip: int,
                 server_client_domain: int,
                 tld_present_params: int,
                 domain_spf: int,
                 tls_ssl_certificate: int,
                 url_google_index: int,
                 domain_google_index: int,
                 url_shortened: int,
                 time_response: float,
                 time_domain_activation: int,
                 time_domain_expiration: int,
                 length_url: int,
                 domain_length: int,
                 directory_length: int,
                 file_length: int,
                 params_length: int,
                 ttl_hostname: int,
                 asn_ip: int):
        self.qty_dot_domain = qty_dot_domain
        self.qty_slash_directory = qty_slash_directory
        self.qty_hyphen_file = qty_hyphen_file
        self.qty_percent_file = qty_percent_file
        self.qty_hyphen_params = qty_hyphen_params
        self.qty_underline_params = qty_underline_params
        self.qty_slash_params = qty_slash_params
        self.qty_hashtag_params = qty_hashtag_params
        self.email_in_url = email_in_url
        self.domain_in_ip = domain_in_ip
        self.server_client_domain = server_client_domain
        self.tld_present_params = tld_present_params
        self.domain_spf = domain_spf
        self.tls_ssl_certificate = tls_ssl_certificate
        self.url_google_index = url_google_index
        self.domain_google_index = domain_google_index
        self.url_shortened = url_shortened
        self.time_response = time_response
        self.time_domain_activation = time_domain_activation
        self.time_domain_expiration = time_domain_expiration
        self.length_url = length_url
        self.domain_length = domain_length
        self.directory_length = directory_length
        self.file_length = file_length
        self.params_length = params_length
        self.ttl_hostname = ttl_hostname
        self.asn_ip = asn_ip

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'qty_dot_domain': [self.qty_dot_domain],
                'qty_slash_directory': [self.qty_slash_directory],
                'qty_hyphen_file': [self.qty_hyphen_file],
                'qty_percent_file': [self.qty_percent_file],
                'qty_hyphen_params': [self.qty_hyphen_params],
                'qty_underline_params': [self.qty_underline_params],
                'qty_slash_params': [self.qty_slash_params],
                'qty_hashtag_params': [self.qty_hashtag_params],
                'email_in_url': [self.email_in_url],
                'domain_in_ip': [self.domain_in_ip],
                'server_client_domain': [self.server_client_domain],
                'tld_present_params': [self.tld_present_params],
                'domain_spf': [self.domain_spf],
                'tls_ssl_certificate': [self.tls_ssl_certificate],
                'url_google_index': [self.url_google_index],
                'domain_google_index': [self.domain_google_index],
                'url_shortened': [self.url_shortened],
                'time_response': [self.time_response],
                'time_domain_activation': [self.time_domain_activation],
                'time_domain_expiration': [self.time_domain_expiration],
                'length_url': [self.length_url],
                'domain_length': [self.domain_length],
                'directory_length': [self.directory_length],
                'file_length': [self.file_length],
                'params_length': [self.params_length],
                'ttl_hostname': [self.ttl_hostname],
                'asn_ip': [self.asn_ip]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    try:
        # Create a CustomData object with additional columns
        custom_data = CustomData(
            qty_dot_domain=2,
            qty_slash_directory=3,
            qty_hyphen_file=1,
            qty_percent_file=0,
            qty_hyphen_params=2,
            qty_underline_params=1,
            qty_slash_params=2,
            qty_hashtag_params=0,
            email_in_url=1,
            domain_in_ip=0,
            server_client_domain=1,
            tld_present_params=0,
            domain_spf=1,
            tls_ssl_certificate=1,
            url_google_index=1,
            domain_google_index=1,
            url_shortened=0,
            time_response=10.5,
            time_domain_activation=20.0,
            time_domain_expiration=30.0,
            length_url=25,
            domain_length=15,
            directory_length=8,
            file_length=10,
            params_length=5,
            ttl_hostname=3600,
            asn_ip=12345
        )
        # Convert custom data to dataframe
        df = custom_data.get_data_as_dataframe()
        print(df)
    except CustomException as e:
        logging.error(f"CustomException occurred: {e}")
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
