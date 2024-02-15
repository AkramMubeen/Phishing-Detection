from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from src.Phishing.logger import logging
from src.Phishing.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

# APP route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            qty_dot_domain=int(request.form.get('qty_dot_domain')),
            qty_slash_directory=int(request.form.get('qty_slash_directory')),
            qty_hyphen_file=int(request.form.get('qty_hyphen_file')),
            qty_percent_file=int(request.form.get('qty_percent_file')),
            qty_hyphen_params=int(request.form.get('qty_hyphen_params')),
            qty_underline_params=int(request.form.get('qty_underline_params')),
            qty_slash_params=int(request.form.get('qty_slash_params')),
            qty_hashtag_params=int(request.form.get('qty_hashtag_params')),
            email_in_url=int(request.form.get('email_in_url')),
            domain_in_ip=int(request.form.get('domain_in_ip')),
            server_client_domain=int(request.form.get('server_client_domain')),
            tld_present_params=int(request.form.get('tld_present_params')),
            domain_spf=int(request.form.get('domain_spf')),
            tls_ssl_certificate=int(request.form.get('tls_ssl_certificate')),
            url_google_index=int(request.form.get('url_google_index')),
            domain_google_index=int(request.form.get('domain_google_index')),
            url_shortened=int(request.form.get('url_shortened')),
            time_response=float(request.form.get('time_response')),
            time_domain_activation=int(request.form.get('time_domain_activation')),
            time_domain_expiration=int(request.form.get('time_domain_expiration')),
            length_url=int(request.form.get('length_url')),
            domain_length=int(request.form.get('domain_length')),
            directory_length=int(request.form.get('directory_length')),
            file_length=int(request.form.get('file_length')),
            params_length=int(request.form.get('params_length')),
            ttl_hostname=int(request.form.get('ttl_hostname')),
            asn_ip=int(request.form.get('asn_ip'))
        )
        pred_df = data.get_data_as_dataframe()
        logging.info(pred_df)
        logging.info(pred_df.columns)
        logging.info("Before Prediction")

        predict_pipeline = PredictPipeline()
        logging.info("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        logging.info("After Prediction")
        return render_template('form.html', results=results[0])

# API route
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        custom_data = CustomData(**data)
        pred_pipeline = PredictPipeline()
        prediction = pred_pipeline.predict(custom_data.get_data_as_dataframe())
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0")
