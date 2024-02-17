# Phishing-Detection
# Machine Learning Solution

This project showcases an end-to-end machine learning solution developed using AWS infrastructure, leveraging DynamoDB and S3 buckets for efficient data storage and handling. The primary goal was to build a highly accurate predictive model using either CatBoost or XGBoost algorithms on a dataset featuring over 100 features. Additionally, the project involved thorough correlation analysis and feature importance assessment to ensure the robustness of the model.

Here is the project design:
![Project Design](https://i.ibb.co/pJ2XMkr/ml-7.jpg)


## Features

- **AWS Infrastructure**: Utilize AWS services such as DynamoDB and S3 buckets for data storage, handling and deployment as well.
- **Machine Learning Algorithms**: Implement CatBoost or XGBoost algorithms to build a highly accurate predictive model.
- **MLflow Experimentation**: Use MLflow for tracking and versioning during extensive experimentation.
- **Evidently AI Monitoring**: Employ Evidently AI for ongoing monitoring and testing of the model's reliability.
- **Docker Containerization**: Utilize Docker for containerization to ensure a consistent environment across different deployments.
- **DVC Pipeline Management**: Employ DVC for pipeline management, ensuring effective project management and reproducibility.
- **Github Actions**: Utilizes it for CI/CD workflow.
- **Airflow Orchestration**: Use Airflow for orchestrating continuous training(CT) pipelines, enabling seamless deployment, testing, and continuous training of the model.

## Getting Started

To get started with this project, follow these steps:

1. **Setup AWS Environment**: Configure AWS environment with DynamoDB for data storage and S3 buckets for data handling.
2. **Install Dependencies**: Install necessary dependencies for the project using `requirements.txt`.
3. **Run Experiments**: Use MLflow for tracking and versioning during model experimentation.
4. **Monitor Model Performance**: Employ Evidently AI for ongoing monitoring and testing of the model's reliability.
5. **Manage Pipelines**: Utilize DVC for managing pipelines to ensure project reproducibility and effective management.
6. **Orchestrate Pipelines**: Use Airflow and Github Actions for orchestrating CI/CD pipelines, enabling seamless deployment, testing, and continuous training of the model.

## Result

The project resulted in the development of a highly accurate predictive model, achieving over 95% accuracy. The comprehensive approach to model development and deployment ensured its effectiveness and scalability within the AWS ecosystem. Additionally, the use of MLflow, Evidently AI, Docker, DVC, and Airflow contributed to efficient experimentation, monitoring, and deployment processes throughout the project lifecycle.

## Improvement

The project is complete but still under development and requires a few improvements which are being worked on.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
