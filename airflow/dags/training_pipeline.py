from __future__ import annotations
import numpy as np
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.Phishing.pipelines.training_pipeline import TrainingPipeline

training_pipeline = TrainingPipeline()

with DAG(
    "phishing_training_pipeline",
    default_args={"retries": 2},
    description="It is my training pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2024, 2, 15, tz="UTC"),
    catchup=False,
    tags=["machine_learning", "classification", "phishing"],
) as dag:

    dag.doc_md = __doc__

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_data_path, test_data_path = training_pipeline.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifact", {"train_data_path": train_data_path, "test_data_path": test_data_path})

    def data_transformation(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifact")
        train_arr, test_arr = training_pipeline.start_data_transformation(data_ingestion_artifact["train_data_path"], data_ingestion_artifact["test_data_path"])
        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()
        ti.xcom_push("data_transformation_artifact", {"train_arr": train_arr, "test_arr": test_arr})

    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifact")
        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        training_pipeline.start_model_training(train_arr, test_arr)

    def model_evaluation(**kwargs):
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifact")
        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        training_pipeline.start_model_evaluation(train_arr, test_arr)

    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
        provide_context=True
    )
    data_ingestion_task.doc_md = dedent(
        """\
    #### Data Ingestion Task
    This task retrieves data from AWS and creates train and test files.
    """
    )

    data_transform_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation,
        provide_context=True
    )
    data_transform_task.doc_md = dedent(
        """\
    #### Data Transformation Task
    This task performs data transformation.
    """
    )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
        provide_context=True
    )
    model_trainer_task.doc_md = dedent(
        """\
    #### Model Trainer Task
    This task performs model training.
    """
    )

    model_evaluation_task = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation,
        provide_context=True
    )
    model_evaluation_task.doc_md = dedent(
        """\
    #### Model Evaluation Task
    This task performs model evaluation.
    """
    )

data_ingestion_task >> data_transform_task >> model_trainer_task >> model_evaluation_task
