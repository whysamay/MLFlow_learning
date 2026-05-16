# My MLFlow Learning Notes

This repository serves as a personal log of everything I have learned and implemented while exploring **MLFlow** for Machine Learning Operations (MLOps). 

---

## 1. Core Concepts Learned
**MLFlow** is a platform for managing the end-to-end machine learning lifecycle. The main components I explored are:
- **Experiment Tracking:** Recording parameters, code versions, metrics, and output files/artifacts.
- **Model Registry:** A centralized hub to manage model versions and lifecycle stages (e.g., Staging, Production).

### Starting the Local Server
To view experiments locally, I learned to spin up the MLFlow UI using the terminal:
```bash
mlflow server --host 127.0.0.1 --port 5000
```
This UI allows me to visually compare different models and hyperparameter tweaks.

---

## 2. Implementations & Scripts

### A. Basic Local Tracking (`src/exp_mlflow_local.py`)
This script was my starting point for local experiment tracking using the Wine dataset and a Random Forest Classifier.
* **Key Takeaways:**
  - Connected the code to the local UI: `mlflow.set_tracking_uri("http://127.0.0.1:5000")`
  - Created an experiment namespace: `mlflow.set_experiment("Exp_1")`
  - Used `with mlflow.start_run():` to encapsulate the training block.
  - Logged parameters manually using `mlflow.log_param("max_depth", max_depth)`
  - Logged evaluation metrics: `mlflow.log_metric("accuracy", accuracy)`
  - Saved and logged artifacts (e.g., saving a Matplotlib confusion matrix and logging it using `mlflow.log_artifact()`).
  - Saved the trained model itself to MLFlow via `mlflow.sklearn.log_model()`.

### B. Remote Tracking via DagsHub (`src/exp_dagshub.py`)
Instead of logging to my local machine, I learned how to log directly to a remote tracking server using **DagsHub**.
* **Key Takeaways:**
  - Used `dagshub.init()` to authenticate with my remote repository.
  - Pointed the tracking URI to the remote server: `mlflow.set_tracking_uri('https://dagshub.com/samaypawar2200/MLFlow_learning.mlflow')`
  - Now, experiments ran on my machine are saved and visible on the cloud.

### C. MLFlow Autologging (`src/autolog.py`)
Manual logging is tedious. I learned that MLFlow can automatically log most of the standard parameters and metrics.
* **Key Takeaways:**
  - Used a single line `mlflow.autolog()` before starting the experiment run.
  - MLFlow intelligently hooks into `sklearn` and captures the Random Forest parameters and metrics automatically without me having to write `mlflow.log_param` for everything.
  - *Note:* Artifacts like custom plots still need to be logged manually.

### D. Hyperparameter Tuning & Nested Runs (`src/hypertuning.py`)
I tackled a more complex scenario: tracking a `GridSearchCV` hyperparameter tuning process on a Breast Cancer dataset.
* **Key Takeaways:**
  - **Nested Runs:** I used `with mlflow.start_run(nested=True) as child:` to log every single iteration of the GridSearch as a child run inside the main parent run. This keeps the MLFlow UI extremely clean and organized.
  - **Data Logging:** I learned to log the exact dataset used for training and testing to ensure complete reproducibility:
    ```python
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")
    ```
  - Extracted the best parameters (`grid_search.best_params_`) and the best model, explicitly logging them.

---

## 3. Model Registry Concepts (`modelregistry.txt`)
Once an experiment yields a good model, it needs to be versioned for production. I took notes on how the **Model Registry** handles this.
* **Key Takeaways:**
  - The registry is a "wrapper" around the logged model artifacts, transitioning a model from an isolated artifact into a version-controlled asset.
  - **Versioning:** Models registered under the same name automatically increment in version (v1, v2).
  - **Stages/Aliases:** Models can be tagged as `Staging`, `Production`, or `Archived`. Modern MLFlow also allows tagging models with aliases like `@champion` and `@challenger`.
  - **How to Register:** I can register a model directly during the logging phase:
    ```python
    mlflow.sklearn.log_model(
        sk_model=rf, 
        artifact_path="Random-Forest-Model",
        registered_model_name="Iris-Classifier-RF"
    )
    ```
    Or post-run via the MLFlow client using the run ID.

---

## Summary
Through these scripts, I've built a solid foundation in **experiment tracking**, **cloud integration**, **automated logging**, **handling complex nested hyperparameter searches**, and the theory behind **model registry**.
