# :dizzy: AZURE MACHINE LEARNING ENGINNER :dizzy: 

This project gives us the opportunity to use the knowledge that we have obtained from the Nanodegree program to solve an interesting problem.In this project, we will create a two models :- one using AutomatedML(AutoML) and one customized model whose hyperparameters are tuned using HyperDrive.Then, compare the performance of the both models and deploy the best performing model.This project will demonstarte our ability to use an external dataset (I am using heart_failure_prediction dataset from kaggle to build a classification model)in our workspace train a model using the different tools available in the AzureML framework as well as will get to know the ability to deploy the model as a webservice.  

![](screenshots/capstone-diagram.png)


# :dizzy: Dataset

## :dizzy: Overview

In this project, I used the "Heart Failure Prediction" dataset from Kaggle.The dataset contains 12 clinical features of 299 patients.Here "DEATH EVENT" is the target variable, we need to predict the possible death event occured due to heat failure of the patients.With the help of HyperDrive and AutoML will get to know the death rates of the patients.

## :dizzy: Task

The task of this project is to predict the "DEATH_EVENT" of the patients.This problem belongs to classification category.The target variable death event is in the boolean form, with the help of two models will predict the survival rate of the patients due to heart failure.

**12 clinical features are** :- age, anaemia, creatinine_phosphokinare, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time.

**Target variable** :- death_event


| Features                    | Description                                              |
| ----------------------------| ---------------------------------------------------------|
| Age                         | 1 for Male 0 for Female
|anaemia                      | a condition in which there is a deficiency of red cells or of haemoglobin in the blood, resulting in tired, weakness, shortness of breath, and a                                poor ability to exercise.
|creatinine_phosphokinase     | (mcg/L):(CPK) or (CK) is a enzyme that catalyzes the reaction of creatine and adenosine triphosphate (ATP),Phosphocreatine created from this                                    reaction is used to supply tissues and cells e.g. brain skeletal muscles, and the heart.
|diabetes                     |  a metabolic disease that causes high blood sugar. Result in increased hunger, increased thirst,weight loss,frequent urination,blurry                                             vision,extreme fatigue,sores that don’t heal
|ejection_fraction            | Percentage of blood leaving the heart at each contraction (percentage)
|high_blood_pressure          | common condition in which the long-term force of the blood against your artery
|platelets                    | (kiloplatelets/mL): small, colorless cell fragments in our blood that form clots and stop or prevent bleeding.
|serum_creatinine             | (mg/dL): Level of serum creatinine in the blood
|serum_sodium                 | (mEq/L): Level of serum sodium in the blood. Reference range for serum sodium is 135-147 mmol/L
|sex                          | Woman or man (binary)
|smoking                      | If the patient smokes or not (Boolean)
|time                         | Follow-up period (days)



|Target Variable | Description                                                      |
|----------------|------------------------------------------------------------------|
| DEATH_EVENT    | whether the patient survived or not due to heart failure (Boolean)|



## :dizzy: Access

In AutoML part, I have registered the dataset in the azure workspace.


```Python
found = False
key = "Heart Failure Prediction"
description_text = "Heart Failure Prediction Dataset for Capstone project"
if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key]
```

In Hyperdrive part, I saved the dataset to my GitHub repository and retrieved the data from a URL using TabularDatasetFactory class in train.py script.

```Python
data_path = "https://github.com/Sneha22SP/nd00333-capstone/blob/master/starter_file/heart_failure_clinical_records_dataset.csv"
ds = TabularDatasetFactory.from_delimited_files(path=data_path)
```


# :dizzy: Automated ML

|Parameters                     |Value                          |
|------------------------------ |-------------------------------| 
|experiment_timeout_minutes     |20                             |
|max_concurrent_iterations      |5                              |
|primary_metric | AUC_weighted|
|task  | Classification|
|compute_target | "General_cluster" previously created|
|training_data | dataset registered in Azure Workspace|
|label_column_name | DEATH_EVENT|
|enable_early_stopping | True|
|featurization | auto|
|debug_log | automl_errors.log|

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

# :dizzy: Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

# :dizzy: Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

# :dizzy: Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

# :dizzy: Standout Suggestions

**Convert model to ONNX format** :- The open neural network exchange(OONX) is an open source artifical intelligence ecosystem.
[ONNX_DOCS](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model)
# :dizzy: References

:collision: [Hyperparameter_tuning_for_ml_models](https://github.com/microsoft/MLHyperparameterTuning)

:collision: [Deploy_and_consume_model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python)

:collision: [HyperDrive_Overview](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python) 

:collision: [AutoML_Overview](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azure-ml-py)

:collision: [Azure_ML_Notebboks](https://github.com/Azure/MachineLearningNotebooks)







  **![Made with love in India](https://madewithlove.now.sh/in?heart=true&template=for-the-badge)**
