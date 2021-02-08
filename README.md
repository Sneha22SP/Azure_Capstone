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

### :dizzy: Description

**automl settings** :- experiment_timeout_minutes - maximum amount of time the experiment can take. So, I set it to 20 minutes to save time. 
max_concurrent_iterations - maximum number of parallel runs executed on a Automl Compute cluster. As it should be less than or equal to the number of nodes (5) so its set when creating the compute cluster(it is set to 5). The primary metric is Under the Curve Weighted, **AUC_weighted**, to deal with class imbalance as The AUC is an estimate of the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance. For this reason, the AUC is widely thought to be a better measure than a classification error rate based upon a single prior probability or KS statistic threshold.

**AutoMLConfig** :-  This is a binary classification task. Heart Failure "dataset" is imported earlier from the registered dataset in Azure Workspace. The target variable which we need to peredict in this experiment is "DEATH_EVENT". To save time and resources, the enable_early_stopping parameter is set to True.



### :dizzy: Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?


(need to write about the votingensamble)


### :dizzy: Parameters of the fitted model

```

Fitted model and its hyperparameters :  ('prefittedsoftvotingclassifier', PreFittedSoftVotingClassifier(classification_labels=None,
                              estimators=[('23',
                                           Pipeline(memory=None,
                                                    steps=[('minmaxscaler',
                                                            MinMaxScaler(copy=True,
                                                                         feature_range=(0,
                                                                                        1))),
                                                           ('extratreesclassifier',
                                                            ExtraTreesClassifier(bootstrap=True,
                                                                                 ccp_alpha=0.0,
                                                                                 class_weight='balanced',
                                                                                 criterion='gini',
                                                                                 max_depth=None,
                                                                                 max_features=None,
                                                                                 max_leaf_nodes=None,
                                                                                 max_samples=None,
                                                                                 m...
                                                                                 min_samples_split=0.056842105263157895,
                                                                                 min_weight_fraction_leaf=0.0,
                                                                                 n_estimators=25,
                                                                                 n_jobs=1,
                                                                                 oob_score=True,
                                                                                 random_state=None,
                                                                                 verbose=0,
                                                                                 warm_start=False))],
                                                    verbose=False))],
                              flatten_transform=None,
                              weights=[0.07692307692307693, 0.07692307692307693,
                                       0.15384615384615385, 0.15384615384615385,
                                       0.07692307692307693, 0.07692307692307693,
                                       0.23076923076923078, 0.07692307692307693,
                                       0.07692307692307693]))
                                       
 
 ```
 
 ### :dizzy: Improvement 
 
To improve model results, we can use k-fold cross validation, we can increase time of the experiment so that we can come up with good algorithms which can be imporved the model further. We can also perform feature selection and engineering and also we can explore different matrics like accuracy, F1-score .
 

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

# :dizzy: Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I have choosen Logistic Regression because it is a binary classification algorithm in which dependent variable is binary i,e 1(True,Sucess),0(False,Failure). Goal is to find the best fitting model for independent and dependent variable in the relationship. Independent variable can be continous or binary, also called as logit regression, used in machine learning,deals with probability to measure the relation between dependent and independent variables.


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

*  **Convert model to ONNX format** :- The open neural network exchange(OONX) is an open source artifical intelligence ecosystem.It allows users to interchange models with various ML frameworks and tools.ONNX is a portability platform for models that was created by Microsoft and that allows you to convert models from one framework to another, or even to deploy models to a device (such as an iOS or Android mobile device).
[ONNX_DOCS](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model)
[OONX](https://onnx.ai/)

* We can try deploying the best model with azure kubernetes service.

* We can choose another sklearn classifier rather then logestic regression.We can apply XGBoost or some Bagging and Boosting algorithms.

* We can choose another sampling policy rather then random sampler, we can choose grid search for betterment.
# :dizzy: References

:collision: [Hyperparameter_tuning_for_ml_models](https://github.com/microsoft/MLHyperparameterTuning)

:collision: [Deploy_and_consume_model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python)

:collision: [HyperDrive_Overview](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python) 

:collision: [AutoML_Overview](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azure-ml-py)

:collision: [Azure_ML_Notebboks](https://github.com/Azure/MachineLearningNotebooks)







  **![Made with love in India](https://madewithlove.now.sh/in?heart=true&template=for-the-badge)**
