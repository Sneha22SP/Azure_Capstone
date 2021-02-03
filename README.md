# :dizzy: AZURE MACHINE LEARNING ENGINNER :dizzy: 

This project gives us the opportunity to use the knowledge that we have obtained from the Nanodegree program to solve an interesting problem.In this project, we will create a two models :- one using AutomatedML(AutoML) and one customized model whose hyperparameters are tuned using HyperDrive.Then, compare the performance of the both models and deploy the best performing model.This project will demonstarte our ability to use an external dataset (I am using heart_failure_prediction dataset from kaggle to build a classification model)in our workspace train a model using the different tools available in the AzureML framework as well as will get to know the ability to deploy the model as a webservice.  

![](screenshots/capstone-diagram.png)


# :dizzy: Dataset

## :dizzy: Overview

In this project, I used the "Heart Failure Prediction" dataset from Kaggle.The dataset contains 12 clinical features of 299 patients.Here "DEATH EVENT" is the target variable, we need to predict the possible death event occured due to heat failure of the patients.With the help of HyperDrive and AutoML will get to know the death rates of the patients.

## :dizzy: Task

The task of this project is to predict the "DEATH_EVENT" of the patients.This problem belongs to classification category.The target variable death event is in the boolean form, with the help of two models will predict the survival rate of the patients due to heart failure.

12 clinical features are :- age, anaemia, creatinine_phosphokinare, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time.

Target variable :- death_event
------------- | ----------- 
Age | Age of the patient

anaemia  | Decrease of red blood cells or hemoglobin (Boolean)

creatinine_phosphokinase | Level of the CPK enzyme in the blood (mcg/L)

diabetes | If the patient has diabetes (Boolean)

ejection_fraction | Percentage of blood leaving the heart at each contraction (percentage)

high_blood_pressure | If the patient has hypertension (Boolean)

platelets | Platelets in the blood (kiloplatelets/mL)

serum_creatinine | Level of serum creatinine in the blood (mg/dL)

serum_sodium | Level of serum sodium in the blood (mEq/L)

sex | Woman or man (binary)

smoking | If the patient smokes or not (Boolean)

time | Follow-up period (days)



Target Variable | Description 

------------- | ----------- 




## :dizzy: Access
*TODO*: Explain how you are accessing the data in your workspace.

# :dizzy: Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

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
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

# :dizzy: References

:collision: [Hyperparameter_tuning_for_ml_models](https://github.com/microsoft/MLHyperparameterTuning)

:collision: [Deploy_and_consume_model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python)

:collision: [HyperDrive_Overview](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python) 

:collision: [AutoML_Overview](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azure-ml-py)







  **![Made with love in India](https://madewithlove.now.sh/in?heart=true&template=for-the-badge)**
