# BigmartSales

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.


The data comes in 2 files:

* Training set -- contains both the independent and dependent (Item_Outlet_Sales) variables for the training set 
* Test set values -- contains the independent variables that need predictions

The original training and test set have 11 features 7 of which are categorical.
Missing values: There are only 2 features ('Item_Weight' and 'Outlet_Size') with missing values. We handle them in the following way. For the nominal 'Item_Weight' we impute the missing values with the average weight of the items in the corresponding 'Item_Type' category. For the categegorical 'Outlet_Size' we use the mode of 'Outlet_Size' for the corresponding 'Outlet_Type'.


For building the prediction model we will use the scikit.learn library. Since it only accepts numerical values, first we need to convert all categorical variables into dummy variables.


In the following table we summarize the performing result for different algorithms. Note that in the CV column the score is calculated via 3-fold cross-validation on the training data. The test results are calculated only once and are obtained via submitting the predictions at DrivenData (https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/). Bolded are the algorighms that performed best on the train and on the test sets. 

The Root Mean Square Error (RMSE) is used for evaluation of the model. As a benchmark model we use a model that predicts the sales as the average sales of product of the same 'Item_Type'. Its Public Leaderboard Score is 1775.34.


| Algorithm | CV | Test |
|---------- | -- | ---- |
| Linear Regression | 1133.77 | 1202.85 | 
| Ridge Regression |  | 1133.72 | 1203.17 |
| Lasso Regression | 1131.89 | 1201.80 | 
| Decision Tree | 1093.78 | 1159.26 | 
| **Random Forest** | 1093.48 | **1153.89** | 
| **xgb** | **1091.03** | 1156.92 | 
