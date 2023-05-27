import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
import pickle
from random import uniform
from sklearn.metrics import accuracy_score   
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import datetime
import warnings

import re
from sklearn.svm import SVR
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

sns.set(font_scale=1)

plt.style.use("Solarize_Light2")

df = pd.DataFrame()
df = pd.read_csv("./megastore-classification-dataset.csv")
# splitting
X = df.iloc[:, :-1]
y = df['ReturnCategory']


# y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
y_train
print("Countries: ", end="")
print(df['Country'].unique())
print('Number of states: ' + str(df['State'].unique().shape[0]))
print('Number of cities: ' + str(df['City'].unique().shape[0]))
print("Number of Products: " + str(df['Product ID'].unique().shape[0]))
print()
print()
categorical_columns = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region']

for col in categorical_columns:
    print(f"{col} unique values:")
    print(df[col].unique())
    print()
print()
categorical_columns = ['Ship Mode', 'Segment', 'Region','ReturnCategory']
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=90)
    plt.title(f"Distribution of {col}")
    plt.show()
    print()
    print()
print("Number of duplicates in the train data : ",X_train.duplicated().sum())
print("Number of Null Values in the train data : ",X_train.isnull().sum())


def timeAnalysis(X,y):
    data = X
    data['ReturnCategory'] = y

    plt.bar(X['Shipping Time'], y, color ='blue',
        width = 0.4)
    plt.xlabel('Shipping Time')
    plt.ylabel('ReturnCategory')
    plt.title("ReturnCategory depending on Shipping Time")
    plt.show()


def outliersD(XX,yy):
    X = XX
    y = yy
    fig = px.box(y, x="ReturnCategory",orientation = 'h')
    # fig.update_traces(orientation='h')
    fig.show()
    data = X
    data['ReturnCategory'] = y
    top5ovr = y
    q2 = top5ovr.median()
    # top5ovr.sort_values("weight_kg")
    q3, q1 = np.percentile(top5ovr, [75, 25])
    iqr = q3 - q1
    bOutliers = top5ovr[data['ReturnCategory'] > (q3 + 1.5*iqr)]
    sOutliers = top5ovr[data['ReturnCategory'] < (q1 - 1.5*iqr)]
    print()
    print('Detecting Outliers')
    print()
    print('Q1 = %i, Q2 = %i, Q3 = %i'%(q1, q2, q3))
    print()
    print('IQR =',iqr)
    print()
    print("Lower Outliers: ",len(sOutliers))
    print()
    print(sOutliers.sort_values().values.tolist())
    print()
    print("Upper Outliers: ",len(bOutliers))
    print()
    print(bOutliers.sort_values().values.tolist())
    print()
    smax = (sOutliers.max())
    bmin = (bOutliers.min())
    print("Lower Fence: ",smax)
    print()
    print("Upper Fence: ",bmin)
    data = data[(data['ReturnCategory'] > (smax)) & (data['ReturnCategory'] < (bmin))]
    X = data.iloc[:, :-1]
    y = data['ReturnCategory']
    # df = px.data.tips()
    return X,y

def categoryPreprocessing(df):
    categoryTree = "{'MainCategory': 'Office Supplies', 'SubCategory': 'Binders'}"
    df['MainCategory'] = 1
    main = []
    for i in df["CategoryTree"]:
        categoryTree = i
        main.append(re.search(r"MainCategory': '([\w\s]+)'", categoryTree).group(1))

    df['MainCategory'] = main
    main = []
    for i in df["CategoryTree"]:
        categoryTree = i
        main.append(re.search(r"SubCategory': '([\w\s]+)'", categoryTree).group(1))

    df['SubCategory'] = main
    # print(df[['CategoryTree','MainCategory','SubCategory']].head())

def labelEncodingX_train(df,type):
    global le
    le = LabelEncoder()
    global encoders
    encoders = {}

    categorical_features = df.columns

    for col in categorical_features:
        if df[col].dtype == 'object':
            unique_values = list(df[col].unique())
            unique_values.append('Unseen')
            le = LabelEncoder().fit(unique_values)
            df[col] = le.transform(df[[col]])
            encoders[col] = le

def labelEncodingX_test(df,type):
    global le
    le = LabelEncoder()
    global encoders

    categorical_features = df.columns

    for col in categorical_features:
        if df[col].dtype == 'object':
            le = encoders.get(col)
            df[col] = [x if x in le.classes_ else 'Unseen' for x in df[col]]
            df[col] = le.transform(df[[col]])
            
def featureSelection(X, y):
    # df['Country'].value_counts()
    # df.info()
    X.drop('CategoryTree', axis=1, inplace=True)
    data = X
    data['ReturnCategory'] = y
    corr = data.corr()
    # print(corr['ReturnCategory'])
    # Top 50% Correlation training features with the Value
    global top_feature
    top_feature = corr.index[abs(corr['ReturnCategory']) > 0.06]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    top_feature = top_feature.delete(-1)
    # print(top_feature)
    X = data[top_feature]
    return X
def datePreprocessing(df):
    OrderDate = pd.to_datetime(df['Order Date'], errors='ignore', dayfirst=False)
    ShipDate = pd.to_datetime(df['Ship Date'], errors='ignore', dayfirst=False)

    sodt = (ShipDate - OrderDate)

    df['Shipping Time'] = sodt.dt.days

def preproccessing(X, y):
    categoryPreprocessing(X)
    datePreprocessing(X)
    X['Sales'].fillna(value=X['Sales'].mean(), inplace=True)
    X['Quantity'].fillna(value=X['Quantity'].mean(), inplace=True)
    X['Discount'].fillna(value=X['Discount'].mean(), inplace=True)
    # timeAnalysis(X, y)
    X = X.reindex(columns=['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                           'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
                           'Postal Code', 'Region', 'Product ID', 'CategoryTree', 'Product Name',
                           'Sales', 'Quantity', 'Discount', 'MainCategory',
                           'SubCategory', 'Shipping Time'])
    labelEncodingX_train(X,0)
    X = featureSelection(X, y)
    return X

def preproccessingtest(X, y):
    categoryPreprocessing(X)
    datePreprocessing(X)
    X['Sales'].fillna(value=X['Sales'].mean(), inplace=True)
    X['Quantity'].fillna(value=X['Quantity'].mean(), inplace=True)
    X['Discount'].fillna(value=X['Discount'].mean(), inplace=True)
    X = X.reindex(columns=['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                           'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
                           'Postal Code', 'Region', 'Product ID', 'CategoryTree', 'Product Name',
                           'Sales', 'Quantity', 'Discount', 'MainCategory',
                           'SubCategory', 'Shipping Time'])
    labelEncodingX_test(X,1)
    X = X[top_feature]

    return X
def labelEncodingY_train(df):
    global le
    
    le = LabelEncoder()
    unique_values = list(df.unique())
    unique_values.append('Unseen')
    le = LabelEncoder().fit(unique_values)
    df = le.transform(df)
    return df        


def labelEncodingY_test(df):
    global le
    
    df = [x if x in le.classes_ else 'Unseen' for x in df]
    df = le.transform(df)
    return df        
y_train =  labelEncodingY_train(y_train)
y_test = labelEncodingY_test(y_test)

def train_linear_model(X_train, X_test, y_train, y_test):
    # linear model
    sln = LogisticRegression(solver = 'newton-cg',max_iter=1000)
    sln.fit(X_train, y_train)
    pickle.dump(sln, open("LogisticRegression.pkl", "wb"))

def random_forest_modelGrid(X_train, X_test, y_train, y_test):
    # Number of trees in random forest
    rf = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    prediction = rf_random.predict(X_test)

    print('----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------')
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",rf_random.best_estimator_)
    print("\n The best score across ALL searched params:\n",rf_random.best_score_)
    print("\n The best parameters across ALL searched params:\n",rf_random.best_params_)
    print('score of test = ', accuracy_score(y_test, prediction))




def random_forest_model(X_train, X_test, y_train, y_test):
    # Number of trees in random forest
    rf_random = RandomForestClassifier(n_estimators= 1400, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto',
                                       max_depth= 40, bootstrap= False)
    rf_random.fit(X_train, y_train)
    pickle.dump(rf_random, open("randomforestclassifier.pkl", "wb"))

    

def DT_model(X_train, X_test, y_train, y_test,criterion = 'gini',max_depth = None):
    # Number of trees in random forest
    rf_random = DecisionTreeClassifier(criterion = criterion,max_depth=max_depth)
    
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    prediction = rf_random.predict(X_test)
    print(rf_random.get_params())
    print('score of test = ', accuracy_score(y_test, prediction)) 
    pickle.dump(rf_random, open("DTclassifier.pkl", "wb"))

    
def PrintModel(X_test,y_test,modelName,model):
    # Number of trees in random forest
    prediction = model.predict(X_test)
    # Fit the random search model

    print('----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------')

    print(modelName,' Results:')
    # print('Co-efficient of multiple linear regression', sln.coef_)
    print('score of test for ',modelName,' = ', accuracy_score(y_test, prediction))  
    return accuracy_score(y_test, prediction)


# print(np.unique(y_train))
# print(np.unique(y_test))
values, counts = np.unique(y_train, return_counts=True)
print(values)
print(counts)
y_train = pd.DataFrame(y_train,columns=['ReturnCategory'])
y_train = y_train['ReturnCategory']
y_test = pd.DataFrame(y_test,columns=['ReturnCategory'])
y_test = y_test['ReturnCategory']
# outliersD(X_train, y_train)
X_train = preproccessing(X_train, y_train)
X_test = preproccessingtest(X_test, y_test)
print(top_feature)
scaler = StandardScaler()
X_train[['Sales', 'Quantity', 'Discount']] = scaler.fit_transform(X_train[['Sales', 'Quantity', 'Discount']])
X_test[['Sales', 'Quantity', 'Discount']] = scaler.transform(X_test[['Sales', 'Quantity', 'Discount']])
# now = time.ctime()
current_time = datetime.datetime.now()
start = current_time
# train_linear_model(X_train, X_test, y_train, y_test)
# random_forest_model(X_train, X_test, y_train, y_test)
# DT_model(X_train, X_test, y_train, y_test)
current_time = datetime.datetime.now()
end = current_time
time1 = end - start
# print(time1)
# pickle.dump(time1, open("logisticTime.pkl", "wb"))
# pickle.dump(time1, open("dtTime.pkl", "wb"))
# pickle.dump(time1, open("rfTime.pkl", "wb"))
timelo = pickle.load(open("logisticTime.pkl", "rb"))
timedt = pickle.load(open("dtTime.pkl", "rb"))
timerf = pickle.load(open("rfTime.pkl", "rb"))
# print(timerf.seconds)
labels = ['LogisticRegression','DTclassifier','RandomForestClassifier']
times = [timelo.seconds,timedt.seconds,timerf.seconds]
model_lr = pickle.load(open("LogisticRegression.pkl", "rb"))
model_DT = pickle.load(open("DTclassifier.pkl", "rb"))
model_rf = pickle.load(open("randomforestclassifier.pkl", "rb"))
# result = model_mtd2.predict(X_test)
scorelo = PrintModel(X_test,y_test,"LogisticRegression",model_lr)
scoredt = PrintModel(X_test,y_test,"DecisionTreeClassifier",model_DT)
scorerf = PrintModel(X_test,y_test,"Random Forest",model_rf)
scores = [scorelo,scoredt,scorerf]
fig = px.bar(x=labels, y=scores)
fig.show()
fig = px.bar(x=labels, y=times)
fig.show()

