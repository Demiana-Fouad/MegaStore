import pickle
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
import datetime
import warnings

from sklearn.svm import SVR
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

sns.set(font_scale=1)
import re

plt.style.use("Solarize_Light2")

df = pd.DataFrame()
df = pd.read_csv("./megastore-regression-dataset.csv")
# splitting
X = df.iloc[:, :-1]
y = df['Profit']


# y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
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
categorical_columns = ['Ship Mode', 'Segment', 'Region']
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
    data['Profit'] = y

    plt.bar(X['Shipping Time'], y, color ='blue',
        width = 0.4)
    plt.xlabel('Shipping Time')
    plt.ylabel('Profit')
    plt.title("Profit depending on Shipping Time")
    plt.show()
def outliersD(XX,yy):
    X = XX
    y = yy
    fig = px.box(y, x="Profit",orientation = 'h')
    # fig.update_traces(orientation='h')
    fig.show()
    data = X
    data['Profit'] = y
    top5ovr = y
    q2 = top5ovr.median()
    # top5ovr.sort_values("weight_kg")
    q3, q1 = np.percentile(top5ovr, [75, 25])
    iqr = q3 - q1
    bOutliers = top5ovr[data['Profit'] > (q3 + 1.5*iqr)]
    sOutliers = top5ovr[data['Profit'] < (q1 - 1.5*iqr)]
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
    data = data[(data['Profit'] > (smax)) & (data['Profit'] < (bmin))]
    X = data.iloc[:, :-1]
    y = data['Profit']
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
    # print(df['SubCategory'])
    # df[['CategoryTree','MainCategory','SubCategory']].head()


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X





def datePreprocessing(df):
    OrderDate = pd.to_datetime(df['Order Date'], errors='ignore', dayfirst=False)
    ShipDate = pd.to_datetime(df['Ship Date'], errors='ignore', dayfirst=False)

    sodt = (ShipDate - OrderDate)

    df['Shipping Time'] = sodt.dt.days
    # print(df['Shipping Time'])



def labelEncoding(df,type):
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
            
def labelEncodingtest(df,type):
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
    data['Profit'] = y
    corr = data.corr()
    # print(corr['Profit'])
    # Top 50% Correlation training features with the Value
    global top_feature
    top_feature = corr.index[abs(corr['Profit']) > 0.06]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    top_feature = top_feature.delete(-1)
    # print(top_feature)
    X = data[top_feature]
    return X





def preproccessing(X, y):
    global scaler,scaler2
    scaler = StandardScaler()
    scaler2 = StandardScaler()
    X_train[['Sales', 'Quantity', 'Discount']] = scaler.fit_transform(X_train[['Sales', 'Quantity', 'Discount']])
    categoryPreprocessing(X)
    datePreprocessing(X)
    X['Sales'].fillna(value=X['Sales'].mean(), inplace=True)
    X['Quantity'].fillna(value=X['Quantity'].mean(), inplace=True)
    X['Discount'].fillna(value=X['Discount'].mean(), inplace=True)
    timeAnalysis(X, y)
    X = X.reindex(columns=['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                           'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
                           'Postal Code', 'Region', 'Product ID', 'CategoryTree', 'Product Name',
                           'Sales', 'Quantity', 'Discount', 'MainCategory',
                           'SubCategory', 'Shipping Time'])
    labelEncoding(X,0)
    X = featureSelection(X, y)
    y1 = np.array(y).reshape(-1, 1)
    y = scaler2.fit_transform(y1)
    y = pd.DataFrame(y,columns=['Profit'])
    y = y['Profit']
    return X,y

def preproccessingtest(X, y):
    X_test[['Sales', 'Quantity', 'Discount']] = scaler.transform(X_test[['Sales', 'Quantity', 'Discount']])
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
    labelEncodingtest(X,1)
    X = X[top_feature]
    y1 = np.array(y).reshape(-1, 1)
    y = scaler2.transform(y1)
    y = pd.DataFrame(y,columns=['Profit'])
    y = y['Profit']
    return X,y




# first model
def train_poly_model(X_train, X_test, y_train, y_test, d=2):
    # polynomial model

    poly = PolynomialFeatures(degree=d)  # declare poly transformer
    x_train_poly = poly.fit_transform(X_train)  # transforms features to higher degree
    x_test_poly = poly.transform(X_test)  # transforms features to higher degree
    leaner = LinearRegression().fit(x_train_poly, y_train)  # declare leaner model & Normalize & fit
    pickle.dump(leaner, open("PolyRegression.pkl", "wb"))
    pickle.dump(x_test_poly, open("PolyRegressionfeatures.pkl", "wb"))




def train_linear_model(X_train, X_test, y_train, y_test):
    # linear model
    sln = linear_model.LinearRegression()
    sln.fit(X_train, y_train)
    pickle.dump(sln, open("LinearRegression.pkl", "wb"))


    

def lasso_model(X_train, X_test, y_train, y_test):
    # linear model
    clf = linear_model.Lasso(alpha=0.01)    
    clf.fit(X_train, y_train)
    pickle.dump(clf, open("LassoRegression.pkl", "wb"))

def random_forest_model(X_train, X_test, y_train, y_test):
    # linear model
    regr = RandomForestRegressor(max_leaf_nodes = 1000,max_depth=15, random_state=0)
    regr.fit(X_train, y_train)
    pickle.dump(regr, open("rfRegression.pkl", "wb"))


def PrintModel(X_test,y_test,modelName,model):
    # Number of trees in random forest
    prediction = model.predict(X_test)
    # Fit the random search model
    fig = px.scatter(x=y_test, y=prediction, title=modelName,labels={'x': 'ground truth', 'y': 'prediction'})
    fig.add_shape(
    type="line", line=dict(dash='dash'),
    x0=prediction.min(), y0=prediction.min(),
    x1=prediction.max(), y1=prediction.max()
    )
    fig.show()
    print('----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------'
          '----------------------------------------------------------------------------')
    print(modelName,'Results:')
    print('Mean Square Error to ',modelName,' = ', metrics.mean_squared_error(y_test, prediction))
    print('R2 score of test = ', r2_score(y_test, prediction))
    


# preparing data
# X_train, y_train = outliersD(X_train, y_train)
outliersD(X_train, y_train)
# scaler = StandardScaler()
scaler2 = StandardScaler()

# X_train[['Sales', 'Quantity', 'Discount']] = scaler.fit_transform(X_train[['Sales', 'Quantity', 'Discount']])
# X_test[['Sales', 'Quantity', 'Discount']] = scaler.transform(X_test[['Sales', 'Quantity', 'Discount']])

X_train,y_train = preproccessing(X_train, y_train)
X_test,y_test = preproccessingtest(X_test, y_test)

# y = np.array(y_train).reshape(-1, 1)
# y_train = scaler2.fit_transform(y)
# y_train = pd.DataFrame(y_train,columns=['Profit'])
# y_train = y_train['Profit']
# y = np.array(y_test).reshape(-1, 1)
# y_test = scaler2.transform(y)
# y_test = pd.DataFrame(y_test,columns=['Profit'])
# y_test = y_test['Profit']

# train_linear_model(X_train, X_test, y_train, y_test)
# lasso_model(X_train, X_test, y_train, y_test)
# random_forest_model(X_train, X_test, y_train, y_test)
# train_poly_model(X_train, X_test, y_train, y_test, 2)
model_lr = pickle.load(open("LinearRegression.pkl", "rb"))
model_la = pickle.load(open("LassoRegression.pkl", "rb"))
model_rf = pickle.load(open("rfRegression.pkl", "rb"))
model_pr = pickle.load(open("PolyRegression.pkl", "rb"))
x_test_poly = pickle.load(open("PolyRegressionfeatures.pkl", "rb"))
print(top_feature)
PrintModel(X_test,y_test,"LinearRegression",model_lr)
PrintModel(X_test,y_test,"LassoRegression",model_la)
PrintModel(X_test,y_test,"Random Forest",model_rf)
PrintModel(x_test_poly,y_test,"PolyRegression",model_pr)