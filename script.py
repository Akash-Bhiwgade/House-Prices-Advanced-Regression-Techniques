# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# load the test and train datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train shape: ", train.shape)
print("Test shape: ", test.shape)

plt.style.use(style="ggplot")
plt.rcParams['figure.figsize'] = [10, 6]

# explore the data
print("Train sale price describe", train.SalePrice.describe())

print("train sale price skew", train.SalePrice.skew())
# plt.hist(train.SalePrice, color='blue')

target = np.log(train.SalePrice)
print("train log transform skew", target.skew())
#plt.hist(target, color='blue')

numeric_features = train.select_dtypes(include=(np.number))
categorical_features = train.select_dtypes(exclude=(np.number))

corr = numeric_features.corr()

print("Top 5 positively correlated ", corr.SalePrice.sort_values(ascending=False)[:5])
print("Top 5 negatively correlated ", corr.SalePrice.sort_values(ascending=False)[-5:])

# plt.scatter(x=train['GarageArea'], y=target)

train = train[train['GarageArea'] < 1200]

#plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null_count']
nulls.index.name = 'Feature'

print("Original: ", train.Street.value_counts())

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)

print("Encoded: ", train.enc_street.value_counts())

#conditional_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
#conditional_pivot.plot(kind='bar', color='blue')

def encode(x): return 1 if x == 'Partial' else 0

train['enc_conditional'] = train.SaleCondition.apply(encode)
test['enc_conditional'] = test.SaleCondition.apply(encode)

conditional_pivot = train.pivot_table(index='enc_conditional', values='SalePrice', aggfunc=np.median)
# conditional_pivot.plot(kind='bar', color='blue')

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

X = data.drop(['SalePrice', 'Id'], axis=1)
y = np.log(train.SalePrice)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = linear_model.LinearRegression()
model = clf.fit(X_train, y_train)

score = model.score(X_test, y_test)
prediction = model.predict(X_test)

mean_squared_error_value = mean_squared_error(y_test, prediction)

plt.scatter(prediction, y_test, alpha=0.75, color='b')

submission = pd.DataFrame()
submission['Id'] = test['Id']

test_features = test.select_dtypes(include=(np.number)).drop(['Id'], axis=1).interpolate()

test_predictions = model.predict(test_features)

final_predictions = np.exp(test_predictions)

submission['SalePrice'] = final_predictions

submission.to_csv('HPI_predictions', index=False)