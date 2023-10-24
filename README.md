# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JAI PRADHIKSHA D P
RegisterNumber:  212221040062
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as pit

dataset = pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())

#hours to X
X = dataset.iloc[:,:-1].values
print(X)
#scores to Y
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("Values of MSE, MAE and RMSE")
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

a = np.array([[10]])

Y_pred1=reg.predict(a)
print(Y_pred1)

```

## Output:
![image](https://github.com/Jai-Pradhiksha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/100289733/d9e9d044-1040-447b-8343-328d81327a5e)

![image](https://github.com/Jai-Pradhiksha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/100289733/8e210a0e-7ac1-4ec4-b168-ef4f4d0a1f32)

![image](https://github.com/Jai-Pradhiksha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/100289733/214f41d5-5d67-475c-a02b-ae28bd820811)

![image](https://github.com/Jai-Pradhiksha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/100289733/c8d9ca6a-2f81-4548-acd9-d882e67334f0)

![image](https://github.com/Jai-Pradhiksha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/100289733/a47bd15f-b077-495c-b381-4de508f1b217)

![image](https://github.com/Jai-Pradhiksha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/100289733/cfec3969-ba1d-481f-9400-3ee26268003a)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
