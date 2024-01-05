import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


def lin(acc , cmm ,ss , sss , mr , ms , ma):
    global y_test
    global y_pred
    # dataset
    data =pd.read_csv('dataset.csv')
    #data.head(8)

    # first columns
    X = data[['Discharge Time (s)','Decrement 3.6-3.4V (s)','Max. Voltage Dischar. (V)','Min. Voltage Charg. (V)','Time at 4.15V (s)','Time constant current (s)','Charging time (s)']].values

    # last column
    y = data.iloc[:, 8]

    # train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # regressop
    lr = LinearRegression()


    # train the LR classifier
    lr.fit(X_train, y_train)
    # predictions
    y_pred = lr.predict(X_test)

    # In this case, we can simply use the average value of y_pred
    threshold = np.mean(y_pred)

    # Convert predictions to binary (1 or 0)
    y_pred_binary = np.zeros(y_pred.shape)
    y_pred_binary[y_pred > threshold] = 1

    # Compute confusion matrix
    cm = sm.confusion_matrix(y_test, y_pred_binary)

    # Compute accuracy score
    # Compute accuracy of train and test
    r_square = r2_score(y_test,y_pred) 

    # compute accuracy
    sc = lr.score(X_train, y_train)
    ts = lr.score(X_test, y_test)

    # cross validation
    scores = cross_val_score(lr, X_train, y_train, scoring='r2', cv=5)


    #print("The accuracy of our model is {}%".format(round(r_square, 2) *100))
    acc.config(text=str("The accuracy of our Linear Regression model is {}%".format(round(r_square, 2) *100)))
    #print("Confusion matrix:" , cm)
    cmm.config(text=str("Confusion matrix: {}".format(cm)))
    #print(f"Train score: {sc}" , f"Test score: {ts}")
    ss.config(text=str("Train score: {} Test score: {}".format(sc , ts)))
    #print("Validation: ", scores)
    sss.config(text=str("Validation: {}".format(scores)))
    #print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
    mr.config(text=str("Mean absolute error = {}".format(round(sm.mean_absolute_error(y_test, y_pred), 2))))
    #print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
    ms.config(text=str("Mean squared error = {}".format(round(sm.mean_squared_error(y_test, y_pred), 2))))
    #print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2))
    ma.config(text=str("Median absolute error = {}".format(round(sm.median_absolute_error(y_test, y_pred), 2))))

def plot1(): 
    plt.scatter(y_test, y_pred, color='brown', label='Predicted Data') 
    plt.scatter(y_test, y_test, color='yellow', label='Actual Data') 
    plt.xlabel('Actual RUL') 
    plt.ylabel('Predicted RUL') 
    plt.title('Linear Regression') 
    plt.legend() 
    plt.show()



