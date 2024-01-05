import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as sm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def knr(acc , cmm , ss , sss , mr , ms , ma):
    global y_pred
    global y_test

    # dataset
    dataset = pd.read_csv("dataset.csv")  
    #dataset.head(8)


    # first columns
    X = dataset[['Discharge Time (s)','Decrement 3.6-3.4V (s)','Max. Voltage Dischar. (V)','Min. Voltage Charg. (V)','Time at 4.15V (s)','Time constant current (s)','Charging time (s)']].values


    # last column
    y = dataset.iloc[:, 8]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)#بيتمرن علي 75 و بيتيست علي ال 25

    # KNN classifier
    knn = KNeighborsRegressor()


    # train t   he knn rogressor
    knn.fit(X_train, y_train)


    # make predictions
    y_pred = knn.predict(X_test)

    # In this case, we can simply use the average value of y_pred
    threshold = np.mean(y_pred)

    # Convert predictions to binary (1 or 0)
    y_pred_binary = np.zeros(y_pred.shape)
    y_pred_binary[y_pred > threshold] = 1

    # Compute confusion matrix
    cm = sm.confusion_matrix(y_test, y_pred_binary)

    # Compute accuracy score
    r_square = r2_score(y_test,y_pred) 

    # compute train and test accuracy
    sc = knn.score(X_train, y_train)
    ts = knn.score(X_test, y_test)

    # cross validation
    scores = cross_val_score(knn, X_train, y_train, scoring='r2', cv=5)

    #print("The accuracy of our model is {}%".format(round(r_square, 2) *100))
    acc.config(text=str("The accuracy of our KNeighbor Regression model is {}%".format(round(r_square, 2) *100)))
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
    return y_pred

def plot2():
    plt.scatter(y_test, y_pred, color='brown', label='Predicted Data') 
    plt.scatter(y_test, y_test, color='yellow', label='Actual Data') 
    plt.xlabel('Actual RUL') 
    plt.ylabel('Predicted RUL') 
    plt.title('Knieghbor Regression') 
    plt.legend() 
    plt.show()

