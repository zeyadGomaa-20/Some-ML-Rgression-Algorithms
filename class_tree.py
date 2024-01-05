import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import sklearn.metrics as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score


def classT(acc, cmm, ss, sss, mr, ms, ma):

    global y_pred
    global y_test
    # dataset
    dataset = pd.read_csv("dataset.csv")  
    #dataset.head(8)


    # first column
    X = dataset[['Discharge Time (s)','Decrement 3.6-3.4V (s)','Max. Voltage Dischar. (V)','Min. Voltage Charg. (V)','Time at 4.15V (s)','Time constant current (s)','Charging time (s)']].values


    # last column
    y = dataset.RUL


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) # 75% training and 25% test

    ct = DecisionTreeClassifier(criterion="entropy", max_depth=50)

    # Train Decision Tree Classifer
    ct = ct.fit(X_train,y_train)


    # compute scores
    sc = ct.score(X_train, y_train)
    ts = ct.score(X_test, y_test)

    #Predict the response for test dataset
    y_pred = ct.predict(X_test)

    # cross validation
    scores = cross_val_score(ct, X_train, y_train, scoring='r2', cv=5)

    con = confusion_matrix(y_test , y_pred)
    r_square = r2_score(y_test,y_pred) 

    #print("The accuracy of our model is {}%".format(round(r_square, 2) *100))
    acc.config(text=str("The R Square of our Classification Tree model is {}%".format(round(r_square, 2) )*100))
    #print("Confusion matrix:" , cm)
    cmm.config(text=str("Confusion matrix: {}".format(con)))
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

def plot3():
    plt.scatter(y_test, y_pred, color='brown', label='Predicted Data') 
    plt.scatter(y_test, y_test, color='yellow', label='Actual Data') 
    plt.xlabel('Actual RUL') 
    plt.ylabel('Predicted RUL') 
    plt.title('Classfication tree') 
    plt.legend() 
    plt.show()