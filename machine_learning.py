import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

student_df = pd.read_csv('student-mat.csv', sep=';')

student_df = student_df[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

X = np.array(student_df.drop([predict],1))      #This returns a new df without G3

y = np.array(student_df[predict])               #This represents all our labels



best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
for _ in range(30):             #How many times to iterate to find possible good predict value to save model
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #Used for the different test cases

    linear = linear_model.LinearRegression()                            

    linear.fit(x_train, y_train)

    accu = linear.score(x_test, y_test)             

    print(accu)

    if accu > best:                                                                 #The best case is saved 
        best = accu
        print(best)
        with open("studentmodel.pickle", "wb") as f:
             pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")       #This enables saving of our best model
linear = pickle.load(pickle_in)


print("Coef: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predcts = linear.predict(x_test)


for x in range(len(predcts)):
    print(predcts[[x]], x_test[x], y_test[x])


style.use("ggplot")
p = 'G1'
plt.scatter(student_df[p], student_df['G3'])

plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()