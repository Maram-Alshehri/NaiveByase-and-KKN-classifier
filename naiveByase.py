#Maram Alshehri - Computer Scince Student 


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#loading dataset from local file
iris_data= pd.read_csv('iris.csv')
iris_data.info()


#getting the features names only not the target
iris = list(iris_data.columns.values.tolist())[:]
#print ("The dataset has the following Features:", iris )

#spilt data into test and train 
x=iris_data[iris]
print(x)
train, test = train_test_split(x,test_size=0.30, random_state=0)


# Split training data by class
data_setosa = train[train['species'] == "setosa"]
data_virginica = train[train['species'] == "virginica"]
data_versicolor = train[train['species'] == "versicolor"]

#-----------------------------------------------------------------------------------------------------------------
# compute the mean vector and variance of the feature in each class.
#class 1 data_virginica
import statistics
###########mean
mean_virginica_sl =statistics.mean(data_virginica["sepal_length"])
mean_virginica_sw = statistics.mean(data_virginica["sepal_width"])
mean_virginica_pl = statistics.mean(data_virginica["petal_length"])
mean_virginica_pw = statistics.mean(data_virginica["petal_width"])

#######variance

variance_virginica_sl = statistics.variance(data_virginica["sepal_length"])
variance_virginica_sw = statistics.variance(data_virginica["sepal_width"])
variance_virginica_pl = statistics.variance(data_virginica["petal_length"])
variance_virginica_pw = statistics.variance(data_virginica["petal_width"])

#-------------------------------------------------------------------------
#class 2 data_versicolor
#mean
mean_versicolor_sl = statistics.mean(data_versicolor["sepal_length"])
mean_versicolor_sw = statistics.mean(data_versicolor["sepal_width"])
mean_versicolor_pl = statistics.mean(data_versicolor["petal_length"])
mean_versicolor_pw = statistics.mean(data_versicolor["petal_width"])
  
#######variance
variance_versicolor_sl = statistics.variance(data_versicolor["sepal_length"])
variance_versicolor_sw= statistics.variance(data_versicolor["sepal_width"])
variance_versicolor_pl = statistics.variance(data_versicolor["petal_length"])
variance_versicolor_pw = statistics.variance(data_versicolor["petal_width"])

#------------------------------------------------------------------------
#class 3 data_setosa
#mean
mean_setosa_sl = statistics.mean(data_setosa["sepal_length"])
mean_setosa_sw = statistics.mean(data_setosa["sepal_width"])
mean_setosa_pl = statistics.mean(data_setosa["petal_length"])
mean_setosa_pw = statistics.mean(data_setosa["petal_width"])

#######variance
variance_setosa_sl = statistics.variance(data_setosa["sepal_length"])
variance_setosa_sw = statistics.variance(data_setosa["sepal_width"])
variance_setosa_pl = statistics.variance(data_setosa["petal_length"])
variance_setosa_pw = statistics.variance(data_setosa["petal_width"])


#-----------------------------------------------------------------------------------------------------------------
##colculate the prior probability
virginica_prior = len(data_virginica)/len(train)
versicolor_prior =len(data_versicolor)/len(train) 
setosa_prior = len(data_setosa)/len(train)


#-----------------------------------------------------------------------------------------------------------------
#function to calulate the gausian probability
def gausianP_x_given_y(x, y_mean, y_variance):
    p = 1/(np.sqrt(2*np.pi )*y_variance) * np.exp((-(x-y_mean)**2)/(2*((y_variance)**2)))
                                                                    
    return p




#-----------------------------------------------------------------------------------------------------------------
#gausian probability #class 1 data_virginica

def class1(x1,x2,x3,x4):
    class_virginica = virginica_prior * gausianP_x_given_y(x1, mean_virginica_sl, variance_virginica_sl)*gausianP_x_given_y(x2, mean_virginica_sw, variance_virginica_sw)*gausianP_x_given_y(x3, mean_virginica_pl, variance_virginica_pl)*gausianP_x_given_y(x4, mean_virginica_pw, variance_virginica_pw)
   
    return class_virginica

#gausian probability #class 2 data_versicolor
def class2(x1,x2,x3,x4):
    class_versicolor = versicolor_prior * gausianP_x_given_y(x1, mean_versicolor_sl, variance_versicolor_sl)*gausianP_x_given_y(x2, mean_versicolor_sw, variance_versicolor_sw)*gausianP_x_given_y(x3, mean_versicolor_pl, variance_versicolor_pl)*gausianP_x_given_y(x4, mean_versicolor_pw, variance_versicolor_pw)
   
    return class_versicolor



#gausian probability #class 3 data_setosa
def class3(x1,x2,x3,x4):
    class_setosa = setosa_prior * gausianP_x_given_y(x1, mean_setosa_sl, variance_setosa_sl)*gausianP_x_given_y(x2, mean_setosa_sw, variance_setosa_sw)*gausianP_x_given_y(x3, mean_setosa_pl, variance_setosa_pl)*gausianP_x_given_y(x4, mean_setosa_pw, variance_setosa_pw)
    return class_setosa

#----------------------------------------------------------------------------------------------------------------------------
# prediction on the testing set.

from copy import deepcopy

ytest = train['species']
yPrediction = deepcopy(ytest)#


for i in range(len(test)):
    
    c1=class1(test.values[i][0],test.values[i][1],test.values[i][2],test.values[i][3])
    c2=class2(test.values[i][0],test.values[i][1],test.values[i][2],test.values[i][3]) 
    c3=class3(test.values[i][0],test.values[i][1],test.values[i][2],test.values[i][3])
    #find the max probability of classes
    maxNum = max(c1,c2,c3)
    if(maxNum ==c1):
         yPrediction.values[i] = "virginica"
         
    elif( maxNum == c2):
         yPrediction.values[i] = "versicolor"
        
         
    else :
         yPrediction.values[i] = "setosa"
      

#---------------------------------------------------------------------------------------------------------------------------
#Compute the accuracy, precision, and recall.
######
from sklearn.metrics import classification_report
print(classification_report(ytest, yPrediction))
print(ytest)
print(yPrediction)
#---------------------------------------------------------------------------------------------------------------------------
#plotting a scatter plot with class label
#---------------------------
# import seaborn
import seaborn as sb
sb.set(font_scale=1.35, style="ticks") 
plot = sb.pairplot(iris_data, hue='species',palette='husl')

plt.legend()
plt.show()


























