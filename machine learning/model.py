
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#load csv

df = pd.read_csv('iris.csv')
print(df.head ())
print("..................................................................................................................................")
print(df.tail())

#independent and dependent variabls

x = df[["Sepal_Length", "Sepal_Width" , "Petal_Length" , "Petal_Width"]]
y = df["Class"]

#split the dataset into train and test
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size  = 0.3 , random_state = 50)

#feature scalling 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test =sc.transform(x_test)

#model selection 

classifier = RandomForestClassifier()


#fir the model
classifier.fit(x_train , y_train)

#make pickle file of our model
pickle.dump(classifier, open("model.pkl" , "wb"))