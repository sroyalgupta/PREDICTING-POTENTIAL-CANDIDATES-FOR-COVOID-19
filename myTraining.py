import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('data.csv')
train,test = train_test_split(df,test_size=0.2,random_state=42)
train_set_x = train[['fever','bodyPain','age','runnyNode','diffBreath']].to_numpy()
test_set_x = test[['fever','bodyPain','age','runnyNode','diffBreath']].to_numpy()
train_set_y = train[['infectionProb']].to_numpy().reshape(445,)
test_set_y = test[['infectionProb']].to_numpy().reshape(112,)
clf = LogisticRegression()
clf.fit(train_set_x,train_set_y)

file = open('model.pkl', 'wb')

# dump information to that file
pickle.dump(clf, file)

# close the file
file.close()

