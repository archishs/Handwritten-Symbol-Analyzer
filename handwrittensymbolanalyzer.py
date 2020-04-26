import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read(csv("datasets/train.csv").as_matrix())
clf = DecisionTreeClassifier()

#training data
xtrain = data[0:21000,0]
train_label = data[:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest = data[21000:,1:]
actual_label = data[21000:.0]

#prediction and graphed output
d = xtest[8]
d.shape = (28,28)
pt.imshow(255 - d, cmap = 'gray')
print(clf.predict(  [xtest[0]]  ))
pt.show()
