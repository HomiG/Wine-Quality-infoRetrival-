import pandas as pd
import random
import sklearn.svm
import sklearn.metrics
import sklearn.model_selection
from numpy import nansum
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r'winequality-red.csv')

y = data.quality
x = data.drop("quality", axis=1)


multipliers = [10, 1000, 100, 10, 1000, 1, 1, 100000, 100, 100, 10, 1]

# Erase decimals
for count, i in enumerate(x):
    x[i] = x[i] * multipliers[count]

# cast to int
x = x.astype('int64')
y = (y * 100).astype('int64')


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)




################################################### Erase PH values
y = x_train.pH
x = x_train.drop('pH', axis=1)


# Find Random Locations
randomLocations = random.sample(range(0, len(y)), int(len(y) / 3))
randomLocations.sort()

# Erase pH Value from main list, based on random location
for i in randomLocations:
    y[i] = 0
###################################################
no_ph = x_train.drop('pH', axis=1)
no_ph = no_ph.to_numpy()

x1_train = []
y1_train = []
x1_test = []
y1_test = []

x_train2 = x_train.values

for i in range(len(x_train)):
    if i not in randomLocations:
        x1_train.append(no_ph[i])
        y1_train.append(x_train2[i][8])
    if i in randomLocations:
        x1_test.append(no_ph[i])
        y1_test.append(x_train2[i][8])




logisticRegr = sklearn.linear_model.LogisticRegression(max_iter=10000)


logisticRegr.fit(x1_train, y1_train)

predictions = logisticRegr.predict(x1_test)

j = 0
for i in x_train2:
    if i[8] in randomLocations:
        i[8] = predictions[j]
        j += 1



classifier = sklearn.svm.SVC(gamma=0.01, C=1000, kernel='rbf')
trainded_model = classifier.fit(x_train2, y_train)

y_predClassifier = classifier.predict(x_test)

sklearn.metrics.accuracy_score(y_test, y_predClassifier)
print(sklearn.metrics.classification_report(y_test, y_predClassifier))



