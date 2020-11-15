import pandas as pd
import random
import sklearn.svm
import sklearn.metrics
import sklearn.model_selection
from numpy import nansum

data = pd.read_csv(r'winequality-red.csv')

y = data.quality
x = data.drop("quality", axis=1)

# Find Random Locations
randomLocations = random.sample(range(0, len(x.pH)), int(len(x.pH) / 3))
randomLocations.sort()

# Erase pH Value from main list, based on random location
for i in randomLocations:
    x.pH[i] = None

# Calculate Mean
mean = round(nansum(x.pH) / (2 / 3 * len(x.pH)))

# Replace With Mean
for i in randomLocations:
    x.pH[i] = mean



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

classifier = sklearn.svm.SVC(gamma=0.01, C=1000, kernel='rbf')
trainded_model = classifier.fit(x_train, y_train)

y_predClassifier = classifier.predict(x_test)

sklearn.metrics.accuracy_score(y_test, y_predClassifier)
print(sklearn.metrics.classification_report(y_test, y_predClassifier))
