import pandas as pd
import sklearn.svm
import sklearn.metrics
import sklearn.model_selection

data = pd.read_csv(r'winequality-red.csv')

y = data.quality
x = data.drop(["quality", "pH"], axis=1)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

classifier = sklearn.svm.SVC(gamma=0.01, C=1000, kernel='rbf')
trainded_model = classifier.fit(x_train, y_train)

y_predClassifier = classifier.predict(x_test)


sklearn.metrics.accuracy_score(y_test, y_predClassifier)
print(sklearn.metrics.classification_report(y_test, y_predClassifier))



