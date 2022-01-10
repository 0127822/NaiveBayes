from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

dataset = datasets.load_iris()
model = GaussianNB()
model.fit(dataset.data , dataset.target)
print(model)
expect = dataset.target
predict = model.predict(dataset.data)
print(metrics.classification_report(expect,predict))
