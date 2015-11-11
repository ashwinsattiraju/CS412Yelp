import pandas as pd
from sklearn import decomposition, cross_validation
import numpy as np

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion

data = pd.read_csv("../data/dataset_review_combined.csv")
print(data)

# get x
# get y
data.dropna()
y = dfList = np.asarray(data['star'].tolist())
del data['star']
x = data.values


X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1, random_state=0)

pipeline = Pipeline([
  ('features', FeatureUnion([
    ('ngram_tf_idf', Pipeline([
      ('one_hot_encoding', OneHotEncoder(categorical_features=[2, 3, 4, 5, 7, 23, 29, 34, 45, 84])),
      ('anova', SelectKBest(f_regression, k=5)),
      ('pca', decomposition.PCA(n_components=5))

    ]))])),
  ('classifier', SVC())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(accuracy_score(y_test, y_pred))




