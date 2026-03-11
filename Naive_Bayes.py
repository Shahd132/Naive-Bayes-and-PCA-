import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

############################################################
# GAUSSIAN NAIVE BAYES FROM SCRATCH
############################################################

class GaussianNB_FromScratch:

    def fit(self, X, y):

        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:

            X_c = X[y == c]

            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian(self, x, mean, var):

        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict(self, X):

        predictions = []

        for x in X:

            posteriors = []

            for c in self.classes:

                prior = np.log(self.priors[c])

                likelihood = np.sum(
                    np.log(self.gaussian(x, self.mean[c], self.var[c]))
                )

                posterior = prior + likelihood

                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)


############################################################
# CATEGORICAL NAIVE BAYES FROM SCRATCH
############################################################

class CategoricalNB_FromScratch:

    def fit(self, X, y):

        self.classes = np.unique(y)
        self.feature_probs = {}
        self.class_probs = {}

        n_samples = len(y)

        for c in self.classes:

            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / n_samples

            self.feature_probs[c] = []

            for i in range(X.shape[1]):

                values, counts = np.unique(X_c[:, i], return_counts=True)

                probs = {}

                for v, count in zip(values, counts):

                    probs[v] = (count + 1) / (len(X_c) + len(values))  # Laplace smoothing
                self.feature_probs[c].append(probs)

    def predict(self, X):

        predictions = []

        for x in X:

            posteriors = []

            for c in self.classes:

                posterior = np.log(self.class_probs[c])

                for i, val in enumerate(x):

                    probs = self.feature_probs[c][i]

                    if val in probs:
                        posterior += np.log(probs[val])
                    else:
                        posterior += np.log(1e-6)

                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)


############################################################
# DATASET 1 — NUMERICAL (IRIS)
############################################################

print("\n========== NUMERICAL DATASET (IRIS) ==========")

iris = load_iris()

X_num = iris.data
y_num = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_num, y_num, test_size=0.2, random_state=42
)

# BASELINE
model = GaussianNB_FromScratch()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nBaseline Results")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

cm = confusion_matrix(y_test, pred)

sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
############################################################
# FEATURE SELECTION (NUMERICAL)
############################################################

selector = SelectKBest(score_func=f_classif, k=2)

X_train_fs = selector.fit_transform(X_train, y_train)
X_test_fs = selector.transform(X_test)

model_fs = GaussianNB_FromScratch()
model_fs.fit(X_train_fs, y_train)

pred_fs = model_fs.predict(X_test_fs)

print("\nFeature Selection Results")
print("Accuracy:", accuracy_score(y_test, pred_fs))
print(classification_report(y_test, pred_fs))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_fs))
cm = confusion_matrix(y_test, pred_fs)

sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

############################################################
# DATASET 2 — CATEGORICAL (CAR EVALUATION)
############################################################

print("\n========== CATEGORICAL DATASET (CAR) ==========")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

columns = ["buying","maint","doors","persons","lug_boot","safety","class"]

df = pd.read_csv(url, names=columns)

X_cat = df.drop("class", axis=1)
y_cat = df["class"]

# Encode categorical features

for col in X_cat.columns:

    le = LabelEncoder()
    X_cat[col] = le.fit_transform(X_cat[col])

le_y = LabelEncoder()
y_cat = le_y.fit_transform(y_cat)

X_cat = X_cat.values
y_cat = y_cat

X_train, X_test, y_train, y_test = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=42
)

# BASELINE
model_cat = CategoricalNB_FromScratch()
model_cat.fit(X_train, y_train)

pred_cat = model_cat.predict(X_test)

print("\nBaseline Results")
print("Accuracy:", accuracy_score(y_test, pred_cat))
print(classification_report(y_test, pred_cat))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_cat))
cm = confusion_matrix(y_test, pred_cat)

sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

############################################################
# FEATURE SELECTION (CATEGORICAL)
############################################################

selector = SelectKBest(score_func=chi2, k=3)

X_train_fs = selector.fit_transform(X_train, y_train)
X_test_fs = selector.transform(X_test)

model_cat_fs = CategoricalNB_FromScratch()
model_cat_fs.fit(X_train_fs, y_train)

pred_cat_fs = model_cat_fs.predict(X_test_fs)

print("\nFeature Selection Results")
print("Accuracy:", accuracy_score(y_test, pred_cat_fs))
print(classification_report(y_test, pred_cat_fs))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_cat_fs))

cm = confusion_matrix(y_test, pred_cat_fs)

sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()