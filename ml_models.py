import numpy as np


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
    
####PCA from scratch#########

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):

        # mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # select top components
        self.components = self.eigenvectors[:, :self.n_components]

    def transform(self, X):

        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)