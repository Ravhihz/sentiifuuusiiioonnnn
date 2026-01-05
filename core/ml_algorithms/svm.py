import numpy as np
from typing import Literal


class SVM:
    """Support Vector Machine (binary) from scratch"""

    def __init__(
        self,
        kernel: Literal["linear", "rbf", "poly"] = "rbf",
        C: float = 1.0,
        gamma="auto",
        degree: int = 3,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_labels = None

    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T

        if self.kernel == "rbf":
            gamma = 1.0 / X1.shape[1] if self.gamma == "auto" else self.gamma
            dist = (
                np.sum(X1 ** 2, axis=1).reshape(-1, 1)
                + np.sum(X2 ** 2, axis=1)
                - 2 * X1 @ X2.T
            )
            return np.exp(-gamma * dist) # type: ignore

        if self.kernel == "poly":
            return (X1 @ X2.T + 1) ** self.degree

        raise ValueError("Kernel tidak dikenali")

    def fit(self, X, y):
        n = X.shape[0]
        self.alpha = np.zeros(n)
        self.b = 0

        K = self._kernel(X, X)

        passes = 0
        max_passes = 50
        tol = 1e-3

        while passes < max_passes:
            changed = 0
            for i in range(n):
                Ei = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]

                if (y[i] * Ei < -tol and self.alpha[i] < self.C) or (
                    y[i] * Ei > tol and self.alpha[i] > 0
                ):
                    j = np.random.choice([x for x in range(n) if x != i])
                    Ej = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]

                    ai_old, aj_old = self.alpha[i], self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - aj_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (aj_old - self.alpha[j])

                    b1 = (
                        self.b
                        - Ei
                        - y[i] * (self.alpha[i] - ai_old) * K[i, i]
                        - y[j] * (self.alpha[j] - aj_old) * K[i, j]
                    )
                    b2 = (
                        self.b
                        - Ej
                        - y[i] * (self.alpha[i] - ai_old) * K[i, j]
                        - y[j] * (self.alpha[j] - aj_old) * K[j, j]
                    )

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    changed += 1

            passes = passes + 1 if changed == 0 else 0

        sv = self.alpha > 1e-5
        self.support_vectors = X[sv]
        self.support_labels = y[sv]
        self.alpha = self.alpha[sv]

        return self

    def decision_function(self, X):
        K = self._kernel(X, self.support_vectors)
        return np.sum(self.alpha * self.support_labels * K.T, axis=1) + self.b # type: ignore

    def predict(self, X):
        return np.sign(self.decision_function(X))


class MultiClassSVM:
    """One-vs-Rest Multi-class SVM"""

    def __init__(self, **svm_params):
        self.svm_params = svm_params
        self.models = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            y_bin = np.where(y == c, 1, -1)
            model = SVM(**self.svm_params)
            model.fit(X, y_bin)
            self.models[c] = model
        return self

    def predict(self, X):
        scores = np.column_stack(
            [model.decision_function(X) for model in self.models.values()]
        )
        return self.classes[np.argmax(scores, axis=1)] # type: ignore
