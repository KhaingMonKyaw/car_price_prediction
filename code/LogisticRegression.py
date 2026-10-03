import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time

class LogisticRegression:
    """
    Multinomial logistic regression with optional L2 (Ridge) regularization.

    X: (m, n) feature matrix
    Y: (m, k) one-hot target matrix
    W: (n, k) parameter matrix
    """

    def __init__(
        self, k, n, method="batch", alpha=0.001, max_iter=5000,
        use_ridge=False, lambda_=0.1
    ):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.use_ridge = use_ridge
        self.lambda_ = lambda_
        self.W = None
        self.losses = []

    def softmax(self, scores):
        # Numerically stable softmax: subtract each row's maximum.
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def h_theta(self, X, W=None):
        if W is None:
            W = self.W
        return self.softmax(X @ W)

    def _loss_and_gradient(self, X, Y):
        m = X.shape[0]
        H = self.h_theta(X)
        eps = 1e-15

        # Mean cross-entropy keeps the loss scale comparable across batch sizes.
        loss = -np.sum(Y * np.log(np.clip(H, eps, 1.0))) / m
        grad = (X.T @ (H - Y)) / m

        if self.use_ridge:
            # L2 penalty and its derivative. The intercept is column 0 in X,
            # so exclude W[0, :] from regularization when an intercept is present.
            if X.shape[1] == self.n and self.n > 0:
                penalty_weights = self.W[1:, :] if self.n > 1 else self.W
                penalty_gradient = np.zeros_like(self.W)
                if self.n > 1:
                    penalty_gradient[1:, :] = 2 * self.lambda_ * self.W[1:, :]
                else:
                    penalty_gradient = 2 * self.lambda_ * self.W
            else:
                penalty_weights = self.W
                penalty_gradient = 2 * self.lambda_ * self.W

            loss += self.lambda_ * np.sum(penalty_weights ** 2)
            grad += penalty_gradient

        return loss, grad

    def fit(self, X, Y):
           X = np.asarray(X, dtype=float)
           Y = np.asarray(Y, dtype=float)
   
           if X.ndim != 2 or Y.ndim != 2:
               raise ValueError("X and Y must both be 2D arrays; Y must be one-hot encoded.")
           if X.shape[0] != Y.shape[0]:
               raise ValueError("X and Y must contain the same number of samples.")
           if X.shape[1] != self.n or Y.shape[1] != self.k:
               raise ValueError("Input dimensions do not match n or k in the constructor.")
           if self.method not in {"batch", "minibatch", "sto"}:
               raise ValueError('method must be "batch", "minibatch", or "sto".')
   
           rng = np.random.default_rng()
           self.W = rng.normal(0, 0.01, size=(self.n, self.k))
           self.losses = []
           start_time = time.time()
           m = X.shape[0]
   
           for iteration in range(self.max_iter):
               if self.method == "batch":
                   batch_X, batch_Y = X, Y
               elif self.method == "minibatch":
                   batch_size = max(1, int(0.3 * m))
                   indices = rng.choice(m, size=batch_size, replace=False)
                   batch_X, batch_Y = X[indices], Y[indices]
               else:  # stochastic gradient descent
                   idx = rng.integers(0, m)
                   batch_X, batch_Y = X[idx:idx + 1], Y[idx:idx + 1]
   
               loss, grad = self._loss_and_gradient(batch_X, batch_Y)
               self.losses.append(loss)
               self.W -= self.alpha * grad
   
               if iteration % 500 == 0:
                   print(f"Loss at iteration {iteration}: {loss:.6f}")
   
           print(f"Time taken: {time.time() - start_time:.2f} seconds")
           return self
   
    def gradient(self, X, Y):
        """Return the current loss and gradient (useful for checking by hand)."""
        return self._loss_and_gradient(np.asarray(X, dtype=float),
                                        np.asarray(Y, dtype=float))

    def softmax_grad(self, X, error):
        return X.T @ error

    def predict_proba(self, X):
        return self.h_theta(np.asarray(X, dtype=float))

    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test), axis=1)

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train loss")
        plt.title("Training loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    # ----- Per-class metrics -----
    @staticmethod
    def _as_labels(y):
        y = np.asarray(y)
        # Accept either integer labels or one-hot matrices.
        return np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int).ravel()

    def accuracy(self, y_true, y_pred):
        yt, yp = self._as_labels(y_true), self._as_labels(y_pred)
        if yt.size == 0:
            return 0.0
        return float(np.mean(yt == yp))

    def _class_counts(self, y_true, y_pred, c):
        yt, yp = self._as_labels(y_true), self._as_labels(y_pred)
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        support = int(np.sum(yt == c))
        return tp, fp, fn, support

    def precision(self, y_true, y_pred, c):
        tp, fp, _, _ = self._class_counts(y_true, y_pred, c)
        return tp / (tp + fp) if tp + fp else 0.0

    def recall(self, y_true, y_pred, c):
        tp, _, fn, _ = self._class_counts(y_true, y_pred, c)
        return tp / (tp + fn) if tp + fn else 0.0

    def f1_score(self, y_true, y_pred, c):
        p = self.precision(y_true, y_pred, c)
        r = self.recall(y_true, y_pred, c)
        return 2 * p * r / (p + r) if p + r else 0.0

    # ----- Macro averages: equal weight for each class -----
    def macro_precision(self, y_true, y_pred):
        return float(np.mean([self.precision(y_true, y_pred, c)
                                for c in range(self.k)]))

    def macro_recall(self, y_true, y_pred):
        return float(np.mean([self.recall(y_true, y_pred, c)
                                for c in range(self.k)]))

    def macro_f1(self, y_true, y_pred):
        return float(np.mean([self.f1_score(y_true, y_pred, c)
                                for c in range(self.k)]))

    # ----- Weighted averages -----
    # Standard weighted averages use class support / total support as weights.
    def weighted_precision_standard(self, y_true, y_pred):
        yt = self._as_labels(y_true)
        total = len(yt)
        return (sum(np.sum(yt == c) * self.precision(yt, y_pred, c)
                    for c in range(self.k)) / total) if total else 0.0

    def weighted_recall_standard(self, y_true, y_pred):
        yt = self._as_labels(y_true)
        total = len(yt)
        return (sum(np.sum(yt == c) * self.recall(yt, y_pred, c)
                    for c in range(self.k)) / total) if total else 0.0

    def weighted_f1_standard(self, y_true, y_pred):
        yt = self._as_labels(y_true)
        total = len(yt)
        return (sum(np.sum(yt == c) * self.f1_score(yt, y_pred, c)
                    for c in range(self.k)) / total) if total else 0.0

    # weighted sum divided by number of classes.
    # This is intentionally separate because it differs from sklearn's definition.
    def weighted_precision(self, y_true, y_pred):
        return self.weighted_precision_standard(y_true, y_pred) / self.k

    def weighted_recall(self, y_true, y_pred):
        return self.weighted_recall_standard(y_true, y_pred) / self.k

    def weighted_f1(self, y_true, y_pred):
        return self.weighted_f1_standard(y_true, y_pred) / self.k