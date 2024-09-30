import numpy as np
from sklearn.model_selection import train_test_split


class UniClassLearner:
    def __init__(self, insensitivity=0.1, iterations=1000):
        self.insensitivity = insensitivity
        self.iterations = iterations
        self.w = None

    def fit(self, X):
        self.w = X[0, :].copy()
        for i in range(self.iterations):
            # self.w = np.nan_to_num(self.w)
            mod_i = i % len(X)
            current_loss = self.suffer_loss(X[mod_i, :])
            if current_loss > 0:
                v_t = (X[mod_i] - self.w)
                norm_v_t = np.linalg.norm(v_t)
                if norm_v_t < self.insensitivity:  # Use insensitivity to avoid division by zero or small norms
                    continue  # Skip this iteration if v_t is too small
                v_t /= norm_v_t  # Normalize v_t
                tao_t = current_loss / (norm_v_t ** 2 + self.insensitivity)  # Use insensitivity as the margin
                self.w += tao_t * v_t

    def suffer_loss(self, x_t):
        norm = np.linalg.norm(self.w - x_t)
        return 0 if norm < self.insensitivity else norm + self.insensitivity

    def predict(self, X: np.ndarray):
        return np.array([self.suffer_loss(x) for x in X])

    def predict_single(self, x: np.ndarray):
        return self.suffer_loss(x)

    def score(self, X):
        predictions = self.predict(X)
        return 1 - np.mean(predictions)


def learn_uni_class_model(features):
    # Split the data into training and testing sets
    X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

    # Initialize the uni-class learner
    learner = UniClassLearner()

    # Train the uni-class model
    learner.fit(X_train)

    # Evaluate the model
    accuracy = learner.score(X_test)

    return learner, accuracy

