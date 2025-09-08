# Model trainer module
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

class ChurnModelRFTrainer:
    def __init__(
        self, X_train, y_train,
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight={0:5, 1:6},
        random_state=42,
        bootstrap=True
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.model = None

    def train(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            bootstrap=self.bootstrap
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluates the model and returns accuracy."""
        preds = self.model.predict_proba(X_test)
        preds = (preds[:, 1] >= threshold).astype(int)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"Random Forest Accuracy: {acc:.4f}")
        print(f"Random Forest F1 Score: {f1:.4f}")
        return preds

    def save(self, path):
        """Saves the trained model to a file."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    


class ChurnModelLGTrainer:
    """
    Trainer class for Logistic Regression churn prediction model.
    - Scales features using MinMaxScaler.
    - Trains LogisticRegression with class weighting and regularization.
    - Supports custom probability threshold for classification.
    Methods:
        train(): Fit pipeline and return trained model.
        predict(X_test): Predict churn labels for test data.
        evaluate(X_test, y_test): Print accuracy and F1 score, return predictions.
    """
    def __init__(
        self, X_train, y_train,
        max_iter=10000,
        class_weight={0:5, 1:6},
        random_state=42,
        penalty='l2',
        solver='liblinear',
        C=0.9,
        threshold=0.5
    ):
        """
        Initialize the trainer.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            max_iter (int): Maximum iterations for solver.
            class_weight (dict): Class weights for imbalance.
            random_state (int): Random seed.
            penalty (str): Regularization type.
            solver (str): Solver for optimization.
            C (float): Inverse regularization strength.
            threshold (float): Probability threshold for positive class.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.penalty = penalty
        self.solver = solver
        self.C = C
        self.threshold = threshold
        self.pipeline = None

    def train(self):
        """
        Train the logistic regression pipeline.
        Returns:
            Pipeline: Trained pipeline with scaler and logistic regression.
        """
        self.pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('logreg', LogisticRegression(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                penalty=self.penalty,
                solver=self.solver,
                C=self.C
            ))
        ])
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def predict(self, X_test):
        """
        Predict churn labels for test data using probability threshold.
        Args:
            X_test (pd.DataFrame): Test features.
        Returns:
            np.ndarray: Predicted labels.
        """
        proba = self.pipeline.predict_proba(X_test)
        preds = (proba[:, 1] >= self.threshold).astype(int)
        return preds

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        Prints accuracy and weighted F1 score.
        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels.
        Returns:
            np.ndarray: Predicted labels.
        """
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"Logistic Regression Accuracy: {acc:.4f}")
        print(f"Logistic Regression F1 Score: {f1:.4f}")
        return preds