import csv
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from hand_tracker import FEATURE_COUNT


class ASLNeuralNetwork:
    """Manages landmark dataset persistence, MLP neural network training, and inference."""

    def __init__(self, dataset_name="asl_dataset.csv", model_name="asl_nn_model.pkl"):
        base_dir = Path(__file__).resolve().parent
        self.dataset_path = base_dir / dataset_name
        self.model_path = base_dir / model_name
        self.model = None
        self.status_message = ""

        self._load_or_initialize()

    def _load_or_initialize(self):
        if self.model_path.exists():
            loaded = joblib.load(self.model_path)
            if getattr(loaded, "n_features_in_", FEATURE_COUNT) == FEATURE_COUNT:
                self.model = loaded
                classes = ", ".join(self.model.classes_)
                self.status_message = f"Loaded NN model ({classes})"
                return
        self.train()

    def save_sample(self, label, features):
        """Appends a single labeled 79-feature sample to the CSV dataset."""
        with open(self.dataset_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + features.tolist())

    def train(self):
        """Trains a 2-hidden-layer MLPClassifier on all recorded 79-feature samples."""
        if not self.dataset_path.exists():
            self.model = None
            self.status_message = "No dataset yet. Press TAB to record letters."
            return False

        X, y = [], []
        legacy_rows = 0
        expected_cols = FEATURE_COUNT + 1

        with open(self.dataset_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == expected_cols:
                    y.append(row[0])
                    X.append([float(val) for val in row[1:]])
                elif len(row) == 43:
                    legacy_rows += 1

        if legacy_rows > 0 and not X:
            self.model = None
            self.status_message = "Old 42-feat CSV found. Delete asl_dataset.csv & re-record."
            return False

        unique_labels = sorted(set(y))
        if len(unique_labels) < 2:
            self.model = None
            have_str = ", ".join(unique_labels) if unique_labels else "none"
            self.status_message = f"Need 2+ letters to train (have: {have_str})."
            return False

        # Architecture: 79 inputs -> 128 ReLU -> 64 ReLU -> N letter classes
        pipeline = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.001,
                max_iter=800,
                random_state=42,
            ),
        )
        pipeline.fit(np.array(X, dtype=np.float32), np.array(y))
        joblib.dump(pipeline, self.model_path)

        self.model = pipeline
        self.status_message = f"Model ready ({len(y)} samples | Letters: {', '.join(unique_labels)})"
        return True

    def predict(self, features):
        """Returns (predicted_letter, confidence) or (None, 0.0) if untrained."""
        if self.model is None or features is None:
            return None, 0.0

        probs = self.model.predict_proba([features])[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        letter = str(self.model.classes_[best_idx])
        return letter, confidence