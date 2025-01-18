from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np

class AdvancedModel:
    def __init__(self, data):
        self.data = data
        self.preprocessed = False

    def preprocess_data(self):
        label_encoders = {}
        for col in self.data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            label_encoders[col] = le

        scaler = StandardScaler()
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        if 'DEATH_EVENT' in numerical_cols:
            numerical_cols = numerical_cols.drop('DEATH_EVENT', errors='ignore')

       # self.data[numerical_cols] = scaler.fit_transform(self.data[numerical_cols])
       # self.data.loc[:, numerical_cols] = scaler.fit_transform(self.data[numerical_cols])

        X = self.data.drop('DEATH_EVENT', axis=1)
        y = self.data['DEATH_EVENT']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.preprocessed = True

    def balance_data(self):
        if not self.preprocessed:
            raise ValueError("Data must be preprocessed before balancing.")

        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def train_model(self):
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=self.y_train)

        # Create the logistic regression model with class weights
        self.model = LogisticRegression(max_iter=2000, class_weight={0: class_weights[0], 1: class_weights[1]})

        # Fit the model to the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        if not hasattr(self, 'model'):
            raise ValueError("Model must be trained before evaluation.")

        y_test_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_test_pred)
        precision = precision_score(self.y_test, y_test_pred, average='weighted')
        recall = recall_score(self.y_test, y_test_pred, average='weighted')
        f1 = f1_score(self.y_test, y_test_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def run_advanced_model(self):
        print("Preprocessing the data...")
        self.preprocess_data()

        print("Balancing the data...")
        self.balance_data()

        print("Training the improved model...")
        self.train_model()

        print("Evaluating the improved model...")
        self.evaluate_model()

