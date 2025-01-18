from sklearn.metrics import accuracy_score, precision_score, recall_score

class BaseModel:
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # train the model (The most common class)
    def train_baseline_model(self):
        most_common_class = self.y_train.mode()[0]
        y_train_pred = [most_common_class] * len(self.y_train)
        y_test_pred = [most_common_class] * len(self.y_test)

        return y_train_pred, y_test_pred

    # evaluate Accuracy,Precision,Recall
    def evaluate_baseline_model(self, y_test_pred):
        accuracy = accuracy_score(self.y_test, y_test_pred)
        precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=1)
        recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
