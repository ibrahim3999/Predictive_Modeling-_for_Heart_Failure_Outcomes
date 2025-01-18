import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import BaseModel  as bm # נוודא ש-BaseModel מיובאת
import AdvancedModel as am
from sklearn.preprocessing import LabelEncoder  # ייבוא ה-LabelEncoder
import BasicNNModel as bnnm
import  AdvancedNNModel as annm
from sklearn.metrics import accuracy_score, precision_score, recall_score,classification_report
from sklearn.metrics import confusion_matrix


class DataHandler:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.features = None
        self.X = None
        self.y = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print("Data loaded successfully!")
        #print("Preview of the data:")
        #print(self.data.head())
        return self.data

    def clean_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Use load_data() first.")

        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            print(f"Found {missing_values} missing values. Removing rows with missing values...")
            self.data = self.data.dropna()
        else:
            print("No missing values found.")

        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            #print(f"Found {duplicates} duplicate rows. Removing duplicates...")
            self.data = self.data.drop_duplicates()
        else:
            pass
            #print("No duplicate rows found.")

        print("Data cleaned successfully!")
        return self.data

    def analyze_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Use load_data() first.")

        print("General Information:")
        print(self.data.info())
        print("\n")
        print("Basic Statistics:")
        print(self.data.describe())
        print("\n")
        print(f"Target Column Distribution ({self.target_column}):")
        print(self.data[self.target_column].value_counts())
        sns.countplot(x=self.target_column, data=self.data)
        plt.title(f"Distribution of {self.target_column}")
        plt.show()

        print("Correlation Matrix:")
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def prepare_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Use load_data() first.")

        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def run_baseline_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        base_model = bm.BaseModel(X_train, X_test, y_train, y_test)

        y_train_pred, y_test_pred = base_model.train_baseline_model()

        base_model.evaluate_baseline_model(y_test_pred)

    def convert_categorical_to_numeric(self):
        if self.data is None:
            raise ValueError("Data not loaded. Use load_data() first.")

        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                #print(f"Converting column '{column}' to numeric...")
                encoder = LabelEncoder()
                self.data[column] = encoder.fit_transform(self.data[column])
        #print("All categorical columns have been converted to numeric.")
        #print("\nPreview of data after conversion:")

        #print("Preview of the data after encoding:")
        #print(self.data.head())

    def run_nn_model(self):

        print("Preparing data for Neural Network model...")
        X_train, X_test, y_train, y_test = self.prepare_data()

        print("Initializing Basic Neural Network model...")
        nn_model = bnnm.BasicNNModel(X_train, y_train, X_test, y_test)

        print("Building the model...")
        nn_model.build_model(input_dim=X_train.shape[1])

        print("Training the model...")
        nn_model.train_model(epochs=50, batch_size=32)

        print("Evaluating the model...")
        nn_model.evaluate_model()

    def run_advanced_nn_model(self):

        print("Preparing data for Advanced Model...")

        X_train, X_test, y_train, y_test = self.prepare_data()

        print("Initializing Advanced Model...")
        advanced_model =annm.AdvancedNNModel(X_train, y_train, X_test, y_test)


        #advanced_model = annm.AdvancedNNModel(X_train, y_train, X_test, y_test)

        print("Building the model...")
        advanced_model.build_model()
#        advanced_model.analyze_features()

        print("Training the model...")
        advanced_model.train_model(epochs=50, batch_size=32)

        print("Evaluating the model...")
        advanced_model.evaluate_model()
        print("Confusion Matrix:")
        y_pred_classes = advanced_model.y_pred.argmax(axis=1)

        print(confusion_matrix(advanced_model.y_test, y_pred_classes))
