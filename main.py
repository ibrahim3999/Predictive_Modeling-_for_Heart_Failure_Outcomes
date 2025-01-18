import DataHandler as dh
import BaseModel as bl
import AdvancedModel as am
import  pandas as ps
import BasicNNModel as bnnm
if __name__ == '__main__':
    file_path = "data/heart_failure_dataset.csv"

    data_handler = dh.DataHandler(file_path, "DEATH_EVENT")

    data = data_handler.load_data()

    data_handler.convert_categorical_to_numeric()





    cleaned_data = data_handler.clean_data()
    ##print(data.isnull().sum())

#    data_handler.analyze_data()

    #print("baseline model running")
    data_handler.run_baseline_model()

#    data_handler.convert_categorical_to_numeric()

    ## advanced model
   ## X_train, X_test, y_train, y_test = data_handler.prepare_data()
#    print("AdvanceModel is running")
    model = am.AdvancedModel(data_handler.data)
    model.run_advanced_model()

    ## basic neural network model
    #print("basic neural network model is running")
    data_handler.run_nn_model()

    #print("Advanced neural network model is running")

    data_handler.run_advanced_nn_model()
