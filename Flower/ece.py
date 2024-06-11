import numpy as np

def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


def expected_calibration_error_class(samples, true_labels, M=5):

    num_classes = 10
    ece_per_class = np.zeros(num_classes)
    
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Iterate over each class
    for class_label in range(num_classes):
        # Filter samples and true labels for the current class
        class_true_labels = true_labels == class_label

        class_samples = samples[class_true_labels]

        # Get max probability per sample
        confidences = np.max(class_samples, axis=1)

        # Get predictions from confidences (positional in this case)
        predicted_label = np.argmax(class_samples, axis=1)


        # Get a boolean list of correct/false predictions for the current class
        accuracies = predicted_label == class_label
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample is in bin m (between bin lower & upper)
            in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())

            # Calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prob_in_bin = in_bin.mean()
            if prob_in_bin.item() > 0:
                # Get the accuracy of bin m: acc(Bm)
                accuracy_in_bin = accuracies[in_bin].mean()
                # Get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = confidences[in_bin].mean()
                # Calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prob_in_bin

        # Store ECE for the current class
        ece_per_class[class_label] = ece
        

    return ece_per_class


import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from logging import ERROR, INFO, WARN
from flwr.common.logger import log
from tqdm import tqdm
import pandas as pd
import csv
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve

import threading
import flwr as fl
from tutorial import client
import pickle
from sklearn.calibration import CalibratedClassifierCV
import joblib

log(
        INFO,
        "Creazione dataset",
    )

df = dd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file4.csv')
# value_counts_attack = [ 3781419, 6099258, 1153323, 2026234, 2455020, 712609,  684465, 7514, 3425, 16809]    
# l = []
# for i in range(10):
#     k = df[(df['Attack_label'] == i)]
#     k = k.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Attack_label'])
#     selected_rows = k.sample(frac=10/value_counts_attack[i], random_state=42).compute()
#     l.append(selected_rows)

# with open('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/selected_rows.pkl', 'wb') as f:
#     pickle.dump(l, f)

#df = df[(df['Attack_label'] < 7)]
# X = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Attack_label'])
# Y = df['Attack_label']

# # # # tenendo lo stesso random state dovrei ottenere gli stessi dati che ho usato nel main.py per creare le partizioni
# X_drop, X_keep, y_drop, y_keep = train_test_split(X, Y, test_size=0.2, shuffle = True, random_state=42)

# # # se lavoro con (X_drop;  y_drop) ho dati nuovi non utilizzati per la fase di training
# train_x, test_x, train_y, test_y = train_test_split(X_drop, y_drop, test_size = 0.1, shuffle= True, random_state=42) #circa 1.35M per test
# train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.05, shuffle= True, random_state=42) #circa 0.6M per validation
# # # #train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, shuffle= True, random_state=42) 
# # # #train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.1, shuffle= True, random_state=42) 


# # train_DMatrix = xgb.DMatrix(train_x, train_y)
# valid_Dmatrix = xgb.DMatrix(valid_x, valid_y)
# file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/VALID_MATRIX.bin"
# valid_Dmatrix.save_binary(file_path_valid)
# # test_Dmatrix = xgb.DMatrix(test_x, label=test_y)


# log(
#         INFO,
#         "load model e classification_report",
#     )

# valutazione modello prima di fare calibrazione
model_federated = xgb.XGBClassifier()
model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/algoritmo_alpha_0,3/model_federato.json')

# # calcolo prestazioni su dataset più grande
# y_pred = model_federated.predict(test_Dmatrix.get_data())
# cl_rp = classification_report(test_y, y_pred, output_dict= True)
# file_name = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/client_singoli_csv/algoritmo_alpha_0,45.xlsx"
# data = pd.DataFrame.from_dict(cl_rp)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     data.to_excel(writer, sheet_name = f"federato_prima")


# log(
#         INFO,
#         "Calcolo ece",
#     )

# test_y = test_y.compute() #questo serve !!!!
# # calcolo ece e ece per classe prima della calibrazione
# federated_proba = model_federated.predict_proba(test_Dmatrix.get_data())
# federated_ece = expected_calibration_error(federated_proba, test_y)
# federated_ece_per_class = expected_calibration_error_class(federated_proba, test_y)

# # print(f'Valore ece: {federated_ece}')
# # print(f'valore ece per classe: {federated_ece_per_class}')

# f = open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\algo_alpha_0,3.txt", "a")
# f.write(f'modello federato: ')
# f.write(f'Valore ece: {federated_ece}')
# f.write(f'valore ece per classe: {federated_ece_per_class}')
# f.close()
# # log(
# #         INFO,
# #         "Calibrazione",
# #     )



# cl_federated = CalibratedClassifierCV(model_federated, cv="prefit")
# cl_federated.fit(valid_Dmatrix.get_data(), valid_Dmatrix.get_label())

# y_pred = cl_federated.predict(test_Dmatrix.get_data())
# cl_rp = classification_report(test_y, y_pred, output_dict= True)
# file_name = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/client_singoli_csv/algoritmo_alpha_0,45.xlsx"
# data = pd.DataFrame.from_dict(cl_rp)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     data.to_excel(writer, sheet_name = f"federato_dopo")

# cl_federated = joblib.load(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/algoritmo_alpha_0,3/model_federato_calibrato.pkl')
# federated_proba = cl_federated.predict_proba(test_Dmatrix.get_data())
# federated_ece = expected_calibration_error(federated_proba, test_y)
# federated_ece_per_class = expected_calibration_error_class(federated_proba, test_y)

# f = open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\algo_alpha_0,3.txt", "a")
# f.write(f'federato calibrato: ')
# f.write(f'Valore ece: {federated_ece}')
# f.write(f'valore ece per classe: {federated_ece_per_class}')
# f.close()

# joblib.dump(cl_federated, 'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/algoritmo_alpha_0,45/model_federato_calibrato.pkl')



# ''' calibrazione su singoli client '''
# model_federated = xgb.XGBClassifier()
# model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/algoritmo_alpha_0,45/model_federato.json')
# file_name = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/client_singoli_csv/algoritmo_alpha_0,45.xlsx"

# for i in tqdm(range(10)):
#     cl_federated = CalibratedClassifierCV(model_federated, cv="prefit")
#     file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algoritmo_alpha_0,45/valid_matrix{i}.bin"
#     valid_matrix = xgb.DMatrix(file_path_valid)
#     valid_x = valid_matrix.get_data()
#     valid_y = valid_matrix.get_label()
#     cl_federated.fit(valid_x, valid_y)

#     test_y = test_Dmatrix.get_label()
#     joblib.dump(cl_federated, f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/algoritmo_alpha_0,45/calibrated_model{i}.pkl')
#     federated_proba_cl = cl_federated.predict_proba(test_Dmatrix.get_data())
#     federated_ece_cl = expected_calibration_error(federated_proba_cl, test_y)
#     federated_ece_per_class_cl = expected_calibration_error_class(federated_proba_cl, test_y)

#     f = open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\algo_alpha_0,45.txt", "a")
#     f.write(f'client{i}:\n')
#     f.write(f'Valore ece: {federated_ece_cl}\n')
#     f.write(f'valore ece per classe: {federated_ece_per_class_cl}\n')
#     f.close()
    
# #### valutazione modello 
#     y_pred = cl_federated.predict(test_Dmatrix.get_data())
#     cl_rp1 = classification_report(test_y, y_pred, output_dict= True)
#     data = pd.DataFrame.from_dict(cl_rp1)
#     with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#       data.to_excel(writer, sheet_name = f"client{i}_cl")


''' valutazione su dataset locale '''
# file_name = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/client_singoli_csv/valutazione_caso_sbilanciato.xlsx"
# model_federated = xgb.XGBClassifier()
# model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/model_sbilanciato_train_valid_test.json')
# for i in tqdm(range(10)):
#     file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_sbilanciato/test_matrix{i}.bin"
#     test_matrix = xgb.DMatrix(file_path_valid)
#     test_x = test_matrix.get_data()
#     test_y = test_matrix.get_label()
#     y_pred = model_federated.predict(test_x)
#     cl_rp1 = classification_report(test_y, y_pred, output_dict= True)
#     # print(cl_rp)
#     data = pd.DataFrame.from_dict(cl_rp1)
#     with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#         data.to_excel(writer, sheet_name = f"client{i}_local_fed")

#     federated_proba = model_federated.predict_proba(test_x)
#     federated_ece = expected_calibration_error(federated_proba, test_y)
#     federated_ece_per_class = expected_calibration_error_class(federated_proba, test_y)
#     print(f'Client{i}: ')
#     print(f'Valore ece: {federated_ece}')
#     print(f'valore ece per classe: {federated_ece_per_class}')
#     print("--------------------------------------------------------")

#     cl_federated = joblib.load(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/client_sbilanciati_calibrati/calibrated_model{i}.pkl')
#     print(f"client{i}:")
#     federated_proba_cl = cl_federated.predict_proba(test_x)
#     federated_ece_cl = expected_calibration_error(federated_proba_cl, test_y)
#     federated_ece_per_class_cl = expected_calibration_error_class(federated_proba_cl, test_y)

#     print(f'Valore ece: {federated_ece_cl}')
#     print(f'valore ece per classe: {federated_ece_per_class_cl}')
#     print('***********************************************************')

#     y_pred = cl_federated.predict(test_x)
#     cl_rp1 = classification_report(test_y, y_pred, output_dict= True)
#     data = pd.DataFrame.from_dict(cl_rp1)
#     with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#         data.to_excel(writer, sheet_name = f"client{i}_local_calib")

#file_name = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/client_singoli_csv/valutazione_caso_sbilanciato.xlsx"
# # file = open("C:/Users/lukyl/OneDrive/Desktop/Flower/ece_sbilanciato.txt", "r")
# # content=file.readlines()
# # file.close()
# import pandas as pd 
# d = pd.read_csv("C:/Users/lukyl/OneDrive/Desktop/Flower/ece_sbilanciato.txt", delimiter=" ", header = None).to_dict()[0]

# data = pd.DataFrame.from_dict(d)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     data.to_excel(writer, sheet_name = f"ece1")

''' scrittura valori ece del file client.txt in un file excel (ece1) '''

# import pandas as pd

# #Read data from the text file
# with open('C:/Users/lukyl/OneDrive/Desktop/Flower/client.txt', 'r') as file:
#     lines = file.readlines()

# import ast

# # Process the data
# data = []
# current_client = None
# for line in lines:
#     line = line.strip()
#     if line.startswith('client'):
#         # If a new client is found, start a new dictionary
#         if current_client:
#             # Append the current dictionary to the list
#             data.append(current_data)
#         current_client = line
#         current_data = {'Client': current_client, 'Valore ece': None}
#     elif line.startswith('Valore ece'):
#         value_str = line.split(':')[1].strip()
#         value = float(ast.literal_eval(value_str)[0])  # Remove square brackets and convert to float
#         current_data['Valore ece'] = value
#     elif line.startswith('valore ece per classe'):
#         value_str = line.split(':')[1].strip()
#         values = ast.literal_eval(value_str)  # Remove square brackets
#         for i, val in enumerate(values, start=1):
#             current_data[f'Valore ece per classe {i}'] = float(val)

# # Append the last client's data
# if current_client:
#     data.append(current_data)

# # Create DataFrame
# df = pd.DataFrame(data)

# # Write DataFrame to Excel
# df.to_excel('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/client_singoli_csv/ece.xlsx', index=False)


''' modello centralizzato '''

# file_name = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/client_singoli_csv/7_classi_Sbilanciato.xlsx"
# params = {
#     "objective": "multi:softprob", #"multi:softmax", #oppure "multi:softprob"
#     "eta": 0.1,  # lr
#     "max_depth": 8,
#     "eval_metric": "mlogloss",
#     "nthread": 16,
#     "num_class" : 7,
#     "num_parallel_tree": 10,
#     "subsample": 1,
#     "tree_method": "hist",
# }
# num_local_round = 1
# num_round = 5
# # fase di training; primo round con train, poi uso update
# bst = xgb.train(
#             params,
#             train_DMatrix,
#             num_boost_round=num_local_round,
#         )
# for round in range (num_round - 1):
#     bst.update(train_DMatrix, round + 1)

# # salvataggio modello
# bst.save_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/7_classi/Sbilanciato/Global_model.json')

# # carico modello come XGBClassifier per fare calibrazione
# model = xgb.XGBClassifier()
# model.load_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/7_classi/Sbilanciato/Global_model.json')


# #valutazione modello
# pred = model.predict(test_Dmatrix.get_data())
# test_y = test_Dmatrix.get_label()
# # r = []
# # for j in pred:
# #     r.append(j.argmax())

# cl_rp = classification_report(test_y, pred, output_dict= True)

# #scrittura su excel
# data = pd.DataFrame.from_dict(cl_rp)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     data.to_excel(writer, sheet_name = f"global_model")
    

# # calcolo ece
# proba = model.predict_proba(test_Dmatrix.get_data())
# ece = expected_calibration_error(proba, test_Dmatrix.get_label())
# ece_per_class = expected_calibration_error_class(proba, test_Dmatrix.get_label())

# print(f'Valore ece modello_Globale: {ece}')
# print(f'valore ece per classe modello_Globale: {ece_per_class}')
# print("---------------------------------------------------------------")

# # calibrazione

# cl_model = CalibratedClassifierCV(model, cv="prefit")
# cl_model.fit(valid_Dmatrix.get_data(), valid_Dmatrix.get_label())

# # ricalcolo ece
# proba_cl = cl_model.predict_proba(test_Dmatrix.get_data())
# ece_cl = expected_calibration_error(proba_cl, test_Dmatrix.get_label())
# ece_per_class_cl = expected_calibration_error_class(proba_cl, test_Dmatrix.get_label())

# print(f'Valore ece modello_Globale_calibrato: {ece_cl}')
# print(f'valore ece per classe modello_Globale_calibrato: {ece_per_class_cl}')
# print("---------------------------------------------------------------")

# y_pred = cl_model.predict(test_Dmatrix.get_data())
# cl_rp1 = classification_report(test_Dmatrix.get_label(), y_pred, output_dict= True)
# data = pd.DataFrame.from_dict(cl_rp1)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     data.to_excel(writer, sheet_name = f"model_calib")

# # salvo modello calibrato
# joblib.dump(cl_model, f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/7_classi/Sbilanciato/Global_model_calibrato.pkl')



''' controllo valori probabilità '''
# controllo valori di probabilità
# with open('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/selected_rows.pkl', 'rb') as f:
#     l = pickle.load(f)
# num_rows = [7, 7, 8, 9, 7, 9, 6, 9, 10, 10]


# model_federated = xgb.XGBClassifier()
# model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/model_bilanciato_train_valid_test.json')

# calibrator = joblib.load('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/client_calibrati/calibrated_model0.pkl')

# print(l[8])

# p = model_federated.predict_proba(l[8])
# p1 = calibrator.predict_proba(l[8])

# print(f"probabilità modello federato: {p}")
# print(f"probabilità modello federato calibrato: {p1}")

# parametri di slope e intercept
# slope = np.zeros(10)
# intercept = np.zeros(10)

# for j in range(10):
#     calibrator = joblib.load(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/client_calibrati/calibrated_model{j}.pkl')
#     c = calibrator.calibrated_classifiers_
#     for clf in c:
#         for i in range(10):
#             slope[i] += clf.calibrators[i].a_
#             intercept[i] += clf.calibrators[i].b_
# slope = slope/10
# intercept = intercept/10


            


# estimator_params = params["estimator"]
# print(estimator_params)
# controllo parametri calibratore
# i = 2
# cl_federated = joblib.load(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/client_calibrati/calibrated_model{i}.pkl')
# param = cl_federated.get_params()
# print(param)

# model_federated = xgb.XGBClassifier()
# model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/model_bilanciato_train_valid_test.json')

# # calcolo prestazioni su dataset più grande
# y_pred = model_federated.predict(test_x)
# cl_rp = classification_report(test_y, y_pred, output_dict= True)

# model = joblib.load(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/calibrated_model_federated.pkl')

# y_pred = model.predict(test_x)
# cl_rp1 = classification_report(test_y, y_pred, output_dict= True)
# data = pd.DataFrame.from_dict(cl_rp)
# data1 = pd.DataFrame.from_dict(cl_rp1)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     data.to_excel(writer, sheet_name = f"model_fed_")
#     data1.to_excel(writer, sheet_name = f"model_fed_calibrato")


''' rete neurale con sigmoide come act_function'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xgboost as xgb

# Define your neural network architecture
class Calibrator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Calibrator, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.fc.weight.data = torch.eye(input_size)  # Initialize with identity matrix
        self.fc.weight.requires_grad = False
        self.a = nn.Parameter(torch.ones(output_size))  # Initialize a as ones
        self.b = nn.Parameter(torch.zeros(output_size))  # Initialize b as zeros
        # self.a = torch.nn.init.xavier_normal_(self.a, gain=1.0, generator=None) #-> xavier solo per vettori multidimensionali
        nn.init.normal_(self.a, mean=0.0, std=0.1)  # Inizializza a con distribuzione normale
        nn.init.normal_(self.b, mean=0.0, std=0.1)

    def forward(self, x):
        #x = self.fc(x)#x*a +b sigmoide(x)
        #x = 1 / (1 + torch.exp(-(self.a * x + self.b)))
        x = self.a*x +self.b 
        x = torch.sigmoid(x)
        sum_x = torch.sum(x, dim=1, keepdim=True)  # Calcola la somma degli elementi del vettore
        x = x / sum_x
        
        return x

file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/VALID_MATRIX.bin"
valid_server = xgb.DMatrix(file_path_valid)

x_valid = valid_server.get_data()
y_valid = valid_server.get_label()

model_federated = xgb.XGBClassifier()
model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/model_federato.json')

# Predict probabilities
pred = model_federated.predict_proba(x_valid)


# If y_valid is 1D, you need to convert it to a 2D one-hot encoded matrix
num_classes = pred.shape[1]
y_valid_one_hot = np.eye(num_classes)[y_valid.astype(int)]

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(pred, dtype=torch.float32)
y_train = torch.tensor(y_valid_one_hot, dtype=torch.float32)

# Define hyperparameters
input_size = X_train.size(1)
output_size = y_train.size(1)
learning_rate = 0.5
num_epochs = 200

# Initialize model, loss function, and optimizer
model = Calibrator(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if(epoch % 20 == 0):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# test
model_calibrato = CalibratedClassifierCV(model_federated, cv="prefit")
model_calibrato.fit(valid_server.get_data(), valid_server.get_label())

with open('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/selected_rows.pkl', 'rb') as f:
    l = pickle.load(f)

# volgio le pred del modello federato come tensor per andare in input al mio modello
p = model_federated.predict_proba(l[3])
print(f"probabilità modello prima della calibrazione: {p}")
p_tensor = torch.tensor(p, dtype=torch.float32)
model.eval()
with torch.no_grad():
    #mio_pred = model(p_tensor).numpy()
    mio_pred = model(p_tensor).numpy()


p1 = model_calibrato.predict_proba(l[3])

print(f"probabilità modello calibrato sklearn: {p1}")
print(f"probabilità mio modello calibrato : {mio_pred}")

print("--------------------------------------------------------------------")
print("Pesi a dopo addestramento:", model.a.data)
print("Pesi b dopo addestramento:", model.b.data)

calib_a = []
calib_b = []
c = model_calibrato.calibrated_classifiers_
for clf in c:
    for i in range(10):
        calib_a.append( clf.calibrators[i].a_)
        calib_b.append(clf.calibrators[i].b_)

print("valore a sklearn: ", calib_a)
print("valore b sklearn: ", calib_b)


''' test algo alpha '''

def __getDirichletData__(y, n, alpha, num_c):

        min_size = 0
        N = len(y)
        net_dataidx_map = {}
        p_client = np.zeros((n,num_c))

        for i in tqdm(range(n)):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,num_c))
        idx_batch = [[] for _ in range(n)]

        for k in tqdm(range(num_c)):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = p_client[:,k]
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        
        for j in tqdm(range(n)):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        
        net_cls_counts = {}
        
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
            # values = [y[idx] for idx in dataidx]
            # unq, unq_cnt = np.unique(values, return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in tqdm(range(n)):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

        return idx_batch, net_cls_counts

# import pandas as pd

# df = pd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file4.csv')
# Y = df['Attack_label']
# # Y = Y.compute()
# print("chiamata funzione: ")
# idx_batch, net_cls_counts = __getDirichletData__(Y, 10, 0.45, 10)

# while True:
#     user_input = input("Do you want to continue? (Y/N): ")
#     if user_input.upper() == 'Y':
#         for i in tqdm(range(10)):
#             client_indices = idx_batch[i]
#             client_data = df.iloc[client_indices] 
#             client_x = client_data.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Attack_label'])
#             client_y = client_data['Attack_label']
            
#             client_train_x, client_test_x, client_train_y, client_test_y = train_test_split(client_x, client_y, test_size=0.2, shuffle = True)# divido in test e train per ogni client
#             client_train_x, client_valid_x, client_train_y, client_valid_y = train_test_split(client_train_x, client_train_y, test_size=0.1, shuffle = True)
#             train_matrix = xgb.DMatrix(client_train_x, label=client_train_y)
#             test_matrix = xgb.DMatrix(client_test_x, label=client_test_y)
#             valid_matrix = xgb.DMatrix(client_valid_x, client_valid_y)

#             file_path_train = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algoritmo_alpha_0,45/train_matrix{i}.bin"
#             file_path_test = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algoritmo_alpha_0,45/test_matrix{i}.bin"
#             file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algoritmo_alpha_0,45/valid_matrix{i}.bin"
#             train_matrix.save_binary(file_path_train)
#             test_matrix.save_binary(file_path_test)
#             valid_matrix.save_binary(file_path_valid)
#     elif user_input.upper() == 'N':
#         print("Program interrupted.")
#         break  


''' stampa istogrammi '''
# # Dati

# for j in range (10):
#     file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algoritmo_alpha_0,1/valid_matrix{j}.bin"
#     valid_matrix = xgb.DMatrix(file_path_valid)
#     y = valid_matrix.get_label()
#     num_rows = len(y)

#     counts = np.zeros(10, dtype=int)

#     unique, class_counts = np.unique(y, return_counts=True)
#     counts[unique.astype(int)] = class_counts

#     v = np.zeros(10)
#     i = 0
#     for item in unique:
#         v[i] = counts[i] / num_rows
#         i = i + 1

#     categories = ['scanning', 'Benign', 'password', 'ddos', 'xss', 'dos', 'injection', 'mitm', 'ransomware', 'backdoor']

#     # Disegna l'istogramma
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(categories, v, color='skyblue')
#     plt.xlabel('Categoria')
#     plt.ylabel('Frequenza')
#     plt.title(f'Distribuzione delle categorie client{j}')
#     plt.xticks(rotation=45, ha='right')

#     # Adding text labels for sample counts
#     for bar, count in zip(bars, counts):
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

#     plt.tight_layout()
#     plt.show()



''' altro test '''
# # create a list of clients with local datasets
# clients = [...]

# # Define the FedAvg algorithm
# def federated_averaging(clients):
#     # Initialize a global model
#     global_model = SigmoidModel()

#     # Perform FedAvg
#     for client in clients:
#         # Create a copy of the global model for the client
#         client_model = copy.deepcopy(global_model)

#         # Train the client model locally
#         client_optimizer = optim.SGD(client_model.parameters(), lr=0.1)
#         criterion = nn.MSELoss()
#         for _ in range(num_epochs):
#             for data in client.dataset:
#                 inputs, labels = data
#                 client_optimizer.zero_grad()
#                 outputs = client_model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 client_optimizer.step()

#         # Aggregate the client model updates using FedAvg
#         with torch.no_grad():
#             for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
#                 global_param.data.add_(client_param.data)

#     # Average the global model parameters
#     num_clients = len(clients)
#     with torch.no_grad():
#         for global_param in global_model.parameters():
#             global_param.data.div_(num_clients)

#     return global_model

# # Perform Federated Averaging
# global_model = federated_averaging(clients)

''' lettura valori file txt'''
# import re
# import numpy as np
# text = '''
# Server: Metriche: {'acc': 0.9111803061252342, 'f1-weighted': 0.8930974433536318, 'f1-macro': 0.6890261376717441}Valore ece : [0.05732183]
# Media clients: Metriche: {'acc': 0.8886773203382441, 'f1-weighted': 0.8641698263481947, 'f1-macro': 0.6621244720233167}Valore ece : [0.07051924]
# Server: Metriche: {'acc': 0.9219615975048484, 'f1-weighted': 0.9209905068289137, 'f1-macro': 0.8428496065705297}Valore ece : [0.04764243]
# Media clients: Metriche: {'acc': 0.9134845443567334, 'f1-weighted': 0.9134966579120493, 'f1-macro': 0.8026004095597171}Valore ece : [0.066864]
# Server: Metriche: {'acc': 0.9499693491168784, 'f1-weighted': 0.9496948157043659, 'f1-macro': 0.8653483867595961}Valore ece : [0.02743873]
# Media clients: Metriche: {'acc': 0.9347727814256386, 'f1-weighted': 0.934497691937983, 'f1-macro': 0.8488086499853109}Valore ece : [0.03287136]
# Server: Metriche: {'acc': 0.9578397287083327, 'f1-weighted': 0.9571804925773413, 'f1-macro': 0.8592002903312762}Valore ece : [0.01615427]
# Media clients: Metriche: {'acc': 0.951954390305616, 'f1-weighted': 0.9515427381025727, 'f1-macro': 0.8426183902534076}Valore ece : [0.03024404]
# Server: Metriche: {'acc': 0.9309466217267918, 'f1-weighted': 0.9302089643027284, 'f1-macro': 0.720448237757527}Valore ece : [0.04861594]
# Media clients: Metriche: {'acc': 0.927838452510311, 'f1-weighted': 0.9265752437031125, 'f1-macro': 0.715921961792889}Valore ece : [0.05069331]

# '''
# def parse_metrics(text):
#     pattern = re.compile(r"Metriche: \{'acc': ([0-9.]+), 'f1-weighted': ([0-9.]+), 'f1-macro': ([0-9.]+)\}Valore ece : \[([0-9.]+)\]")
#     matches = pattern.findall(text)

#     server_metrics = {'acc': [], 'f1-weighted': [], 'f1-macro': [], 'ece': []}
#     clients_metrics = {'acc': [], 'f1-weighted': [], 'f1-macro': [], 'ece': []}

#     for i, match in enumerate(matches):
#         acc, f1_weighted, f1_macro, ece = map(float, match)
#         if i % 2 == 0:
#             server_metrics['acc'].append(acc)
#             server_metrics['f1-weighted'].append(f1_weighted)
#             server_metrics['f1-macro'].append(f1_macro)
#             server_metrics['ece'].append(ece)
#         else:
#             clients_metrics['acc'].append(acc)
#             clients_metrics['f1-weighted'].append(f1_weighted)
#             clients_metrics['f1-macro'].append(f1_macro)
#             clients_metrics['ece'].append(ece)

#     return server_metrics, clients_metrics

# # Calculate the average values for each metric
# def calculate_averages(metrics):
#     averages = {key: np.mean(values) for key, values in metrics.items()}
#     return averages

# # Parse the metrics from the text
# server_metrics, clients_metrics = parse_metrics(text)

# # Calculate the averages for server and clients
# server_averages = calculate_averages(server_metrics)
# clients_averages = calculate_averages(clients_metrics)

# print("Server Metrics Averages:")
# print(server_averages)
# print("\nClients Metrics Averages:")
# print(clients_averages)


''' grafici metrics vs alpha '''

#import matplotlib.pyplot as plt

# # Data
# alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
# acc = [0.808106292, 0.915242894, 0.913591509, 0.943877975, 0.934379521]
# f1_weighted = [0.770874268, 0.906474178, 0.909365261, 0.941702471, 0.930234445]
# f1_macro = [0.483301538, 0.746606442, 0.780242994, 0.818408966, 0.795374532]
# ece = [0.109668098, 0.055761542, 0.05137587, 0.027617976, 0.03943464]

# # client
# acc_client = [0.511873894, 0.834379403, 0.862238864, 0.936322971, 0.923345498]
# f1_weighted_client = [0.43647292, 0.805950109, 0.846844557, 0.933984065, 0.918056432]
# f1_macro_client = [0.251841882, 0.62535678, 0.695802856, 0.793161121, 0.774414777]
# ece_client = [0.343704012, 0.101796826, 0.088268944, 0.044610672, 0.05023839]

# #centralizzato riferimento
# acc_centr = [0.963782533,0.963782533,0.963782533,0.963782533,0.963782533]
# f1_weighted_centr = [0.962013839,0.962013839,0.962013839,0.962013839,0.962013839]
# f1_macro_centr = [0.88955765,0.88955765,0.88955765,0.88955765,0.88955765]
# ece_centr = [ 0.00878548,0.00878548,0.00878548,0.00878548,0.00878548]
# # Data for the second table

# # Plotting
# plt.figure(figsize=(10, 6))

# plt.plot(alpha, acc, marker='o', label='acc_server', color='blue')
# plt.plot(alpha, acc_client, marker='o', label='acc_client', color='green')
# plt.plot(alpha, acc_centr, marker='o', label='acc_centr', color='red')

# plt.xlabel('Alpha')
# plt.ylabel('acc')
# plt.title('acc vs Alpha')
# plt.xticks(alpha)
# #plt.grid(True)
# plt.legend()

# # plt.tight_layout()
# # plt.show()