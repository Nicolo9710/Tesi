import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from logging import ERROR, INFO, WARN
from flwr.common.logger import log
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import copy

import tensorflow as tf


import threading
import flwr as fl
from tutorial import client


#creazione dataset NF_ToN_IoT_V2
# df = dd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file4.csv')
# # print(df['Attack_label'].value_counts().compute())
# #df = df[(df['Attack_label'] < 7)]

# X = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Attack_label'])
# Y = df['Attack_label']
#X_drop, X_keep, y_drop, y_keep = train_test_split(X, Y, test_size=0.2, shuffle = True, random_state=42) # 20% del dataframe totale

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

        # print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

        return idx_batch, net_cls_counts

log(
        INFO,
        "Preparazione dataSet",
    )

n_partitions = 10 # deve essere uguale al numero di client che voglio usare
partitions = []

df = pd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file4.csv')
Y = df['Attack_label']
# Y = Y.compute()
print("chiamata funzione: ")
idx_batch, net_cls_counts = __getDirichletData__(Y, 10, 0.9, 10)

log(
        INFO,
        "Creazione Dmatrix",
    )

for i in tqdm(range(10)):
    client_indices = idx_batch[i]
    client_data = df.iloc[client_indices] 
    client_x = client_data.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Attack_label'])
    client_y = client_data['Attack_label']
    
    client_train_x, client_test_x, client_train_y, client_test_y = train_test_split(client_x, client_y, test_size=0.2, shuffle = True)# divido in test e train per ogni client
    client_train_x, client_valid_x, client_train_y, client_valid_y = train_test_split(client_train_x, client_train_y, test_size=0.1, shuffle = True)
    train_matrix = xgb.DMatrix(client_train_x, label=client_train_y)
    test_matrix = xgb.DMatrix(client_test_x, label=client_test_y)
    valid_matrix = xgb.DMatrix(client_valid_x, client_valid_y)
    train_num = train_matrix.num_row()
    test_num = test_matrix.num_row()
    partitions.append((train_matrix, valid_matrix, test_matrix, train_num, test_num))




# for i in tqdm(range(n_partitions)):
#     # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle = True)# assegno 10% per ogni client 
#     # client_train_x, client_test_x, client_train_y, client_test_y = train_test_split(x_test, y_test, test_size=0.2, shuffle = True)# divido in test e train per ogni client
#     # client_train_x, client_valid_x, client_train_y, client_valid_y = train_test_split(client_train_x, client_train_y, test_size=0.1, shuffle = True)
#     # train_matrix = xgb.DMatrix(client_train_x, label=client_train_y)
#     # test_matrix = xgb.DMatrix(client_test_x, label=client_test_y)
#     # valid_matrix = xgb.DMatrix(client_valid_x, client_valid_y)
    
    
#     file_path_train = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_bilanciato_train_test_valid/train_matrix{i}.bin"
#     file_path_test = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_bilanciato_train_test_valid/test_matrix{i}.bin"
#     file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_bilanciato_train_test_valid/valid_matrix{i}.bin"
#     # train_matrix.save_binary(file_path_train)
#     # test_matrix.save_binary(file_path_test)
#     # valid_matrix.save_binary(file_path_valid)

#     train_matrix = xgb.DMatrix(file_path_train)
#     test_matrix = xgb.DMatrix(file_path_test)
#     valid_matrix = xgb.DMatrix(file_path_valid)

#     train_num = train_matrix.num_row()
#     test_num = test_matrix.num_row()
#     partitions.append((train_matrix, valid_matrix, test_matrix, train_num, test_num))

# creazione server -> il server lo faccio partire io da un altro terminale. (python server.py)
# va fatto prima di chiamare main.py, altrimenti i client non trovano il server e si blocca tutto

log(
        INFO,
        "Creazione e avvio client",
    )
#creazione clients, utilizzo threading così da poter usare più client in parallelo, altrimenti non funziona l'addestramento

num_clients = 10

def start_clients(num_clients, partitions):
    # Define a function to start a client
    def start_client(partition, node_id):
        fl.client.start_client(server_address="127.0.0.1:8079", client=client.XgbClient(partition, node_id))

    # Start clients in separate threads
    threads = []
    for i in range(num_clients):
        thread = threading.Thread(target=start_client, args=(partitions[i], i))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Call the function to start clients
start_clients(num_clients, partitions)

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

class Calibrator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Calibrator, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.fc.weight.data = torch.eye(input_size)  # Initialize with identity matrix
        self.fc.weight.requires_grad = False
        self.a = nn.Parameter(torch.ones(input_size))  # Initialize a as ones
        self.b = nn.Parameter(torch.ones(input_size))  # Initialize b as zeros
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

    def get_parameters(self):
        tensor_str_a = ' '.join([str(self.a[i].item()) for i in range(self.a.numel())])
        tensor_str_b = ' '.join([str(self.b[i].item()) for i in range(self.b.numel())])
        combined_str = f"{tensor_str_a};{tensor_str_b}"
        bytes_obj = combined_str.encode('utf-8')
        # a = tf.io.serialize_tensor(self.a)
        return bytes_obj
    
    def set_parameters(self, a_params, b_params):
        with torch.no_grad():
            self.a.copy_(torch.tensor(a_params))
            self.b.copy_(torch.tensor(b_params))


from sklearn.base import BaseEstimator, ClassifierMixin

class MyCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, input_size, output_size, lr=0.02, epochs=200):
        self.base_model = base_model
        self.calibrator = Calibrator(input_size, output_size)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.calibrator.to(self.device)
        self.is_fitted_ = False
        self.classes_ = None

    def fit(self, X, y):
        # Get the predicted probabilities from the base model
        base_preds = self.base_model.predict_proba(X)
        X_calib, y_calib = torch.FloatTensor(base_preds).to(self.device), torch.LongTensor(y).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.calibrator.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.calibrator.train()
            optimizer.zero_grad()
            outputs = self.calibrator(X_calib)
            loss = criterion(outputs, y_calib)
            loss.backward()
            optimizer.step()
        
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self


    def predict_proba(self, X):
        self.calibrator.eval()
        with torch.no_grad():
            # Get the predicted probabilities from the base model
            base_preds = self.base_model.predict_proba(X)
            X_calib = torch.FloatTensor(base_preds).to(self.device)
            outputs = self.calibrator(X_calib)
        return outputs.cpu().numpy()    
    
    def get_parameters(self):
        return self.calibrator.get_parameters()
    
    def set_parameters(self,a_params, b_params)  -> None:
         self.calibrator.set_parameters(a_params, b_params)



from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

# server
file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/VALID_MATRIX.bin"
valid_server = xgb.DMatrix(file_path_valid)

model_federated = xgb.XGBClassifier()
model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/model_federato.json')

file_path_test = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/TEST_MATRIX.bin"
test_server= xgb.DMatrix(file_path_test)

cl_federated = CalibratedClassifierCV(model_federated, cv="prefit")
cl_federated.fit(valid_server.get_data(), valid_server.get_label())

# mio modello
my_model = MyCalibrator(model_federated, input_size=10, output_size=10)
my_model.calibrator.load_state_dict(torch.load('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/federated_calib_model.pth'))
my_model.calibrator.to(my_model.device)

param = my_model.get_parameters()
str_list = param.decode('utf-8').split(';')
# Separazione delle stringhe nei due tensori
a = [float(item) for item in str_list[0].split()]
b = [float(item) for item in str_list[1].split()]
print("parametri a: ", a)
print("parametri b: ", b)

test_x = test_server.get_data()
test_y = test_server.get_label()

print("valutazione server: ")

y_pred = cl_federated.predict(test_x)
cl_rp = classification_report(test_y, y_pred, output_dict= True)

metrics={"acc": cl_rp["accuracy"], "f1-weighted": cl_rp["weighted avg"]["f1-score"], 
                                "f1-macro": cl_rp["macro avg"]["f1-score"]}

federated_proba = cl_federated.predict_proba(test_x)
federated_ece = expected_calibration_error(federated_proba, test_y)

f = open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\client.txt", "a")
f.write(f'Server: ')
f.write(f'Metriche: {metrics}')
f.write(f'Valore ece : {federated_ece}')
f.close()


# clients
acc = 0
f1_weight = 0
f1_mac = 0
ece_avg = 0
for i in tqdm(range(10)):

    cl_federated = CalibratedClassifierCV(model_federated, cv="prefit")
    train_client,valid_client,test_client,num1, num2 = partitions[i]
    cl_federated.fit(valid_client.get_data(), valid_client.get_label())
    y_pred = cl_federated.predict(test_x)
    cl_rp = classification_report(test_y, y_pred, output_dict= True)
    acc += cl_rp["accuracy"]
    f1_weight += cl_rp["weighted avg"]["f1-score"]
    f1_mac +=  cl_rp["macro avg"]["f1-score"]

    federated_proba = cl_federated.predict_proba(test_x)
    ece_avg += expected_calibration_error(federated_proba, test_y)

metrics_client={"acc": acc/10, "f1-weighted": f1_weight/10, "f1-macro": f1_mac/10}
ece_avg = ece_avg/10

f = open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\client.txt", "a")
f.write(f'Media clients: ')
f.write(f'Metriche: {metrics_client}')
f.write(f'Valore ece : {ece_avg}')
f.close()

# my_model
print("valutazione My_model")
y_pred_proba = my_model.predict_proba(test_x)
y_pred = np.argmax(y_pred_proba, axis=1)
cl_rp = classification_report(test_y, y_pred, output_dict= True)

metrics={"acc": cl_rp["accuracy"], "f1-weighted": cl_rp["weighted avg"]["f1-score"], 
                                "f1-macro": cl_rp["macro avg"]["f1-score"]}

federated_ece = expected_calibration_error(y_pred_proba, test_y)

f = open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\client.txt", "a")
f.write(f'My_model: ')
f.write(f'Metriche: {metrics}')
f.write(f'Valore ece : {federated_ece}')
f.close()


