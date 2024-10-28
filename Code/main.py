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
from torch.optim import lr_scheduler
import copy

import tensorflow as tf


import threading
import flwr as fl
from tutorial import client

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

# # creazione dataset NF_ToN_IoT_V2
# # df = dd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file4.csv')
# # # print(df['Attack_label'].value_counts().compute())
# # #df = df[(df['Attack_label'] < 7)]

# # X = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack', 'Attack_label'])
# # Y = df['Attack_label']
# # X_drop, X_keep, y_drop, y_keep = train_test_split(X, Y, test_size=0.2, shuffle = True, random_state=42) # 20% del dataframe totale
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
        with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Federato.txt", "a") as f:
            f.write(f"data_ratio: {str(weights)}\n")
            
        return idx_batch, net_cls_counts, weights

# log(
#         INFO,
#         "Preparazione dataSet",
#     )

# n_partitions = 10 # deve essere uguale al numero di client che voglio usare
# partitions = []


# train_x = pd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/train_x.csv')
# train_y = pd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/train_y.csv')

# train_y_df = pd.DataFrame(train_y, columns=['Attack_label'])

# # Estrai la colonna 'Attack_label' come serie
# train_y_series = train_y_df['Attack_label']


# print("chiamata funzione: ")
# idx_batch, net_cls_counts, data_ratio = __getDirichletData__(train_y_series.values, n_partitions, 0.8, 10)

# log(
#         INFO,
#         "Creazione Dmatrix",
#     )

# server_valid_matrix_x = []
# server_valid_matrix_y = []


# for i in tqdm(range(n_partitions)):
#     client_indices = idx_batch[i]
#     client_x_df= train_x.iloc[client_indices]
#     client_y_df = train_y_df.iloc[client_indices]
#     client_x = client_x_df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack'])
#     client_y = client_y_df['Attack_label']

    
#     client_train_x, client_test_x, client_train_y, client_test_y = train_test_split(client_x, client_y, test_size=0.2, shuffle = True)# divido in test e train per ogni client
#     client_train_x, client_valid_x, client_train_y, client_valid_y = train_test_split(client_train_x, client_train_y, test_size=0.1, shuffle = True)
#     train_matrix = xgb.DMatrix(client_train_x, label=client_train_y)
#     test_matrix = xgb.DMatrix(client_test_x, label=client_test_y)
#     valid_matrix = xgb.DMatrix(client_valid_x, label=client_valid_y)
#     train_num = train_matrix.num_row()
#     test_num = test_matrix.num_row()
#     file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/valid 08/4/valid_matrix{i}.bin"
#     valid_matrix.save_binary(file_path_valid)

#     partitions.append((train_matrix, valid_matrix, test_matrix, train_num, test_num, data_ratio[i]))
#     server_valid_matrix_x.append(client_valid_x)
#     server_valid_matrix_y.append(client_valid_y)

# combined_data = np.vstack(server_valid_matrix_x)
# combined_labels = np.concatenate(server_valid_matrix_y)
# server_valid_matrix = xgb.DMatrix(combined_data, label = combined_labels)

# # server_valid_matrix = xgb.DMatrix(server_valid_matrix_x, label = server_valid_matrix_y)
# file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/valid 08/4/VALID_MATRIX.bin"
# server_valid_matrix.save_binary(file_path_valid)




# log(
#         INFO,
#         "Creazione e avvio client",
#     )
# # creazione clients, utilizzo threading così da poter usare più client in parallelo, altrimenti non funziona l'addestramento

# num_clients = 10

# def start_clients(num_clients, partitions):
#     # Define a function to start a client
#     def start_client(partition, node_id):
#         fl.client.start_client(server_address="127.0.0.1:8079", client=client.XgbClient(partition, node_id))

#     # Start clients in separate threads
#     threads = []
#     for i in range(num_clients):
#         thread = threading.Thread(target=start_client, args=(partitions[i], i))
#         thread.start()
#         threads.append(thread)

#     # Wait for all threads to complete
#     for thread in threads:
#         thread.join()

# # Call the function to start clients
# start_clients(num_clients, partitions)

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
        x = torch.sigmoid(-x)
        sum_x = torch.sum(x, dim=1, keepdim=True)  # Calcola la somma degli elementi del vettore
        x = x / sum_x
        return x

    def get_parameters(self):
        # tensor_str_a = ' '.join([str(self.a[i].item()) for i in range(self.a.numel())])
        # tensor_str_b = ' '.join([str(self.b[i].item()) for i in range(self.b.numel())])
        # combined_str = f"{tensor_str_a};{tensor_str_b}"
        # bytes_obj = combined_str.encode('utf-8')
        # # a = tf.io.serialize_tensor(self.a)
        # return bytes_obj
        return [self.a.detach().numpy(), self.b.detach().numpy()]
    
    def set_parameters(self, a_params, b_params):
        with torch.no_grad():
            self.a.copy_(torch.tensor(a_params))
            self.b.copy_(torch.tensor(b_params))


from sklearn.base import BaseEstimator, ClassifierMixin

class MyCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, input_size, output_size, lr=0.2, epochs=2000):
        self.base_model = base_model
        self.calibrator = Calibrator(input_size, output_size)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.calibrator.to(self.device)
        self.is_fitted_ = False
        self.classes_ = None

    def fit(self, X, y, lr=None, epochs=None):
        if lr is None:
            lr = self.lr
        if epochs is None:
            epochs = self.epochs

        # Get the predicted probabilities from the base model
        base_preds = self.base_model.predict_proba(X) 
        X_calib, y_calib = torch.FloatTensor(base_preds).to(self.device), torch.LongTensor(y).to(self.device)
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(self.calibrator.parameters(), lr=self.lr)
        optimizer = torch.optim.RMSprop(self.calibrator.parameters(), lr=self.lr)
        #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # lmbda = lambda epoch: 0.95
        # scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        for epoch in range(self.epochs):
            self.calibrator.train()
            optimizer.zero_grad()
            outputs = self.calibrator(X_calib)
            loss = criterion(outputs, y_calib)
            loss.backward()
            optimizer.step()
            # scheduler.step()

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
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
       
    def get_parameters(self):
        return self.calibrator.get_parameters()
    
    def set_parameters(self,a_params, b_params)  -> None:
         self.calibrator.set_parameters(a_params, b_params)







file_path_test = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/TEST_MATRIX.bin"
test_server= xgb.DMatrix(file_path_test)

Alpha = "09"

test_x = test_server.get_data()
test_y = test_server.get_label()

######## prima della calibrazione ####################

print("\n")
print("################################################################################")
print("valutazione modello federato:")

def federato(model_federato):

    fed_pred = model_federato.predict(test_x)
    cl_rp_fed = classification_report(test_y, fed_pred, output_dict= True)
    fed_proba = model_federato.predict_proba(test_x)

    all_reports = [cl_rp_fed]
    all_probs = [fed_proba]

    average_metrics = {}
    for label in all_reports[0].keys():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = {'precision': [], 'recall': [], 'f1-score': [], 'ece_per_class': []}
            for i, report in enumerate(all_reports):
                for metric in metrics:
                    if metric == 'ece_per_class':
                        class_index = int(float(label))  # Converti la label stringa in intero
                        metrics[metric].append(expected_calibration_error_class(all_probs[i], test_y)[class_index])
                    else:
                        metrics[metric].append(report[label][metric])
            average_metrics[label] = {metric: np.mean(values) for metric, values in metrics.items()}

    Val = expected_calibration_error(fed_proba, test_y)
    Val_num = Val[0] if isinstance(Val, np.ndarray) else Val
    global_metrics = {
        "acc": cl_rp_fed["accuracy"],
        "f1-weighted": cl_rp_fed["weighted avg"]["f1-score"],
        "f1-macro": cl_rp_fed["macro avg"]["f1-score"],
        "ece": Val_num
    }

    acceptance_percentages = []
    thresholds = np.arange(0.01, 1.0, 0.01)
    for threshold in thresholds:
        # Controlla se la probabilità della classe predetta è sopra la soglia
        above_threshold = np.max(fed_proba, axis=1) >= threshold

        percentuale_dati_scartati = ( 1 - np.mean(above_threshold) ) * 100

        # Se nessuna probabilità è sopra la soglia
        if not np.any(above_threshold):
             acceptance_percentages.append((100, 0.0))

        else:
        # Filtra le classi predette sopra la soglia
            pred_label = fed_pred[above_threshold]
            
            # Controlla se la predizione corrisponde all'etichetta vera
            correct_predictions = pred_label == test_y[above_threshold]
            
            # Calcola la percentuale di accettazione
            percentage_correct = np.mean(correct_predictions) * 100
            acceptance_percentages.append((percentuale_dati_scartati, percentage_correct))

    # rimozione duplicati e ordinamento rispetto percenutali dati scartati
    acceptance_percentages = list(set(acceptance_percentages))
    acceptance_percentages.sort()

    summary_array = []

    # Creiamo un dizionario per memorizzare la percentuale di accettazione per ogni fascia
    summary_dict = {i: [] for i in range(0, 101, 5)}

    for discarded_percentage, acceptance_percentage in acceptance_percentages:
    # Trova la fascia di appartenenza
        range_key = int(discarded_percentage // 5) * 5
    
        # Append valore per fare poi la media   
        summary_dict[range_key].append(acceptance_percentage)

    # Creazione dell'array riassuntivo calcolando la media per ogni fascia
    summary_array = [(k, np.mean(summary_dict[k]) if summary_dict[k] else 0.0) for k in sorted(summary_dict.keys())]

    # print(summary_array)

    # Separiamo i range e i valori di percentuale
    ranges, values = zip(*summary_array)

    # Identifica gli indici dove il valore è 0.0 e non tutti i valori successivi sono zero
    for i in range(len(values)):
        if values[i] == 0.0:
            # Cerca il prossimo valore non zero
            j = i + 1
            while j < len(values) and values[j] == 0.0:
                j += 1
            
            # Se abbiamo trovato un valore non zero, facciamo l'interpolazione
            if j < len(values):
                # Interpolazione lineare tra i valori validi più vicini
                values = list(values)
                if i > 0 and values[i-1] != 0.0:
                    values[i] = values[i-1] + (values[j] - values[i-1]) * (ranges[i] - ranges[i-1]) / (ranges[j] - ranges[i-1])
                else:
                    values[i] = values[j]

    # # Ricostruisci il summary array con i valori interpolati
    # summary_array_interpolated = list(zip(ranges, values))

    # print(values)
    # # Stampa o ritorna l'array interpolato
    # print(summary_array_interpolated)

    return average_metrics, global_metrics, values

federato_class = []
federato_global = []
federato_acceptance = []

with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Federato.txt", "a") as f:
    f.write(f"\nALPHA = {Alpha} ")
    f.close()

for i in tqdm(range(5)):

    model_federated = xgb.XGBClassifier()
    model_federated.load_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha {Alpha}/model_federato{i}.json')
        
    fed_average_metrics, fed_global_metrics, fed_acceptance_percentages = federato(model_federated)
    federato_class.append(fed_average_metrics)
    federato_global.append(fed_global_metrics)
    federato_acceptance.append(fed_acceptance_percentages)
    with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Federato.txt", "a") as f:
        f.write("\n Federato: ")
        f.write(f'Metriche: {fed_global_metrics}')
        f.close()

# Aggregazione delle metriche medie per classe
final_average_metrics = {}
for label in federato_class[0].keys():
    final_average_metrics[label] = {metric: np.mean([run_metrics[label][metric] for run_metrics in federato_class])
                                    for metric in federato_class[0][label].keys()}

# Aggregazione delle metriche globali
final_global_metrics = {metric: np.mean([run_metrics[metric] for run_metrics in federato_global])
                        for metric in federato_global[0].keys()}

# Aggregazione delle percentuali di accettazione
final_acceptance_percentages = np.mean(federato_acceptance, axis=0)


with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Federato.txt", "a") as f:
    f.write("\n\nAverage Metrics per Class:\n")
    for label, metrics in final_average_metrics.items():
        f.write(f"Class {label}:\n")
        for metric, value in metrics.items():
            f.write(f"  Federato_avg_{metric}_cl{label} = [ {value:.4f} ]\n")
        f.write("\n")
    
    f.write("Global Metrics:\n")
    for metric, value in final_global_metrics.items():
        f.write(f"  Federato_{metric} = [ {value:.4f} ]\n")
    f.write("\n")
    
    f.write("Acceptance Percentages:\n")
    # thresholds = np.arange(0.1, 1.0, 0.1)
    # for threshold, percentage in zip(thresholds, final_acceptance_percentages):
    #     f.write(f"  Threshold {threshold:.1f}: {percentage:.2f}%\n")
    f.write('[' + ', '.join(f'{v:.5f}' for v in final_acceptance_percentages) + ']')
    f.write("\n")

#########################################################################################################################################

### SERVER

#########################################################################################################################################
print("\n")
print("################################################################################")
print("valutazione lato server:")

def ServerSub(n, valid_server, model_federated):

    data = valid_server.get_data().toarray()
    labels = valid_server.get_label()
    df = pd.DataFrame(data)
    df['label'] = labels

    num_rows_per_class = n
    value_counts = df['label'].value_counts()
    l = []
    # Campiona righe per ciascuna classe
    for label, count in value_counts.items():
        k = df[df['label'] == label]
        frac = min(num_rows_per_class / count, 1.0)
        selected_rows = k.sample(frac=frac, random_state=42)
        l.append(selected_rows)
    result_df = pd.concat(l)
    sampled_data = result_df.drop(columns=['label']).values
    sampled_labels = result_df['label'].values
    sampled_matrix = xgb.DMatrix(sampled_data, label=sampled_labels)
   

    cl_federated = MyCalibrator(model_federated, input_size=10, output_size=10)
    cl_federated.fit(sampled_matrix.get_data(), sampled_matrix.get_label())

    server_y_pred = cl_federated.predict(test_x)
    server_proba = cl_federated.predict_proba(test_x)
    server_cl_rp = classification_report(test_y, server_y_pred, output_dict= True)

    all_reports = [server_cl_rp]
    all_probs = [server_proba]

    average_metrics = {}
    for label in all_reports[0].keys():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = {'precision': [], 'recall': [], 'f1-score': [], 'ece_per_class': []}
            for i, report in enumerate(all_reports):
                for metric in metrics:
                    if metric == 'ece_per_class':
                        class_index = int(float(label))  # Converti la label stringa in intero
                        metrics[metric].append(expected_calibration_error_class(all_probs[i], test_y)[class_index])
                    else:
                        metrics[metric].append(report[label][metric])
            average_metrics[label] = {metric: np.mean(values) for metric, values in metrics.items()}

    Val = expected_calibration_error(server_proba, test_y)
    Val_num = Val[0] if isinstance(Val, np.ndarray) else Val
    global_metrics = {
        "acc": server_cl_rp["accuracy"],
        "f1-weighted": server_cl_rp["weighted avg"]["f1-score"],
        "f1-macro": server_cl_rp["macro avg"]["f1-score"],
        "ece": Val_num
    }

    acceptance_percentages = []
    thresholds = np.arange(0.01, 1.0, 0.01)
    for threshold in thresholds:
        # Controlla se la probabilità della classe predetta è sopra la soglia
        above_threshold = np.max(server_proba, axis=1) >= threshold

        percentuale_dati_scartati = ( 1 - np.mean(above_threshold) ) * 100

        # Se nessuna probabilità è sopra la soglia
        if not np.any(above_threshold):
             acceptance_percentages.append((100, 0.0))

        else:
        # Filtra le classi predette sopra la soglia
            pred_label = server_y_pred[above_threshold]
            
            # Controlla se la predizione corrisponde all'etichetta vera
            correct_predictions = pred_label == test_y[above_threshold]
            
            # Calcola la percentuale di accettazione
            percentage_correct = np.mean(correct_predictions) * 100
            acceptance_percentages.append((percentuale_dati_scartati, percentage_correct))

    # rimozione duplicati e ordinamento rispetto percenutali dati scartati
    acceptance_percentages = list(set(acceptance_percentages))
    acceptance_percentages.sort()

    summary_array = []

    # Creiamo un dizionario per memorizzare la percentuale di accettazione per ogni fascia
    summary_dict = {i: [] for i in range(0, 101, 5)}

    for discarded_percentage, acceptance_percentage in acceptance_percentages:
    # Trova la fascia di appartenenza
        range_key = int(discarded_percentage // 5) * 5
    
        # Append valore per fare poi la media   
        summary_dict[range_key].append(acceptance_percentage)

    # Creazione dell'array riassuntivo calcolando la media per ogni fascia
    summary_array = [(k, np.mean(summary_dict[k]) if summary_dict[k] else 0.0) for k in sorted(summary_dict.keys())]

    # print(summary_array)

    # Separiamo i range e i valori di percentuale
    ranges, values = zip(*summary_array)

    # Identifica gli indici dove il valore è 0.0 e non tutti i valori successivi sono zero
    for i in range(len(values)):
        if values[i] == 0.0:
            # Cerca il prossimo valore non zero
            j = i + 1
            while j < len(values) and values[j] == 0.0:
                j += 1
            
            # Se abbiamo trovato un valore non zero, facciamo l'interpolazione
            if j < len(values):
                # Interpolazione lineare tra i valori validi più vicini
                values = list(values)
                if i > 0 and values[i-1] != 0.0:
                    values[i] = values[i-1] + (values[j] - values[i-1]) * (ranges[i] - ranges[i-1]) / (ranges[j] - ranges[i-1])
                else:
                    values[i] = values[j]

    return average_metrics, global_metrics, values


num_rows = [50, 1000]
with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Server.txt", "a") as f:
    f.write(f"\nALPHA = {Alpha} ")
    f.close()

for k in num_rows:
    server_average_metrics = []
    server_global_metrics = []
    server_acceptance_percentages = []

    for i in tqdm(range(5)):
        file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/valid {Alpha}/{i}/VALID_MATRIX.bin"
        valid_server = xgb.DMatrix(file_path_valid)

        model_federated_server = xgb.XGBClassifier()
        model_federated_server.load_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha {Alpha}/model_federato{i}.json')
        average_metrics, global_metrics, acceptance_percentages = ServerSub(k, valid_server, model_federated_server)

        server_average_metrics.append(average_metrics)
        server_global_metrics.append(global_metrics)
        server_acceptance_percentages.append(acceptance_percentages)

        with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Server.txt", "a") as f:
            f.write(f"\n Server_My_model_sub{k}: ")
            f.write(f'Metriche: {global_metrics}')
            f.close()

    # Aggregazione delle metriche medie per classe
    final_average_metrics = {}
    for label in server_average_metrics[0].keys():
        final_average_metrics[label] = {metric: np.mean([run_metrics[label][metric] for run_metrics in server_average_metrics])
                                        for metric in server_average_metrics[0][label].keys()}

    # Aggregazione delle metriche globali
    final_global_metrics = {metric: np.mean([run_metrics[metric] for run_metrics in server_global_metrics])
                            for metric in server_global_metrics[0].keys()}

    # Aggregazione delle percentuali di accettazione
    final_acceptance_percentages = np.mean(server_acceptance_percentages, axis=0)


    with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Server.txt", "a") as f:
        f.write(f"Server: {k}\n\n")
        f.write("Average Metrics per Class:\n")
        for label, metrics in final_average_metrics.items():
            f.write(f"Class {label}:\n")
            for metric, value in metrics.items():
                f.write(f"  Server_avg_{metric}_cl{label} = [ {value:.4f} ]\n")
            f.write("\n")
        
        f.write("Global Metrics:\n")
        for metric, value in final_global_metrics.items():
            f.write(f"  Server_{metric} = [ {value:.4f} ]\n")
        f.write("\n")
        
        f.write("Acceptance Percentages:\n")
        f.write('[' + ', '.join(f'{v:.5f}' for v in final_acceptance_percentages) + ']')
        f.write("\n")




########################################################################################################################################

## SERVER SKLEARN

########################################################################################################################################
print("\n")
print("################################################################################")
print("valutazione server Sklearn:")

def ServerSklearn(n, valid_server, model_federated):

    if n == 1:
        cl_federated = CalibratedClassifierCV(model_federated, cv="prefit")
        cl_federated.fit(valid_server.get_data(), valid_server.get_label())
    else:
        data = valid_server.get_data().toarray()
        labels = valid_server.get_label()
        df = pd.DataFrame(data)
        df['label'] = labels

        num_rows_per_class = n
        value_counts = df['label'].value_counts()
        l = []
        # Campiona righe per ciascuna classe
        for label, count in value_counts.items():
            k = df[df['label'] == label]
            frac = min(num_rows_per_class / count, 1.0)
            selected_rows = k.sample(frac=frac, random_state=42)
            l.append(selected_rows)
        result_df = pd.concat(l)
        sampled_data = result_df.drop(columns=['label']).values
        sampled_labels = result_df['label'].values
        sampled_matrix = xgb.DMatrix(sampled_data, label=sampled_labels)

        cl_federated = CalibratedClassifierCV(model_federated, cv="prefit")
        cl_federated.fit(sampled_matrix.get_data(), sampled_matrix.get_label())


    server_sklearn_y_pred = cl_federated.predict(test_x)
    server_sklearn_proba = cl_federated.predict_proba(test_x)
    server_cl_rp = classification_report(test_y, server_sklearn_y_pred, output_dict= True)

    all_reports = [server_cl_rp]
    all_probs = [server_sklearn_proba]

    average_metrics = {}
    for label in all_reports[0].keys():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = {'precision': [], 'recall': [], 'f1-score': [], 'ece_per_class': []}
            for i, report in enumerate(all_reports):
                for metric in metrics:
                    if metric == 'ece_per_class':
                        class_index = int(float(label))  # Converti la label stringa in intero
                        metrics[metric].append(expected_calibration_error_class(all_probs[i], test_y)[class_index])
                    else:
                        metrics[metric].append(report[label][metric])
            average_metrics[label] = {metric: np.mean(values) for metric, values in metrics.items()}

    Val = expected_calibration_error(server_sklearn_proba, test_y)
    Val_num = Val[0] if isinstance(Val, np.ndarray) else Val
    global_metrics = {
        "acc": server_cl_rp["accuracy"],
        "f1-weighted": server_cl_rp["weighted avg"]["f1-score"],
        "f1-macro": server_cl_rp["macro avg"]["f1-score"],
        "ece": Val_num
    }

    acceptance_percentages = []
    thresholds = np.arange(0.01, 1.0, 0.01)
    for threshold in thresholds:
        # Controlla se la probabilità della classe predetta è sopra la soglia
        above_threshold = np.max(server_sklearn_proba, axis=1) >= threshold

        percentuale_dati_scartati = ( 1 - np.mean(above_threshold) ) * 100

        # Se nessuna probabilità è sopra la soglia
        if not np.any(above_threshold):
             acceptance_percentages.append((100, 0.0))

        else:
        # Filtra le classi predette sopra la soglia
            pred_label = server_sklearn_y_pred[above_threshold]
            
            # Controlla se la predizione corrisponde all'etichetta vera
            correct_predictions = pred_label == test_y[above_threshold]
            
            # Calcola la percentuale di accettazione
            percentage_correct = np.mean(correct_predictions) * 100
            acceptance_percentages.append((percentuale_dati_scartati, percentage_correct))

    # rimozione duplicati e ordinamento rispetto percenutali dati scartati
    acceptance_percentages = list(set(acceptance_percentages))
    acceptance_percentages.sort()

    summary_array = []

    # Creiamo un dizionario per memorizzare la percentuale di accettazione per ogni fascia
    summary_dict = {i: [] for i in range(0, 101, 5)}

    for discarded_percentage, acceptance_percentage in acceptance_percentages:
    # Trova la fascia di appartenenza
        range_key = int(discarded_percentage // 5) * 5
    
        # Append valore per fare poi la media   
        summary_dict[range_key].append(acceptance_percentage)

    # Creazione dell'array riassuntivo calcolando la media per ogni fascia
    summary_array = [(k, np.mean(summary_dict[k]) if summary_dict[k] else 0.0) for k in sorted(summary_dict.keys())]

    # print(summary_array)

    # Separiamo i range e i valori di percentuale
    ranges, values = zip(*summary_array)

    # Identifica gli indici dove il valore è 0.0 e non tutti i valori successivi sono zero
    for i in range(len(values)):
        if values[i] == 0.0:
            # Cerca il prossimo valore non zero
            j = i + 1
            while j < len(values) and values[j] == 0.0:
                j += 1
            
            # Se abbiamo trovato un valore non zero, facciamo l'interpolazione
            if j < len(values):
                # Interpolazione lineare tra i valori validi più vicini
                values = list(values)
                if i > 0 and values[i-1] != 0.0:
                    values[i] = values[i-1] + (values[j] - values[i-1]) * (ranges[i] - ranges[i-1]) / (ranges[j] - ranges[i-1])
                else:
                    values[i] = values[j]

    return average_metrics, global_metrics, values

num_rows = [1,50, 1000]
for row in num_rows:

    if row == 1:
        with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Server_Sklearn.txt", "a") as f:
            f.write(f"\n Server_Sklearn_Defautl: ")
    else:
        with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Server_Sklearn.txt", "a") as f:
            f.write(f"\n Server_Sklearn{row}: ")

    sklearn_average_metrics = []
    sklearn_global_metrics = []
    sklearn_acceptance_percentages = []

    for i in tqdm(range(5)):
        
        file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/valid {Alpha}/{i}/VALID_MATRIX.bin"
        valid_server = xgb.DMatrix(file_path_valid)

        model_federated_sklearn = xgb.XGBClassifier()
        model_federated_sklearn.load_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha {Alpha}/model_federato{i}.json')
        average_metrics, global_metrics, acceptance_percentages = ServerSklearn(row, valid_server, model_federated_sklearn)


        sklearn_average_metrics.append(average_metrics)
        sklearn_global_metrics.append(global_metrics)
        sklearn_acceptance_percentages.append(acceptance_percentages)

        with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Server_Sklearn.txt", "a") as f:
            f.write(f"\n Server_Sklearn: ")
            f.write(f'Metriche: {global_metrics}')
            f.close()

    # Aggregazione delle metriche medie per classe
    final_average_metrics = {}
    for label in sklearn_average_metrics[0].keys():
        final_average_metrics[label] = {metric: np.mean([run_metrics[label][metric] for run_metrics in sklearn_average_metrics])
                                        for metric in sklearn_average_metrics[0][label].keys()}

    # Aggregazione delle metriche globali
    final_global_metrics = {metric: np.mean([run_metrics[metric] for run_metrics in sklearn_global_metrics])
                            for metric in sklearn_global_metrics[0].keys()}

    # Aggregazione delle percentuali di accettazione
    final_acceptance_percentages = np.mean(sklearn_acceptance_percentages, axis=0)


    with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Server_Sklearn.txt", "a") as f:
        f.write(f"\nServer_Sklearn: alpha = {Alpha}\n\n")
        f.write("Average Metrics per Class:\n")
        for label, metrics in final_average_metrics.items():
            f.write(f"Class {label}:\n")
            for metric, value in metrics.items():
                f.write(f"  Server_avg_{metric}_cl{label} = [ {value:.4f} ]\n")
            f.write("\n")
        
        f.write("Global Metrics:\n")
        for metric, value in final_global_metrics.items():
            f.write(f"  Server_{metric} = [ {value:.4f} ]\n")
        f.write("\n")
        
        f.write("Acceptance Percentages:\n")
        f.write('[' + ', '.join(f'{v:.5f}' for v in final_acceptance_percentages) + ']')
        f.write("\n")




# # ----------------------------------------------------------- clients -----------------------------------------------------------------------------------------------------------------------
print("\n")
print("################################################################################")
print("valutazione lato client myModel sub: ")

def process_partition(i, j, num_rows_per_class, model_federato):

    file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/valid {Alpha}/{j}/valid_matrix{i}.bin"
    valid_client = xgb.DMatrix(file_path_valid)
    data = valid_client.get_data().toarray()
    labels = valid_client.get_label()
    df = pd.DataFrame(data)
    df['label'] = labels

    value_counts = df['label'].value_counts()
    l = []
    # Campiona righe per ciascuna classe
    for label, count in value_counts.items():
        k = df[df['label'] == label]
        frac = min(num_rows_per_class / count, 1.0)
        selected_rows = k.sample(frac=frac, random_state=42)
        l.append(selected_rows)

    result_df = pd.concat(l)
    sampled_data = result_df.drop(columns=['label']).values
    sampled_labels = result_df['label'].values
    sampled_matrix = xgb.DMatrix(sampled_data, label=sampled_labels)

    
    cl_federated = MyCalibrator(model_federato, input_size=10, output_size=10)
    cl_federated.fit(sampled_matrix.get_data(), sampled_matrix.get_label())

    client_y_pred = cl_federated.predict(test_x)
    client_proba = cl_federated.predict_proba(test_x)
    client_cl_rp = classification_report(test_y, client_y_pred, output_dict= True)

    all_reports = [client_cl_rp]
    all_probs = [client_proba]

    average_metrics = {}
    for label in all_reports[0].keys():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = {'precision': [], 'recall': [], 'f1-score': [], 'ece_per_class': []}
            for i, report in enumerate(all_reports):
                for metric in metrics:
                    if metric == 'ece_per_class':
                        class_index = int(float(label))  # Converti la label stringa in intero
                        metrics[metric].append(expected_calibration_error_class(all_probs[i], test_y)[class_index])
                    else:
                        metrics[metric].append(report[label][metric])
            average_metrics[label] = {metric: np.mean(values) for metric, values in metrics.items()}
    
    Val = expected_calibration_error(client_proba, test_y)
    Val_num = Val[0] if isinstance(Val, np.ndarray) else Val
    global_metrics = {
        "acc": client_cl_rp["accuracy"],
        "f1-weighted": client_cl_rp["weighted avg"]["f1-score"],
        "f1-macro": client_cl_rp["macro avg"]["f1-score"],
        "ece": Val_num
    }

    # acceptance_percentages = []
    # thresholds = np.arange(0.01, 1.0, 0.01)
    # for threshold in thresholds:
    #      # Controlla se la probabilità della classe predetta è sopra la soglia
    #     above_threshold = np.max(client_proba, axis=1) >= threshold

    #     if not np.any(above_threshold):
    #          acceptance_percentages.append(0.0)
    #     else:
    #         # Filtra le classi predette sopra la soglia
    #         pred_label = client_y_pred[above_threshold]
            
    #         # Controlla se la predizione corrisponde all'etichetta vera
    #         correct_predictions = pred_label == test_y[above_threshold]
            
    #         # Calcola la percentuale di accettazione
    #         percentage_correct = np.mean(correct_predictions) * 100
    #         acceptance_percentages.append(percentage_correct)

    acceptance_percentages = []
    thresholds = np.arange(0.01, 1.0, 0.01)
    for threshold in thresholds:
        # Controlla se la probabilità della classe predetta è sopra la soglia
        above_threshold = np.max(client_proba, axis=1) >= threshold

        percentuale_dati_scartati = ( 1 - np.mean(above_threshold) ) * 100

        # Se nessuna probabilità è sopra la soglia
        if not np.any(above_threshold):
             acceptance_percentages.append((100, 0.0))

        else:
        # Filtra le classi predette sopra la soglia
            pred_label = client_y_pred[above_threshold]
            
            # Controlla se la predizione corrisponde all'etichetta vera
            correct_predictions = pred_label == test_y[above_threshold]
            
            # Calcola la percentuale di accettazione
            percentage_correct = np.mean(correct_predictions) * 100
            acceptance_percentages.append((percentuale_dati_scartati, percentage_correct))

    # rimozione duplicati e ordinamento rispetto percenutali dati scartati
    acceptance_percentages = list(set(acceptance_percentages))
    acceptance_percentages.sort()

    summary_array = []

    # Creiamo un dizionario per memorizzare la percentuale di accettazione per ogni fascia
    summary_dict = {i: [] for i in range(0, 101, 5)}

    for discarded_percentage, acceptance_percentage in acceptance_percentages:
    # Trova la fascia di appartenenza
        range_key = int(discarded_percentage // 5) * 5
    
        # Append valore per fare poi la media   
        summary_dict[range_key].append(acceptance_percentage)

    # Creazione dell'array riassuntivo calcolando la media per ogni fascia
    summary_array = [(k, np.mean(summary_dict[k]) if summary_dict[k] else 0.0) for k in sorted(summary_dict.keys())]

    # print(summary_array)

    # Separiamo i range e i valori di percentuale
    ranges, values = zip(*summary_array)

    # Identifica gli indici dove il valore è 0.0 e non tutti i valori successivi sono zero
    for i in range(len(values)):
        if values[i] == 0.0:
            # Cerca il prossimo valore non zero
            j = i + 1
            while j < len(values) and values[j] == 0.0:
                j += 1
            
            # Se abbiamo trovato un valore non zero, facciamo l'interpolazione
            if j < len(values):
                # Interpolazione lineare tra i valori validi più vicini
                values = list(values)
                if i > 0 and values[i-1] != 0.0:
                    values[i] = values[i-1] + (values[j] - values[i-1]) * (ranges[i] - ranges[i-1]) / (ranges[j] - ranges[i-1])
                else:
                    values[i] = values[j]

    # # Ricostruisci il summary array con i valori interpolati
    # summary_array_interpolated = list(zip(ranges, values))

    #print(values)
    # # Stampa o ritorna l'array interpolato
    # print(summary_array_interpolated)


    return average_metrics, global_metrics, values


def ClientRun(j, num_rows, model):
    average_metrics_C = []
    global_metrics_C = []
    acceptance_percentages_C = []

    for i in range (10):
        average_metrics, global_metrics, acceptance_percentages = process_partition(i, j, num_rows, model)
        average_metrics_C.append(average_metrics)
        global_metrics_C.append(global_metrics)
        acceptance_percentages_C.append(acceptance_percentages)

    # Aggregazione delle metriche medie per classe
    final_average_metrics = {}
    for label in average_metrics_C[0].keys():
        final_average_metrics[label] = {metric: np.mean([run_metrics[label][metric] for run_metrics in average_metrics_C])
                                        for metric in average_metrics_C[0][label].keys()}

    # Aggregazione delle metriche globali
    final_global_metrics = {metric: np.mean([run_metrics[metric] for run_metrics in global_metrics_C])
                            for metric in global_metrics_C[0].keys()}

    # Aggregazione delle percentuali di accettazione
    final_acceptance_percentages = np.mean(acceptance_percentages_C, axis=0)
   
    return final_average_metrics, final_global_metrics, final_acceptance_percentages

num_rows = [50, 1000]

with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Clients.txt", "a") as f:
    f.write(f"\nALPHA = {Alpha} ")
    f.close()

for row in num_rows:
    client_average_metrics = []
    client_global_metrics = []
    client_acceptance_percentages = []

    for i in tqdm(range(5)):

        model_federated = xgb.XGBClassifier()
        model_federated.load_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha {Alpha}/model_federato{i}.json')
        average_metrics, global_metrics, acceptance_percentages = ClientRun(i, row, model_federated)

        client_average_metrics.append(average_metrics)
        client_global_metrics.append(global_metrics)
        client_acceptance_percentages.append(acceptance_percentages)

        with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Clients.txt", "a") as f:
            f.write(f"\n Media clients My_model_sub{row}: ")
            f.write(f'Metriche: {global_metrics}')
            f.close()
            
    # Aggregazione delle metriche medie per classe
    final_average_metrics = {}
    for label in client_average_metrics[0].keys():
        final_average_metrics[label] = {metric: np.mean([run_metrics[label][metric] for run_metrics in client_average_metrics])
                                        for metric in client_average_metrics[0][label].keys()}

    # Aggregazione delle metriche globali
    final_global_metrics = {metric: np.mean([run_metrics[metric] for run_metrics in client_global_metrics])
                            for metric in client_global_metrics[0].keys()}

    # Aggregazione delle percentuali di accettazione
    final_acceptance_percentages = np.mean(client_acceptance_percentages, axis=0)


    with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_Clients.txt", "a") as f:
        f.write(f"\n Client: {row}\n\n")
        f.write("Average Metrics per Class:\n")
        for label, metrics in final_average_metrics.items():
            f.write(f"Class {label}:\n")
            for metric, value in metrics.items():
                f.write(f"  Client_avg_{metric}_cl{label} = [ {value:.4f} ]\n")
            f.write("\n")
        
        f.write("Global Metrics:\n")
        for metric, value in final_global_metrics.items():
            f.write(f"  Client_{metric} = [ {value:.4f} ]\n")
        f.write("\n")
        
        f.write("Acceptance Percentages:\n")
        # thresholds = np.arange(0.1, 1.0, 0.1)
        # for threshold, percentage in zip(thresholds, final_acceptance_percentages):
        #     f.write(f"  Threshold {threshold:.1f}: {percentage:.2f}%\n")
        f.write('[' + ', '.join(f'{v:.5f}' for v in final_acceptance_percentages) + ']')
        f.write("\n")



