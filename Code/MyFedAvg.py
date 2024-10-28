import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from logging import ERROR, INFO, WARN
from flwr.common.logger import log
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score

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

file_path_test = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/TEST_MATRIX.bin"
test_server= xgb.DMatrix(file_path_test)

test_x = test_server.get_data()
test_y = test_server.get_label()

Alpha = "01"
# model_federated = xgb.XGBClassifier()
# model_federated.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha 02/model_federato4.json')

def run(nrpc, j):
    partitions = []
    num_clients = 10
    num_rounds_fed = 50

    num_rows_per_class = nrpc
    for i in range(num_clients):
        file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/valid {Alpha}/{j}/valid_matrix{i}.bin"
        valid_matrix = xgb.DMatrix(file_path_valid)
        if num_rows_per_class == 1:
            partitions.append(valid_matrix)
        else:
            # partitions.append(valid_matrix)

            # prendo solo una parte di dati per ogni classe
            data = valid_matrix.get_data().toarray()
            labels = valid_matrix.get_label()
            df = pd.DataFrame(data)
            df['label'] = labels
            
            
            value_counts = df['label'].value_counts()
            l = []
            # Campiona righe per ciascuna classe
            for label, count in value_counts.items():
                k = df[df['label'] == label]
                frac = min(num_rows_per_class / count, 1.0)

                
                #     # if aggiunto per avere 1000 righe per classe
                #     if frac == 1.0:
                #         selected_rows = k.sample(n=num_rows_per_class, replace=True, random_state=42)
                #     else: 
                #         selected_rows = k.sample(frac=frac, random_state=42)
                # else: 
                selected_rows = k.sample(frac=frac, random_state=42)
                l.append(selected_rows)
            result_df = pd.concat(l)
            sampled_data = result_df.drop(columns=['label']).values
            sampled_labels = result_df['label'].values
            sampled_matrix = xgb.DMatrix(sampled_data, label=sampled_labels)

            partitions.append(sampled_matrix)

            val = sampled_matrix.get_label()
            unique, counts = np.unique(val, return_counts=True)
            print(unique, counts)


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
        def __init__(self, base_model, input_size, output_size, lr=0.15, epochs=3):
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
            #optimizer = optim.Adam(self.calibrator.parameters(), lr=self.lr)
            optimizer = torch.optim.RMSprop(self.calibrator.parameters(), lr=self.lr)
            #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            
            for epoch in range(self.epochs):
                self.calibrator.train()
                optimizer.zero_grad()
                outputs = self.calibrator(X_calib)
                loss = criterion(outputs, y_calib)
                loss.backward()
                optimizer.step()
                #scheduler.step()

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



    print("calibrazione federata: ")

    class MyCalibratorClient(fl.client.NumPyClient):
        def __init__(self, model, X_train, y_train):
            self.model = model
            self.X_train = X_train
            self.y_train = y_train

        def get_parameters(self):
            return self.model.get_parameters()

        def set_parameters(self, parameters):
            a_params, b_params = parameters
            self.model.set_parameters(a_params, b_params)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.fit(self.X_train, self.y_train)
            new_parameters = self.get_parameters()
            return new_parameters, self.X_train.shape[0], {}

        # def evaluate(self, parameters, config):
        #     self.set_parameters(parameters)
        #     loss = self.model.fit(self.X_train, self.y_train).score(self.X_train, self.y_train)
        #     return loss, self.X_train.shape[0], {}

    # Funzione per creare client
    def client_fn(cid: str, partition):
        valid_dmatrix = partition
        X_train = valid_dmatrix.get_data()
        y_train = valid_dmatrix.get_label()
        base_model = model_federated
        model = MyCalibrator(base_model, input_size=10, output_size=10)
        return MyCalibratorClient(model, X_train, y_train)

    # Funzione per aggregare i parametri
    def aggregate_parameters(parameters_list):
        avg_a_params = np.mean([parameters[0] for parameters in parameters_list], axis=0)
        avg_b_params = np.mean([parameters[1] for parameters in parameters_list], axis=0)
        return [avg_a_params, avg_b_params]

    # Definizione della strategia personalizzata
    class CustomStrategy(fl.server.strategy.FedAvg):
        def __init__(self, initial_parameters, **kwargs):
            super().__init__(**kwargs)
            self.initial_parameters = initial_parameters

        def aggregate_fit(self, rnd, results, failures):
            if not results:
                return self.initial_parameters, {}
            
            parameters_list = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
            aggregated_parameters = self.aggregate_parameters(parameters_list)
            
            aggregated_parameters_proto = fl.common.ndarrays_to_parameters(aggregated_parameters)
            
            self.final_parameters = aggregated_parameters_proto
            return aggregated_parameters_proto, {}

        def aggregate_parameters(self, parameters_list):
            avg_a_params = np.mean([parameters[0] for parameters in parameters_list], axis=0)
            avg_b_params = np.mean([parameters[1] for parameters in parameters_list], axis=0)
            return [avg_a_params, avg_b_params]

        def on_fit_end(self):
            return self.final_parameters
        

    calibrator = Calibrator(input_size=10, output_size=10)
    initial_parameters = calibrator.get_parameters()
    fl_initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)

    num_clients = 10

    strategy = CustomStrategy(
            initial_parameters=initial_parameters,
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=10,
        ) 


    # Inizia il server
    strategy = CustomStrategy(initial_parameters=fl_initial_parameters)
    print("start_server: ")
    def start_server():
        fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds = num_rounds_fed ),
                strategy=strategy
            )

    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    def start_clients_calibration(num_clients, partitions):
        def start_client_calibration(partition, node_id):
            client = client_fn(node_id, partition)
            fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())

        threads = []
        for i in range(num_clients):
            thread = threading.Thread(target=start_client_calibration, args=(partitions[i], i))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    # Avvia i client in parallelo
    start_clients_calibration(num_clients, partitions)
    server_thread.join()


    base_model = model_federated
    final_model = MyCalibrator(base_model, input_size=10, output_size=10)
    final_parameters = strategy.on_fit_end()
    final_model.set_parameters(fl.common.parameters_to_ndarrays(final_parameters)[0], fl.common.parameters_to_ndarrays(final_parameters)[1])
        # torch.save({
        #     'a_params': final_parameters[0],
        #     'b_params': final_parameters[1],
        # }, 'federated_calibrator_model.pth')
    print("a: ", fl.common.parameters_to_ndarrays(final_parameters)[0])
    print("b: ", fl.common.parameters_to_ndarrays(final_parameters)[1])
    y_p = final_model.predict_proba(test_x)
    y_pred = np.argmax(y_p, axis=1)
    cl_rp = classification_report(test_y, y_pred, output_dict= True)

   

    all_reports = [cl_rp]
    all_probs = [y_p]

    
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

    Val = expected_calibration_error(y_p, test_y)
    Val_num = Val[0] if isinstance(Val, np.ndarray) else Val
    global_metrics = {
        "acc": cl_rp["accuracy"],
        "f1-weighted": cl_rp["weighted avg"]["f1-score"],
        "f1-macro": cl_rp["macro avg"]["f1-score"],
        "ece ": Val_num
    }
    
    file_path_test = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_algo_alpha_n/TEST_MATRIX_7_CLASSI.bin"
    test_matrix = xgb.DMatrix(file_path_test)
    testx = test_matrix.get_data().toarray()
    testy = test_matrix.get_label()
    y_p = final_model.predict_proba(testx)
    y_pred = np.argmax(y_p, axis=1)

    acceptance_percentages = []
    thresholds = np.arange(0.01, 1.0, 0.01)
    for threshold in thresholds:
        # Controlla se la probabilità della classe predetta è sopra la soglia
        above_threshold = np.max(y_p, axis=1) >= threshold

        percentuale_dati_scartati = ( 1 - np.mean(above_threshold) ) * 100

        # Se nessuna probabilità è sopra la soglia
        if not np.any(above_threshold):
             acceptance_percentages.append((100, 0.0))

        else:
        # Filtra le classi predette sopra la soglia
            pred_label = y_pred[above_threshold]
            
            # Controlla se la predizione corrisponde all'etichetta vera
            correct_predictions = pred_label == testy[above_threshold]
            
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
    

    # f = open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\clie_pesi.txt", "a")
    # f.write(f'MyModel__{num_rows_per_class}: ')
    # f.write(f'Metriche: {metrics}')
    # f.write(f'Valore ece : {federated_ece}\n')
    # f.close()

# num_rows = [50, 1000]
num_rows = [1]
for k in num_rows:
    all_run_average_metrics = []
    all_run_global_metrics = []
    all_run_acceptance_percentages = []

    for i in range(5):
        model_federated = xgb.XGBClassifier()
        model_federated.load_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha {Alpha}/model_federato{i}.json')
        average_metrics, global_metrics, acceptance_percentages = run(k, i)
        all_run_average_metrics.append(average_metrics)
        all_run_global_metrics.append(global_metrics)
        all_run_acceptance_percentages.append(acceptance_percentages)

        with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_myModel.txt", "a") as f:
            f.write(f"\n myModel__{k}: ")
            f.write(f'Metriche: {global_metrics}')
            f.close()

    # Aggregazione delle metriche medie per classe
    final_average_metrics = {}
    for label in all_run_average_metrics[0].keys():
        final_average_metrics[label] = {metric: np.mean([run_metrics[label][metric] for run_metrics in all_run_average_metrics])
                                        for metric in all_run_average_metrics[0][label].keys()}

    # Aggregazione delle metriche globali
    final_global_metrics = {metric: np.mean([run_metrics[metric] for run_metrics in all_run_global_metrics])
                            for metric in all_run_global_metrics[0].keys()}

    # Aggregazione delle percentuali di accettazione
    final_acceptance_percentages = np.mean(all_run_acceptance_percentages, axis=0)


    with open("C:\\Users\\lukyl\\OneDrive\\Desktop\\Flower\\final_metrics\\final_metrics_myModel.txt", "a") as f:
        f.write("\n\nAverage Metrics per Class:\n")
        for label, metrics in final_average_metrics.items():
            f.write(f"Class {label}:\n")
            for metric, value in metrics.items():
                f.write(f"  myModel_avg_{metric}_cl{label} = [ {value:.4f} ]\n")
            f.write("\n")
        
        f.write("Global Metrics:\n")
        for metric, value in final_global_metrics.items():
            f.write(f"  myModel_{metric} = [ {value:.4f} ]\n")
        f.write("\n")
        
        f.write("Acceptance Percentages:\n")
        # thresholds = np.arange(0.1, 1.0, 0.1)
        # for threshold, percentage in zip(thresholds, final_acceptance_percentages):
        #     f.write(f"  Threshold {threshold:.1f}: {percentage:.2f}%\n")
        f.write('[' + ', '.join(f'{v:.5f}' for v in final_acceptance_percentages) + ']')
        f.write("\n")
        

    print("Active threads:", threading.enumerate())
    
# model_federated = xgb.XGBClassifier()
# model_federated.load_model(f'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha 01/model_federato0.json')
# run_metrics = run(50)