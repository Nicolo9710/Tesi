import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from logging import ERROR, INFO, WARN
from flwr.common.logger import log
from tqdm import tqdm

import threading
import flwr as fl
from tutorial import client


#creazione dataset NF_ToN_IoT_V2
# df = dd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file.csv')

# X = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES' , 'Attack','Attack_label'])
# Y = df['Attack_label']
# X_drop, X_keep, y_drop, y_keep = train_test_split(X, Y, test_size=0.2, shuffle = True, random_state=42) # 20% del dataframe totale

log(
        INFO,
        "Preparazione dataSet",
    )

n_partitions = 10 # deve essere uguale al numero di client che voglio usare
partitions = []

for i in tqdm(range(n_partitions)):
    # x_train, x_test, y_train, y_test = train_test_split(X_keep, y_keep, test_size=0.1, shuffle = True)# assegno 10% per ogni client 
    # client_train_x, client_test_x, client_train_y, client_test_y = train_test_split(x_test, y_test, test_size=0.2, shuffle = True)# divido in test e train per ogni client
    # train_matrix = xgb.DMatrix(client_train_x, label=client_train_y)
    # valid_matrix = xgb.DMatrix(client_test_x, label=client_test_y)
    
    
    file_path_train = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_bilanciato/train_matrix{i}.bin"
    file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_bilanciato/valid_matrix{i}.bin"

    train_matrix = xgb.DMatrix(file_path_train)
    valid_matrix = xgb.DMatrix(file_path_valid)

    train_num = train_matrix.num_row()
    valid_num = valid_matrix.num_row()
    partitions.append((train_matrix, valid_matrix, train_num, valid_num))

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