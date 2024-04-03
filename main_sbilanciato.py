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

#df = dd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file.csv')

# # ordina il dataset utilizzando la colonna IPV4_DST_ADDR -> non funziona bene, restituisce un dataFrame che da errore quando si chiama df.loc

# df_ordinato = df.set_index('IPV4_DST_ADDR')

# # utilizzo questo metodo che funziona correttamente

# value_counts_ipDst = df['IPV4_DST_ADDR'].value_counts().compute()
# value_counts_ipDst_sorted = value_counts_ipDst.sort_values(ascending= False)
# #print(value_counts_ipDst_sorted[0:20])


# IPV4_DST_ADDR
# 192.168.1.195     2118567
# 192.168.1.1       2012390
# 192.168.1.190     1805130
# 192.168.1.184     1653333
# 192.168.1.180     1288737
# 192.168.1.152     1190525
# 192.168.1.49      1016646
# 192.168.1.194      867216
# 192.168.1.169      730266
# 192.168.1.193      538197
# 192.168.1.186      457054
# 192.168.1.46       454179
# 176.28.50.165      342235
# 192.168.1.79       310575
# 192.168.1.30       243737
# 192.168.1.31       238884
# 52.28.231.150      149165
# 18.194.169.124     146956
# 192.168.1.38        77513
# 192.168.1.32        71900

# creo la lista indirizzi e lista dimensioni così da non doverle ricalcolare tutte le volte

lista_indirizzi = ["192.168.1.195", "192.168.1.1", "192.168.1.190", "192.168.1.184", "192.168.1.180", "192.168.1.152", "192.168.1.49", 
                   "192.168.1.194", "192.168.1.169", "192.168.1.193", "192.168.1.186", "192.168.1.46", "176.28.50.165", "192.168.1.79",
                     "192.168.1.30", "192.168.1.31", "52.28.231.150", "18.194.169.124", "192.168.1.38", "192.168.1.32"]

lista_dimensioni = [2118567, 2012390, 1805130, 1653333, 1288737, 1190525, 1016646, 867216, 730266, 538197, 457054, 454179, 342235, 
                    310575, 243737, 238884, 149165, 146956, 77513, 71900]


# creo partizioni del dataFrame simili a quelle del caso bilanciato, quindi di circa 338k righe per ogni client

n_partitions = 10
partitions = []
for i in tqdm(range(n_partitions)):
    # # estrazione dal dataframe delle righe corrispondenti al valore indirizzo ip
    # p = df.loc[df['IPV4_DST_ADDR'] == lista_indirizzi[i]]
    # # calcolo percentuale righe da tenere 
    # n = round(338000/lista_dimensioni[i],3)
    # # rimuovo queste colonne dal dataset di train
    # X = p.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'Attack','Attack_label'])
    # # creazione array con le true labels
    # Y = p['Attack_label']

    #distribuzione iniziale
    # num_rows = lista_dimensioni[i]
    # value_counts_attack = p['Attack'].value_counts().compute()
    # print(value_counts_attack/num_rows)

    # # assegno circa 338k righe ad ogni client, mantenedo la distribuzione originale (riferita alla partizione di quell'indirizzo ip)
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=n, shuffle = True)

    # # divido in test e train per ogni client
    # client_train_x, client_test_x, client_train_y, client_test_y = train_test_split(x_test, y_test, test_size=0.2, shuffle = True)
    
    # # creo le dmatrix per poter addestrare e valutare i client, e le salvo
    # train_matrix = xgb.DMatrix(client_train_x, label=client_train_y)
    # valid_matrix = xgb.DMatrix(client_test_x, label=client_test_y)

    # train_matrix.save_binary(file_path_train)
    # valid_matrix.save_binary(file_path_valid)


    file_path_train = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_sbilanciato/train_matrix{i}.bin"
    file_path_valid = f"C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/Dmatrix_sbilanciato/valid_matrix{i}.bin"
    train_matrix = xgb.DMatrix(file_path_train)
    valid_matrix = xgb.DMatrix(file_path_valid)

    # calcolo dimensioni di train e test, servono al client per fare la valutazione delle prestazioni
    train_num = train_matrix.num_row()
    valid_num = valid_matrix.num_row()

    # inserisco in partitions le due dmatrix e le loro dimensioni 
    partitions.append((train_matrix, valid_matrix, train_num, valid_num))

    ## valutazione distribuzione classi nella partizione usata per l'addestramento
    # num_rows = len(client_train_x)
    # value_counts_attack = client_train_y.value_counts().compute()
    # v = np.zeros(10)
    # for item, count in value_counts_attack.items():
    #     v [item] = count/num_rows
    # print(v)
    # #{'scanning': 0, 'Benign': 1, 'password': 2, 'ddos': 3, 'xss': 4, 'dos': 5, 'injection': 6, 'mitm': 7, 'ransomware': 8, 'backdoor': 9}

   
    
log(
        INFO,
        "Creazione e avvio client",
    )


num_clients = n_partitions

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