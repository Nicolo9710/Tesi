import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
import xgboost as xgb

df = dd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file4.csv')
#print(df.head(10)) 

# controllo distribuzione classi

# num_rows = len(df)
# value_counts_attack = df['Attack'].value_counts().compute()
# print(value_counts_attack/num_rows)

# Attack                    Attack_label       num_row: 16.940.496        Distribuzione:   
# mitm             7723     7       7723                                mitm          0.000456
# ddos          2026234     3    2026234                                ddos          0.119609
# ransomware       3425     8       3425                                ransomware    0.000202
# xss           2455020     4    2455020                                xss           0.144920
# scanning      3781419     0    3781419                                scanning      0.223218
# Benign        6099469     1    6099469                                Benign        0.360053
# backdoor        16809     9      16809                                backdoor      0.000992
# password      1153323     2    1153323                                password      0.068081
# dos            712609     5     712609                                dos           0.042065
# injection      684465     6     684465                                injection     0.040404

## statistiche file output_file4.csv
# Attack                            num_row: 16.940.076 (420 righe)         Distribuzione:
#                                                                          
# mitm             7514   ha tolto qui 209                              mitm          0.000444
# ddos          2026234                                                 ddos          0.119612
# ransomware       3425                                                 ransomware    0.000202
# xss           2455020                                                 xss           0.144924
# scanning      3781419                                                 scanning      0.223223
# Benign        6099258  ha tolto qui 211                               Benign        0.360049
# backdoor        16809                                                 backdoor      0.000992
# password      1153323                                                 password      0.068083
# dos            712609                                                 dos           0.042066
# injection      684465                                                 injection     0.040405

# column_names = list(df.columns)

# print(column_names)



# # Dati
# categories = ['mitm', 'ddos', 'ransomware', 'xss', 'scanning', 'Benign', 'backdoor', 'password', 'dos', 'injection']
# values = [0.000456, 0.119609, 0.000202, 0.144920, 0.223218, 0.360053, 0.000992, 0.068081, 0.042065, 0.040404]

# # Disegna l'istogramma
# plt.figure(figsize=(10, 6))
# plt.bar(categories, values, color='skyblue')
# plt.xlabel('Categoria')
# plt.ylabel('Frequenza')
# plt.title('Distribuzione delle categorie')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # controllo distribuzione di una partizione del 20% 

X = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label' , 'Attack','Attack_label'])
Y = df['Attack_label']
X_drop, X_keep, y_drop, y_keep = train_test_split(X, Y, test_size=0.2, shuffle = True)

# cols=[ 'L4_SRC_PORT',  'L4_DST_PORT',
#        'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
#        'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
#        'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL',
#        'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN',
#        'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
#        'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
#        'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
#        'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
#        'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
#        'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
#        'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
#        'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
#        'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE', 'Label']
# cont = 0
# for j in cols:
#     print(cont)
#     unique_values = X_keep[j].unique().compute()
#     for k in unique_values:
#         if k > 1e15:
#             print(j)
#     cont += 1

# num_rows = len(X_keep)
# value_counts_attack = y_keep.value_counts().compute()
# print(value_counts_attack/num_rows)
# print(num_rows)

#Distribuzione DataFrame:        Distribuzione Partizione           nrows_partizione: 3.386.477
# 
# mitm          0.000456          7    0.000456
# ddos          0.119609          3    0.119385
# ransomware    0.000202          8    0.000197
# xss           0.144920          4    0.145084   
# scanning      0.223218          0    0.223644
# Benign        0.360053          1    0.359649
# backdoor      0.000992          9    0.000977
# password      0.068081          2    0.068028
# dos           0.042065          5    0.042090
# injection     0.040404          6    0.040490

#creo 10 partizioni
# n_partitions = 10
# partitions = []
# for _ in range(n_partitions):
#     x_train, x_test, y_train, y_test = train_test_split(X_keep, y_keep, test_size=0.1, shuffle = True)
#     partitions.append((x_test, y_test))

# part5_x, part5_y = partitions[2]
# num_rows = len(part5_y)
# value_counts_attack = part5_y.value_counts().compute()
# print(value_counts_attack/num_rows)
# print(num_rows)


#Distribuzione DataFrame:        Distribuzione Partizione (20%)                 Distribuzione Partizione (5)      nrows: 338.287    
# 
# mitm          0.000456          7    0.000456                                 7    0.000461
# ddos          0.119609          3    0.119385                                 3    0.118801
# ransomware    0.000202          8    0.000197                                 8    0.000171
# xss           0.144920          4    0.145084                                 4    0.144052
# scanning      0.223218          0    0.223644                                 0    0.224150
# Benign        0.360053          1    0.359649                                 1    0.360779
# backdoor      0.000992          9    0.000977                                 9    0.000970
# password      0.068081          2    0.068028                                 2    0.068536
# dos           0.042065          5    0.042090                                 5    0.041885
# injection     0.040404          6    0.040490                                 6    0.040194

# controllo se le partizioni contengono righe differenti
# part2_x, part2_y = partitions[2]
# print(part2_x.head(5))
# part5_x, part5_y = partitions[5]
# print(part5_x.head(5))
    
# # part2_x: 
#         L4_SRC_PORT  L4_DST_PORT  PROTOCOL  L7_PROTO  IN_BYTES  ...  DNS_QUERY_ID  DNS_QUERY_TYPE  DNS_TTL_ANSWER  FTP_COMMAND_RET_CODE  Label
# 134224        50593         5422         6       0.0        48  ...             0               0               0                     0      1      
# 51080         50729         1041         6       0.0        44  ...             0               0               0                     0      0      
# 270142        14116        37637         6       0.0        48  ...             0               0               0                     0      1      
# 277319         4711        29277         6       0.0        48  ...             0               0               0                     0      1      
# 177166        10824         9845         6       0.0        48  ...             0               0               0                     0      1      
    
# # part5_x:
#             L4_SRC_PORT  L4_DST_PORT  PROTOCOL  L7_PROTO  IN_BYTES  ...  DNS_QUERY_ID  DNS_QUERY_TYPE  DNS_TTL_ANSWER  FTP_COMMAND_RET_CODE  Label
# 120237        52724        49161         6       0.0        44  ...             0               0               0                     0      0      
# 126387         9419        61571         6       0.0        48  ...             0               0               0                     0      1      
# 165622        60339         5456         6       0.0        48  ...             0               0               0                     0      1      
# 391887         9019        48185         6       0.0        48  ...             0               0               0                     0      1      
# 144114        44547        39426         6       0.0        48  ...             0               0               0                     0      1      
    
# # feature importance
# import xgboost as xgb
# import matplotlib.pyplot as plt

# # loaded_model_sbilanciato = xgb.Booster()
# loaded_model_sbilanciato = xgb.XGBRFClassifier()
# loaded_model_sbilanciato.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/global_model_bilanciato.json')

# xgb.plot_importance(loaded_model_sbilanciato, importance_type='weight')
# plt.show()
## Get feature importance scores
# importance_scores = loaded_model_sbilanciato.get_score(importance_type='weight')

# ## Plot feature importance
# plt.bar(importance_scores.keys(), importance_scores.values())
# plt.xlabel('Feature')
# plt.ylabel('Importance Score')
# plt.title('Feature Importance Scores')
# plt.xticks(rotation=90)
# plt.show()

# xgb.plot_importance(loaded_model_sbilanciato, importance_type='weight') ##Plot importance based on fitted trees.
# plt.show()

# column_names = list(X.columns)
# # print(column_names[38])
# # print(column_names)

# indici = [3,1,11,26,0,6,8,34,16,12,4,9,28,33,20,17,27,5,29,10,7,30,37,32,14,40,2,21,39,13,31,18,38,24,22,15,23]
# for i in indici:
#     print(column_names[i])

## risultato:

# L7_PROTO
# L4_DST_PORT
# FLOW_DURATION_MILLISECONDS
# SRC_TO_DST_AVG_THROUGHPUT
# L4_SRC_PORT
# OUT_BYTES
# TCP_FLAGS
# TCP_WIN_MAX_OUT
# LONGEST_FLOW_PKT
# DURATION_IN
# IN_BYTES
# CLIENT_TCP_FLAGS
# NUM_PKTS_UP_TO_128_BYTES
# TCP_WIN_MAX_IN
# SRC_TO_DST_SECOND_BYTES
# SHORTEST_FLOW_PKT
# DST_TO_SRC_AVG_THROUGHPUT
# IN_PKTS
# NUM_PKTS_128_TO_256_BYTES
# SERVER_TCP_FLAGS
# OUT_PKTS
# NUM_PKTS_256_TO_512_BYTES
# DNS_QUERY_ID
# NUM_PKTS_1024_TO_1514_BYTES
# MIN_TTL
# FTP_COMMAND_RET_CODE
# PROTOCOL
# DST_TO_SRC_SECOND_BYTES
# DNS_TTL_ANSWER
# DURATION_OUT
# NUM_PKTS_512_TO_1024_BYTES
# MIN_IP_PKT_LEN
# DNS_QUERY_TYPE
# RETRANSMITTED_OUT_BYTES
# RETRANSMITTED_IN_BYTES
# MAX_TTL
# RETRANSMITTED_IN_PKTS

# dataset con colonna binaria (label)
# Label -----------> 120                        
# L4_DST_PORT -------> 107
# L4_SRC_PORT -------> 61
# IN_BYTES  --------> 22
# L7_PROTO ---------> 20
# DNS_QUERY_ID -------> 19
# CLIENT_TCP_FLAGS ------> 16
# NUM_PKTS_UP_TO_128_BYTES -----> 15
# OUT_BYTES -------> 14
# DST_TO_SRC_AVG_THROUGHPUT -----> 13
# TCP_FLAGS -------> 12
# IN_PKTS
# LONGEST_FLOW_PKT ------> 10
# SRC_TO_DST_AVG_THROUGHPUT ------> 9
# NUM_PKTS_512_TO_1024_BYTES ------> 8
# SHORTEST_FLOW_PKT
# FLOW_DURATION_MILLISECONDS
# DURATION_IN --------> 7
# TCP_WIN_MAX_IN ---------> 5
# TCP_WIN_MAX_OUT -------> 4
# DURATION_OUT
# NUM_PKTS_256_TO_512_BYTES--------> 3
# FTP_COMMAND_RET_CODE --------> 2
# DNS_TTL_ANSWER
# NUM_PKTS_128_TO_256_BYTES
# OUT_PKTS
# PROTOCOL
# DNS_QUERY_TYPE --------> 1
# ICMP_TYPE
# NUM_PKTS_1024_TO_1514_BYTES
# RETRANSMITTED_OUT_BYTES
# MAX_TTL