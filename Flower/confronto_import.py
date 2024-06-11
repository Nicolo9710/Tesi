import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report

df = dd.read_csv('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/output_file4.csv')

X = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label' , 'Attack','Attack_label'])
Y = df['Attack_label']
X_drop, x_test, y_drop, y_test = train_test_split(X, Y, test_size=0.3, shuffle = True)
# x_train, x_test, y_train, y_test = train_test_split(X_keep, y_keep, shuffle = True)
test_matrix = xgb.DMatrix(x_test, label=y_test)

# import con xgb.Booster()

clf = xgb.Booster()
clf.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/global_model_bilanciato.json')

# plt.figure(figsize=(200, 100))
# xgb.plot_tree(clf, num_trees=0)
# plt.show()

# # Extract the first few trees from the model
# first_tree = clf.trees_to_dataframe()#[['Feature', 'Split', 'Yes', 'No']]
# trees_xgb = clf.get_dump()
# file_path = "C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/splitting_criteria_tot2.txt"
# # Print the textual representation of each tree
# with open(file_path, "w") as file:
#     for i, tree in enumerate(trees_xgb):
#         file.write(f"Tree {i}:\n{tree}\n\n")


# xgb.plot_importance(clf, importance_type='weight') ##Plot importance based on fitted trees.
# plt.show()


# pred_boost = clf.predict(test_matrix)
# r = []
# for i in pred_boost:
#     r.append(i.argmax())

# cl_rp_xgb = classification_report(y_test, r)
# print(cl_rp_xgb)

# import con xgb.XGBClassifier()

classifier = xgb.XGBClassifier()
classifier.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/global_model_bilanciato.json')
#trees_classifier = classifier.get_booster().get_dump()
# xgb.plot_importance(classifier, importance_type='weight') ##Plot importance based on fitted trees.
# plt.show()
# pred_class = classifier.predict(x_test)
# cl_rp_class = classification_report(y_test, pred_class)
# print(cl_rp_class)

# import con xgb.XGBRFClassifier()

XGBRF_classifier = xgb.XGBRFClassifier()
XGBRF_classifier.load_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/global_model_bilanciato.json')

# trees_XGBRF = XGBRF_classifier.get_booster().get_dump()
# xgb.plot_importance(XGBRF_classifier, importance_type='weight') ##Plot importance based on fitted trees.
# plt.show()

# pred_class_XGBRFClass = XGBRF_classifier.predict(x_test)
# cl_rp_XGBRFClass = classification_report(y_test, pred_class_XGBRFClass)
# print(cl_rp_XGBRFClass)

# risultati:

# xgb.Booster()                                                 xgb.Classifier                                              xgb.XGBRFClassifier
#               precision    recall  f1-score   support                precision    recall  f1-score   support                 precision    recall  f1-score   support

#            0       0.95      0.99      0.97     75438            0       0.95      0.99      0.97     75438               0       0.95      0.99      0.97     75438
#            1       1.00      0.95      0.97    122497            1       1.00      0.95      0.97    122497               1       1.00      0.95      0.97    122497
#            2       0.92      0.92      0.92     23060            2       0.92      0.92      0.92     23060               2       0.92      0.92      0.92     23060
#            3       0.96      0.97      0.96     40616            3       0.96      0.97      0.96     40616               3       0.96      0.97      0.96     40616
#            4       0.89      0.97      0.93     48834            4       0.89      0.97      0.93     48834               4       0.89      0.97      0.93     48834
#            5       0.89      0.92      0.91     14169            5       0.89      0.92      0.91     14169               5       0.89      0.92      0.91     14169
#            6       0.91      0.63      0.75     13848            6       0.91      0.63      0.75     13848               6       0.91      0.63      0.75     13848
#            7       0.67      0.24      0.36       147            7       0.67      0.24      0.36       147               7       0.67      0.24      0.36       147
#            8       1.00      0.54      0.70        59            8       1.00      0.54      0.70        59               8       1.00      0.54      0.70        59
#            9       1.00      0.98      0.99       329            9       1.00      0.98      0.99       329               9       1.00      0.98      0.99       329

#     accuracy                           0.95    338997      accuracy                           0.95    338997      accuracy                           0.95    338997
#    macro avg       0.92      0.81      0.85    338997     macro avg       0.92      0.81      0.85    338997     macro avg       0.92      0.81      0.85    338997
# weighted avg       0.95      0.95      0.95    338997  weighted avg       0.95      0.95      0.95    338997  weighted avg       0.95      0.95      0.95    338997


## sono tutti e tre uguali, prova con più dati di test:

# xgb.Booster                                                       xgb.Classifier                                              xgb.XGBRFClassifier

#               precision    recall  f1-score   support               precision    recall  f1-score   support                      precision    recall  f1-score   support

#            0       0.95      0.99      0.97   1135427             0       0.95      0.99      0.97   1135427               0       0.95      0.99      0.97   1135427
#            1       1.00      0.95      0.97   1827512             1       1.00      0.95      0.97   1827512               1       1.00      0.95      0.97   1827512
#            2       0.92      0.92      0.92    345623             2       0.92      0.92      0.92    345623               2       0.92      0.92      0.92    345623
#            3       0.96      0.97      0.96    607607             3       0.96      0.97      0.96    607607               3       0.96      0.97      0.96    607607
#            4       0.89      0.97      0.93    736373             4       0.89      0.97      0.93    736373               4       0.89      0.97      0.93    736373
#            5       0.89      0.92      0.91    214010             5       0.89      0.92      0.91    214010               5       0.89      0.92      0.91    214010
#            6       0.91      0.64      0.75    205649             6       0.91      0.64      0.75    205649               6       0.91      0.64      0.75    205649
#            7       0.72      0.24      0.36      2227             7       0.72      0.24      0.36      2227               7       0.72      0.24      0.36      2227
#            8       0.97      0.45      0.61      1009             8       0.97      0.45      0.61      1009               8       0.97      0.45      0.61      1009
#            9       1.00      0.98      0.99      4987             9       1.00      0.98      0.99      4987               9       1.00      0.98      0.99      4987

#     accuracy                           0.95   5080424      accuracy                           0.95   5080424        accuracy                           0.95   5080424
#    macro avg       0.92      0.80      0.84   5080424     macro avg       0.92      0.80      0.84   5080424       macro avg       0.92      0.80      0.84   5080424
# weighted avg       0.95      0.95      0.95   5080424  weighted avg       0.95      0.95      0.95   5080424    weighted avg       0.95      0.95      0.95   5080424


## risultano ancora tutte e tre uguali, ora controllo stuttura interna

# if (trees_xgb == trees_XGBRF):
#     print("true")
# else: 
#     print("flase")






























