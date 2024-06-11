import flwr as fl
from flwr.server.strategy import FedXgbBagging
import numpy as np 
import xgboost as xgb
from datasets import Dataset, DatasetDict
from typing import Union
# from datset import tabella_dati
import sklearn
from sklearn.model_selection import train_test_split
import json
import xgboost as xgb
from flwr.common.parameter import ndarray_to_bytes
import pickle
from sklearn.ensemble import RandomForestClassifier
import datset
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from logging import ERROR, INFO, WARN
from flwr.common.logger import log
from dataclasses import asdict


#pool_size = 2
num_rounds = 5
#num_clients_per_round = 2
#num_evaluate_clients = 2

# log(
#         INFO,
#         "Creazione dataset per valutazione centralizzata",
#     )

#valid_dmatrix = xgb.DMatrix(datset.X_keep, label=datset.y_keep)


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    acc_aggregated = (
        #sum([metrics["ERR"] * num for num, metrics in eval_metrics]) / total_num
        sum([metrics["acc"] * num for num, metrics in eval_metrics]) / total_num
    )
    # prec_aggregated = (
    #     #sum([metrics["ERR"] * num for num, metrics in eval_metrics]) / total_num
    #     sum([metrics["prec"] * num for num, metrics in eval_metrics]) / total_num
    # )
    # f1_aggregated = (
    #     #sum([metrics["ERR"] * num for num, metrics in eval_metrics]) / total_num
    #     sum([metrics["f1"] * num for num, metrics in eval_metrics]) / total_num
    # )
    metrics_aggregated = {"acc": acc_aggregated}#, "prec": prec_aggregated, "f1":f1_aggregated}
    #metrics_aggregated = {"ERR": err_aggregated}
    return metrics_aggregated


# def central_eval(server_round, parameters, _): #qui aggiungo {}
#     # If at the first round, skip the evaluation
#     if server_round == 0:
#         return 0, {}
#     else:
#         # bst = xgb.Booster(parameters)
#         # for para in parameters.tensors:
#         #     para_b = bytearray(para)

#         # # Load global model
#         # bst.load_model(para_b)
#         # # Run evaluation
#         # eval_results = bst.eval_set(
#         #     evals=[(valid_dmatrix, "valid")],
#         #     iteration=bst.num_boosted_rounds() - 1,
#         # )
#         parameters_dict = asdict(parameters)
#         bst = xgb.Booster(params=parameters_dict)
#         for para in parameters.tensors:
#             para_b = bytearray(para)

#         # Load global model
#         bst.load_model(para_b)
#     pred = bst.predict(valid_dmatrix)
#     r = []
#     valid_Y = valid_dmatrix.get_label()
#     for i in pred:
#         r.append(i.argmax())
#     cl_rp = classification_report(valid_Y, r, output_dict= True)
#     accuracy = cl_rp["accuracy"]
#     loss = 0

#     return (loss, {"acc": accuracy})

# Define strategy
strategy = FedXgbBagging(
    fraction_fit=1.0,
    min_fit_clients=10,
    min_available_clients=10,
    min_evaluate_clients=10,
    fraction_evaluate=1.0,
    #evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    #evaluate_function=central_eval
    
)

# Start Flower server
hist = fl.server.start_server(
    server_address="0.0.0.0:8079",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)


