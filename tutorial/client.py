import argparse
import warnings
from typing import Union
from logging import INFO
from datasets import Dataset, DatasetDict
import xgboost as xgb

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr_datasets.partitioner import IidPartitioner

# from datset import partitions
#from datset import get_partition 
import sklearn
#from sklearn.model_selection import train_test_split
from dask_ml.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)

# # We first define arguments parser for user to specify the client/node ID.
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--node-id",
#     default=0,
#     type=int,
#     help="Node ID used for the current client.",
# )
# args = parser.parse_args()

# Define data partitioning related functions
def my_train_test_split(partition_X: Dataset, partition_y: Dataset, test_fraction: float):
    """Split the data into train and validation set given split rate."""
    # train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    # partition_train = train_test["train"]
    # partition_test = train_test["test"]
    partition_train , partition_test, partition_train_y, partition_test_y = train_test_split(partition_X, partition_y, test_size = test_fraction, shuffle= True)

    # num_train = len(partition_train)
    # num_test = len(partition_test)

    return partition_train, partition_test, partition_train_y, partition_test_y


def transform_dataset_to_dmatrix(data_x, data_y) -> xgb.core.DMatrix: #data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    # #x = data["inputs"]
    # x = data[["L4_SRC_PORT" , "L4_DST_PORT" , "PROTOCOL" , "L7_PROTO" , "IN_BYTES" , "OUT_BYTES" , "IN_PKTS" , "OUT_PKTS" , "TCP_FLAGS" , "FLOW_DURATION_MILLISECONDS"]]
    # #y = data["label"]
    # y = data["Attack"]
    new_data = xgb.DMatrix(data_x, label=data_y)
    #pandas_df = data_y.compute()
    return new_data


# num_partition = 30
# i = tabella_dati.shape[0]
# n = int(i/num_partition)
# node_id=args.node_id
# partition = tabella_dati[n*node_id : n*(node_id+1)]


# Train/test splitting
# train_data, valid_data, num_train, num_val = my_train_test_split(
#     partition, test_fraction=0.2, seed=42
# )

# node_id=args.node_id
# X, y = get_partition(node_id)
# #Train/test splitting
# train_data, valid_data, train_y, valid_y, num_train, num_val = my_train_test_split(X, y, test_fraction=0.2)

# # Reformat data to DMatrix for xgboost
# train_dmatrix, train_y= transform_dataset_to_dmatrix(train_data, train_y)
# valid_dmatrix, valid_y= transform_dataset_to_dmatrix(valid_data, valid_y)

num_local_round = 1
params = {
    #"objective": "binary:logistic",
    "objective": "multi:softprob", #"multi:softmax", #oppure "multi:softprob"
    "eta": 0.1,  # lr
    "max_depth": 8,
    # "eval_metric": "auc",
    #"eval_metric": "error",
    "eval_metric": "mlogloss",
    "nthread": 16,
    "num_class" : 10,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
    #"loss" : "mlogloss",
}


class XgbClient(fl.client.Client):
    def __init__(self, partition, node_id):
        self.bst = None
        self.config = None
        self.train_dmatrix, self.valid_dmatrix, self.num_train, self.num_valid = partition
        self.node_id = node_id

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            log(INFO, "Start training at round 1")
            #x,y = self.data
            
            bst = xgb.train(
                params,
                self.train_dmatrix,
                num_boost_round=num_local_round,
                evals=[(self.train_dmatrix, "training")],
            )
            self.config = bst.save_config()
            self.bst = bst
            
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            # valutazione modello prima dell'addestramento
            pred = self.bst.predict(self.valid_dmatrix)
            r = []
            for i in pred:
                r.append(i.argmax())
            y_true = self.valid_dmatrix.get_label()
            acc = accuracy_score(y_true, r)
            print(f"prima del fit -> client: {self.node_id}, acc: {acc}")

            # addestramento del modello
            bst = self._local_boost(self.train_dmatrix)
            
            
            eval_result = self.bst.eval_set(evals=[(self.train_dmatrix, "train")])
            a = round(float(eval_result.split("\t")[1].split(":")[1]), 4)
            print(f"training-mlogloss_client{self.node_id}: {a}")


        local_model = self.bst.save_raw("json")
        local_model_bytes = bytes(local_model)


        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples= self.num_train,
            metrics={},
        )
    def _local_boost(self,train_dmatrix):
        # Update trees based on local training data.
        for i in range(num_local_round):
            self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_round : self.bst.num_boosted_rounds()
        ]

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            #eval_results = self.bst.eval_set(
            #     evals=[(valid_dmatrix, "valid")],
            #     iteration=self.bst.num_boosted_rounds() - 1,
            # )
            # auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            #valid_dmatrix = transform_dataset_to_dmatrix(self.test_x, self.test_y)
            if not self.bst:
                 return EvaluateRes(
                status=Status(
                    code=Code.OK,
                    message="NOT_OK",
                ),
                loss=0.0,
                num_examples=0,
                # metrics={"AUC": auc},
                metrics={},
            )
            else:
                #pred = self.bst.predict(valid_dmatrix)
                # # questo stampa ancora la loss perchè chiamo evals
                # eval_results = self.bst.eval_set(
                #     evals=[(self.valid_dmatrix, "valid")])
                # a = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
                # print("a:" , a)
                pred = self.bst.predict(self.valid_dmatrix)

                # # pred restituisce un array con 10 probabilità una per classe, prendo l'indice di quella più alta che corrisponderà alla classe predetta
                r = []
                for i in pred:
                    r.append(i.argmax())
                y_true = self.valid_dmatrix.get_label()
                acc = accuracy_score(y_true, r)
                #prec = precision_score(y_true, r, average='weighted') #micro dovrebbe essere globale, macro per ogni label
                #f1 = f1_score(y_true, r, average='weighted') #come sopra
                # y_true = self.valid_dmatrix.get_label()
                # error_rate = np.sum(r != y_true) / self.num_valid
                # print('Test error using softmax = {}'.format(error_rate))
                print(f"client: {self.node_id}, acc: {acc}")#, prec: {prec}, f1: {f1}")
                return EvaluateRes(
                    status=Status(
                        code=Code.OK,
                        message="OK",
                    ),
                    
                    loss = 0.0,
                    num_examples=self.num_valid,
                    metrics={"acc": acc}#, "prec": prec, "f1": f1},
                )
            
def start(partition, node_id):
    fl.client.start_client(server_address="127.0.0.1:8079", client=XgbClient(partition, node_id))

