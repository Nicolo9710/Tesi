# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""

import xgboost as xgb
from dataclasses import asdict
import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class Server:
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None
        self.calibrator_parameters : Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        flag = True
        log(INFO, "Initializing global parameters")
        while flag:
            if(len(self._client_manager.all()) == 10):
                flag = False
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        parameters_dict = asdict(self.parameters)
        bst = xgb.Booster(params=parameters_dict)
        for para in self.parameters.tensors:
            para_b = bytearray(para)

        # Load global model
        # bst.load_model(para_b)
        # Save global model
        # bst.save_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/MachineLearningCSV/modelli/06/model_federato0.json')
        # bst.save_model('C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/alpha 08/model_federato4.json')
        # PATH = 'C:/Users/lukyl/OneDrive/Desktop/dataset_tesi_CIC_IDS2017/NF_ToN_IoT_V2/modelli/prova_grafici/federated_calib_model.pth'
        # param_a, param_b = self.fit_round_calibrator()
        # federated_model = MyCalibrator(bst, input_size=10, output_size=10)
        # federated_model.set_parameters(param_a, param_b)
        # print("parametri a: ", param_a)
        # print("parametri b: ", param_b)
        # torch.save(federated_model.calibrator.state_dict(), PATH)
        
        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    # def configure_fit(server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        
    #     return 
        
    
    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters

    def fit_round_calibrator(self):
        """Perform a single round of federated averaging."""
        print("fit_round_calibrator")
        client_instructions = self.strategy.configure_fit(
            server_round= 6,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        results, failures= fit_clients_calibrator(self, self._client_manager, client_instructions=client_instructions)
        log(
            DEBUG,
            "fit_clients_calibrator()",
        )
        # # Aggregate calibration parameters
        a_params_list = np.zeros(10)
        b_params_list = np.zeros(10)

        for client, res in results:
            param = res.parameters.tensors[0]
            str_list = param.decode('utf-8').split(';')
            # Separazione delle stringhe nei due tensori
            tensor_bytes_a = [float(item) for item in str_list[0].split()]
            tensor_bytes_b = [float(item) for item in str_list[1].split()]
            # print("tensor_a", tensor_bytes_a)
            # print("tensor_b",tensor_bytes_b)

            a_params_list += tensor_bytes_a
            b_params_list += tensor_bytes_b

        # num_clients = self.strategy.min_fit_clients
        # a_params_list = a_params_list / num_clients
        # b_params_list = b_params_list / num_clients   
        
        return a_params_list, b_params_list


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res

   

def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


###############################

def fit_clients_calibrator(self,
    client_manager: ClientManager, client_instructions: List[Tuple[ClientProxy, FitIns]]) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
     # Sample clients
    # sample_size = 10
    # clients = client_manager.sample(
    #     num_clients=sample_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        submitted_fs = {
            executor.submit(fit_client_calibrator, client_proxy, ins, timeout = None)
            for client_proxy, ins in client_instructions

        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client_calibrator(client: ClientProxy, ins: FitIns, timeout: Optional[float]) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    # print("client: ", client)
    fit_res = client.fit(ins, timeout)
    
    return client, fit_res


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
    def __init__(self, base_model, input_size, output_size, lr=0.2, epochs=100):
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