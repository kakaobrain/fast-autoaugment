import os
import time
import itertools
import datetime

import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core import Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment
from azureml.core.container_registry import ContainerRegistry
from azureml.train.estimator import Estimator
from azureml.core import Environment

import azure
from azure.storage.blob import BlockBlobService

from tqdm import tqdm


def get_experiment(ws, exp_name):
    experiment_exists = ws.experiments.get(exp_name, None)
    if experiment_exists is not None:
        return experiment_exists
    else:
        print('Unable to find an experiment in the current workspace with name {}'.format(exp_name))

def cancel_experiment(ws, experiment_name):
    exp = get_experiment(ws, experiment_name)
    print('Cancelling existing experiment with name: {}'.format(experiment_name))
    for run in tqdm(list(exp.get_runs())):
        run.cancel()

def get_workspace(config_file):
    ws = Workspace.from_config(config_file)
    print('Workspace name: ' + ws.name,
          'Azure region: ' + ws.location,
          'Subscription id: ' + ws.subscription_id,
          'Resource group: ' + ws.resource_group, sep='\n')


def download_results(ws, exp_name, out_path):
    # # Default datastore
    # datastore = ws.get_default_datastore()
    # print('Datastore:', datastore.datastore_type, datastore.account_name, datastore.container_name)
    # datastore.download(out_path, prefix='09-06/', overwrite=False, show_progress=True)

    exp = get_experiment(ws, exp_name)
    print('Pulling logs for experiment {} and storing locally in {}'.format(exp_name, out_path))
    for run in tqdm(list(exp.get_runs())):
        logs = run.get_all_logs(destination=out_path)
        print(run, logs)