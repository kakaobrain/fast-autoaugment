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
from collections import defaultdict
import pdb
import random

from FastAutoAugment.common.config import Config


def launch_experiment(ws, conf_aml, conf_cluster, conf_docker, conf_experiment):

    # Register the input data blob container
    input_ds = Datastore.register_azure_blob_container(workspace=ws,
                                                       datastore_name='petridishdata',
                                                       container_name='datasets',
                                                       account_name='petridishdata',
                                                       account_key=conf_aml['azure_storage_account_key'],
                                                       create_if_not_exists=False)

    output_ds = Datastore.register_azure_blob_container(workspace=ws,
                                                       datastore_name='petridishoutput',
                                                       container_name='amloutput',
                                                       account_name='petridishdata',
                                                       account_key=conf_aml['azure_storage_account_key'],
                                                       create_if_not_exists=False)


    # Create or attach compute cluster
    # cluster_name = conf_cluster['cluster_name'] + datetime.datetime.now().strftime('%Y%m%d%I%M')
    cluster_name = conf_cluster['cluster_name']

    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target.')
    except:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(vm_size=conf_cluster['vm_size'], max_nodes=conf_cluster['max_nodes'],
                                                               vm_priority=conf_cluster['vm_priority'],
                                                               idle_seconds_before_scaledown=conf_cluster['idle_seconds_before_scaledown'])

        # Create the cluster
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # use get_status() to get a detailed status for the current cluster.
    print(compute_target.get_status().serialize())

    # Set project directory
    # Assuming running in extract_features_from_videos folder
    project_folder = '../'

    # Setup custom docker usage
    image_registry_details = ContainerRegistry()
    image_registry_details.address = conf_docker['image_registry_address']
    image_registry_details.username = conf_docker['image_registry_username']
    image_registry_details.password = conf_docker['image_registry_password']

    # don't let the system build a new conda environment
    user_managed_dependencies = True

    # Note that experiment names have to be
    # <36 alphanumeric characters
    #exp_name = conf_experiment['experiment_name'] + datetime.datetime.now().strftime('%Y%m%d%I%M')
    exp_name = conf_experiment['experiment_name']

    experiment = Experiment(ws, name=exp_name)
    script_params = {'--dataset.dataroot': input_ds.path('/').as_mount(),
                     '--common.logdir': output_ds.path('/').as_mount(),
                    }

    est = Estimator(source_directory=project_folder,
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script='scripts/random/cifar_eval.py',
                    custom_docker_image=conf_docker['image_name'],
                    image_registry_details=image_registry_details,
                    user_managed=user_managed_dependencies,
                    source_directory_data_store=input_ds)

    run = experiment.submit(est)


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


if __name__ == "__main__":
    print("SDK Version:", azureml.core.VERSION)
    set_diagnostics_collection(send_diagnostics=True)

    # Read in config
    conf = Config(config_filepath='~/aml_secrets/aml_config_dedey.yaml')

     # Config region
    conf_aml = conf['aml_config']
    conf_cluster = conf['cluster_config']
    conf_docker = conf['azure_docker']
    conf_experiment = conf['experiment']
    # endregion


    # Initialize workspace
    # Make sure you have downloaded your workspace config
    ws = Workspace.from_config(path=conf_aml['aml_config_file'])
    print('Workspace name: ' + ws.name,
          'Azure region: ' + ws.location,
          'Subscription id: ' + ws.subscription_id,
          'Resource group: ' + ws.resource_group, sep='\n')

    launch_experiment(ws, conf_aml, conf_cluster, conf_docker, conf_experiment)