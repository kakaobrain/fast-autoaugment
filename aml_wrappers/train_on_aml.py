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


def launch_experiment(ws):

    # Register the input data blob container
    input_ds = Datastore.register_azure_blob_container(workspace=ws,
                                                       datastore_name='petridishdata',
                                                       container_name='datasets',
                                                       account_name='petridishdata',
                                                       account_key='FVSBHodZL99VkIL0Mn33X0735NYbIpdQg99I+54/OW5LZsrbSclWMuRAnjUBdqS1ylMD9NK2Gsg06XEzftuCoA==',
                                                       create_if_not_exists=False)        
    
    output_ds = Datastore.register_azure_blob_container(workspace=ws,
                                                       datastore_name='petridishoutput',
                                                       container_name='amloutput',
                                                       account_name='petridishdata',
                                                       account_key='FVSBHodZL99VkIL0Mn33X0735NYbIpdQg99I+54/OW5LZsrbSclWMuRAnjUBdqS1ylMD9NK2Gsg06XEzftuCoA==',
                                                       create_if_not_exists=False)        


    # Create or attach compute cluster
    cluster_name = "pet-" + datetime.datetime.now().strftime('%Y%m%d%I%M')

    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target.')
    except:
        print('Creating a new compute target...')
        # STANDARD_NC6 (K80), STANDARD_NC6S_V2 (P100)
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC24S_V2', max_nodes=100,
                                                               vm_priority='lowpriority',
                                                               idle_seconds_before_scaledown=120)

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
    image_registry_details.address = "dedeyfvimobd1b7f78.azurecr.io"
    image_registry_details.username = "dedeyfvimobd1b7f78"
    image_registry_details.password = "cHzHmgsqAdk4EM1GQr=zJLwWrYyNnKur"

    # don't let the system build a new conda environment
    user_managed_dependencies = True    

    # Note that experiment names have to be 
    # <36 alphanumeric characters
    exp_name = 'pet-' + datetime.datetime.now().strftime('%Y%m%d%I%M')
    
    experiment = Experiment(ws, name=exp_name)
    script_params = {'--dataroot': input_ds.path('/').as_mount(),
                     '--save': output_ds.path(exp_name).as_mount(),
                     '-c': 'FastAutoAugment/confs/darts.yaml',
                     '--aug': 'fa_reduced_cifar10',
                     '--dataset': 'cifar10'
                    }
    
    est = Estimator(source_directory=project_folder,
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script='FastAutoAugment/train.py',
                    custom_docker_image='petridishpytorch',
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

    # Initialize workspace
    # Make sure you have downloaded your workspace config
    ws = Workspace.from_config(path='dedey-fvimo-config.json')
    print('Workspace name: ' + ws.name,
          'Azure region: ' + ws.location,
          'Subscription id: ' + ws.subscription_id,
          'Resource group: ' + ws.resource_group, sep='\n')

    launch_experiment(ws)