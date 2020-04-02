from azureml.core import Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.authentication import InteractiveLoginAuthentication
import os


# WORKSPACE
def get_workspace(
    subscription_id,
    resource_group,
    workspace_name,
    tenant_id
):
    auth = InteractiveLoginAuthentication(
        tenant_id = tenant_id,
        force = True
    )
    
    ws = Workspace(
        workspace_name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=auth
    )
    return ws


# COMPUTE TARGET
def get_compute_target(
    ws,
    ct_name,
    vm_name="STANDARD_NC6S_V3",
    min_nodes=2,
    max_nodes=2,
    vnet_rg=None,
    vnet_name=None,
    subnet_name=None,
    admin_username=None,
    ssh_key_pub=None,
):
    if ct_name not in ws.compute_targets:
        # REQUIRED PARAMS
        ct_params = [
            ("vm_size", vm_name),
            ("min_nodes", min_nodes),
            ("max_nodes", max_nodes),
        ]

        # DO WE HAVE VNET
        if vnet_rg is not None and vnet_name is not None and subnet_name is not None:
            ct_params += [
                ("vnet_resourcegroup_name", vnet_rg),
                ("vnet_name", vnet_name),
                ("subnet_name", subnet_name),
            ]

        # REQUIRED FOR LOCAL SUBMISSIONS -- SSH
        if ssh_key_pub is not None and admin_username is not None:
            with open(ssh_key_pub, "r") as f:
                ssh_key_pub = f.read().strip()

            ct_params += [
                ("admin_username", admin_username),
                ("admin_user_ssh_key", ssh_key_pub),
                ("remote_login_port_public_access", "Enabled"),
            ]

        # create config for Azure ML cluster
        config = AmlCompute.provisioning_configuration(**dict(ct_params))
        ct = ComputeTarget.create(ws, ct_name, config)
        ct.wait_for_completion(show_output=True)
    else:
        ct = ws.compute_targets[ct_name]

    return ct


# ENVIRONMENT
def get_environment(
    ws,
    environment_name,
    docker_image="todrabas/aml_rapids:latest",
    python_interpreter="/opt/conda/envs/rapids/bin/python",
    conda_packages=["matplotlib"],
):
    if environment_name not in ws.environments:
        env = Environment(name=environment_name)
        env.docker.enabled = True
        env.docker.base_image = docker_image

        env.python.interpreter_path = python_interpreter
        env.python.user_managed_dependencies = True

        conda_dep = CondaDependencies()

        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)

        env.python.conda_dependencies = conda_dep
        env.register(workspace=ws)
    else:
        env = ws.environments[environment_name]

    return env


# DATA UPLOAD
def download_and_upload_data(ws, get_data=False, years=["2016"]):
    if get_data:
        import nyc_data

        print(" Downloading data ".center(80, "#"))
        print()

        nyc_data.download_nyctaxi_data(years, os.getcwd())

        print(" Uploading data ".center(80, "#"))
        print()
        nyc_data.upload_nyctaxi_data(
            ws,
            ws.datastores["datafileshare"],
            os.path.join(os.getcwd(), "nyctaxi"),
            os.path.join("data", "nyctaxi"),
        )
