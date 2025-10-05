import json
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional, Set, Union

import git
import torch
import tyro
from tqdm import trange
from tyro.conf import arg, subcommand

from src import loggers
from src.architectures import CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet
from src.datasets import CIFAR10, SST2, Sorting, Copying, Moons, Circles, Classification #,#SparseParity, FlattenedMNIST
from src.functional import FunctionalModel
from src.loss_function import SupervisedLossFunction
from src.processes import (
    CentralFlow,
    CentralFlowConfig,
    DiscreteProcess,
    DiscreteProcessConfig,
    MidpointProcess,
    MidpointProcessConfig,
    StableFlow,
    StableFlowConfig,
    EigConfig
)
from src.saving import Checkpointer, DataSaver, LoadOptions
from src.update_rules import GradientDescent, RMSProp, ScalarRMSProp
from src.utils import convert_dataclasses

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)

ValidOpt = Union[
    Annotated[GradientDescent, subcommand("gd")],
    Annotated[ScalarRMSProp, subcommand("scalar_rmsprop")],
    Annotated[RMSProp, subcommand("rmsprop")],
]
ValidData = Union[CIFAR10, SST2, Sorting, Copying, Moons, Circles, Classification]#, FlattenedMNIST, SparseParity]
ValidArch = Union[CNN, MLP, VIT, LSTM, Mamba, Transformer, Resnet]
ValidRuns = Set[Literal["discrete", "midpoint", "central", "stable", "stationary"]]

def main(
    opt: ValidOpt,    # which optimization algorithm to use
    data: ValidData,  # which dataset to train on
    arch: ValidArch,  # which architecture to train
    runs: ValidRuns,  # which processes (e.g. discrete alg, central flow, stable flow) to run
    eig_config: Annotated[EigConfig, arg(name="eig")],                        # global config for eigenvalues
    discrete_config: Annotated[DiscreteProcessConfig, arg(name="discrete")],  # config for discrete process
    midpoint_config: Annotated[MidpointProcessConfig, arg(name="midpoint")],  # config for midpoint process
    stable_config: Annotated[StableFlowConfig, arg(name="stable")],           # config for stable flow
    central_config: Annotated[CentralFlowConfig, arg(name="central")],        # config for central flow
    checkpointer: Annotated[Checkpointer, arg(name="checkpoint")],            # settings for saving checkpoints
    load: Annotated[LoadOptions, arg(name="load")],                           # optionally, settings for loading checkoints
    warm_start: int = -1,           # how many steps to warm-start for
    epochs: int = 10,               # how many epochs to train for
    batch_size: int = 128,          # mini-batch size for training
    compare_full_vs_mini: bool = True,  # whether to compare full-batch and mini-batch
    device: str = "cuda",           # cuda or cpu
    seed: int = 0,                  # random seed
    expid: Optional[str] = None,    # optionally, an experiment id (defaults to a random UUID)
):
    print("loaded cli")
    
    # collect configs that were passed in
    config = convert_dataclasses(locals())
    config["git_hash"] = git.Repo(".").git.rev_parse("HEAD")
    config["cmd"] = " ".join(sys.argv)
    
    # experiment id defaults to random uuid
    expid = expid or uuid.uuid4().hex
    
    # create experiment folder
    folder = _create_experiment_folder(expid)
    print("Saving data to: ", folder, flush=True)
    
    # dump configs to json file
    with open(folder / "config.json", "w") as config_file:
        json.dump(config, config_file, indent=4)
        
    # initialize checkpointer
    checkpointer.init(folder / "checkpoints")

    # set random seed
    torch.manual_seed(seed)
    
    # load the dataset with batch_size
    print("Loading Data")
    data.train_batch_size = batch_size  # set batch size for dataset
    dataset = data.load(device=device)
    
    # instantiate the model as a PyTorch module
    model = arch.create(dataset.input_shape, dataset.output_dim).to(device)
    
    # make the model functional. 'model_fn' is a functional version of the network; 'w' are the initial weights
    w, model_fn = FunctionalModel.make_functional(model)
    
    # initialize optimizer state
    state = opt.initialize_state(w) 
    
    # put together loss functions
    full_loss_fn = SupervisedLossFunction(
        model_fn=model_fn, criterion=dataset.criterion_fn, batches=dataset.trainset
    )

    print(f"Dataset contains {len(dataset.trainset)} full-batches and {len(dataset.trainset_batches)} mini-batches")
    
    # these are loggers that are called on each individual process 
    process_loggers = {
        k: [
            loggers.LossAndAccuracy(model_fn, dataset, split="train"),   # log train loss/acc
            loggers.LossAndAccuracy(model_fn, dataset, split="test"),    # log test loss/acc
            loggers.OutputLogger(model_fn, dataset),                     # log network output
            loggers.EigLogger(),                                         # log top eigenvalues of effective Hessian
            loggers.RawEigLogger(),                                      # log top eigenvalues of "raw" Hessian
            loggers.GradientLogger(),                                    # log gradient sq entries and sq norm
            loggers.StateLogger(),                                       # log optimizer state
            loggers.CentralFlowPredictionLogger(),                       # log central flow predictions for time-averages
        ]
        for k in runs
    }
    
    # these are loggers that are called collectively on the set of all processes
    group_loggers = [
        loggers.DistanceLogger(),        # log distance between each pair of processes
        loggers.EmpiricalDeltaLogger(),  # log delta between the central flow and the other processes
    ]                                    # along the central flow's top eigenvectors
    
    initial_step = 0

    # warm start if appropriate
    if warm_start > 0:
        if load.path is not None:
            print("Skipping warm start because checkpoint file is provided")
        else:
            print(f"Warm starting for {warm_start} steps")
            for _ in trange(warm_start):
                w, state = opt.update(w, state, full_loss_fn.D(w))
            initial_step = warm_start


    # initialize processes
    processes = {}

    # 根据参数决定创建哪些进程
    active_runs = set()
    if compare_full_vs_mini:
        # 同时运行full和mini版本
        if "discrete" in runs:
            active_runs.update(["discrete_full", "discrete_mini"])
            # 创建两个版本独立的进程，都有自己的参数和状态副本
            # 它们从同样的初始点开始，但会沿着不同的轨迹演化
            kwargs_full = dict(loss_fn=full_loss_fn, w=w.clone(), state=state.clone(),
                              opt=opt, eig_config=eig_config)
            kwargs_mini = dict(loss_fn=full_loss_fn, w=w.clone(), state=state.clone(),
                              opt=opt, eig_config=eig_config)
            processes["discrete_full"] = DiscreteProcess(**kwargs_full, config=discrete_config)
            processes["discrete_mini"] = DiscreteProcess(**kwargs_mini, config=discrete_config)
        # 其他进程类型暂不支持full vs mini对比
        if "central" in runs:
            active_runs.add("central")
            kwargs = dict(loss_fn=full_loss_fn, w=w.clone(), state=state.clone(),
                         opt=opt, eig_config=eig_config)
            processes["central"] = CentralFlow(**kwargs, config=central_config)
    else:
        # 维持原版运行方式
        active_runs = runs
        kwargs = dict(loss_fn=full_loss_fn, w=w, state=state, opt=opt, eig_config=eig_config)
        if "discrete" in runs:
            processes["discrete"] = DiscreteProcess(**kwargs, config=discrete_config)
        if "midpoint" in runs:
            processes["midpoint"] = MidpointProcess(**kwargs, config=midpoint_config)
        if "central" in runs:
            processes["central"] = CentralFlow(**kwargs, config=central_config)
        if "stable" in runs:
            processes["stable"] = StableFlow(**kwargs, config=stable_config)

    # load from checkpoint, if appropriate
    if load.path is not None:
        print(f"Loading Checkpoint from {load.path}")
        initial_step = checkpointer.load(load, processes)

    # main training loop - compare full-batch and mini-batch
    print(f"Running training for {epochs} epochs with batch_size={batch_size}")
    print(f"Comparing full-batch vs mini-batch: {compare_full_vs_mini}")
    print(f"Hessian will be computed after each parameter update using full-batch data")

    # Calculate total steps needed: all processes save at batch frequency
    num_processes_saving = 0
    if compare_full_vs_mini and "discrete" in runs:
        num_processes_saving += 2  # discrete_full and discrete_mini
    if "central" in processes:  # central also saves at batch frequency for comparison
        num_processes_saving += 1
    total_steps = epochs * len(dataset.trainset_batches) * num_processes_saving

    with DataSaver(
        folder / "data.hdf5", initial_step=0, total_steps=total_steps
    ) as data_saver:

        total_step_counter = 0

        for epoch in trange(epochs):
            train_batches = list(dataset.trainset_batches)

            # shuffle mini-batches for each epoch
            import random
            random.shuffle(train_batches)

            # ================== mini-batch training ==================
            if compare_full_vs_mini and "discrete_mini" in processes:
                process = processes["discrete_mini"]

                # mini-batch iterations within this epoch
                for batch_idx, batch in enumerate(train_batches):
                    step_data = defaultdict(lambda: {})

                    # create batch-specific loss function for gradient
                    batch_loss_fn = SupervisedLossFunction(
                        model_fn=model_fn,
                        criterion=dataset.criterion_fn,
                        batches=[batch]
                    )

                    # mini-batch update: compute gradient on current batch
                    process.loss_fn = batch_loss_fn
                    process.prepare()  # compute gradient on this batch, but NOT Hessian
                    process.step()     # update parameters using batch gradient

                    # immediately compute Hessian at new parameter location using full-batch
                    process.loss_fn = full_loss_fn
                    process.prepare()  # compute Hessian at current location

                    # collect data for this step
                    step_data["discrete_mini"].update({
                        "loss": process.loss,
                        "grad_norm": process.gradient.norm().item() if process.gradient is not None else 0.0,
                    })
                    if process.eff_eigs is not None:
                        step_data["discrete_mini"]["hessian_eigs"] = process.eff_eigs

                    # save step-level data
                    data_saver.save(total_step_counter, step_data)
                    total_step_counter += 1

            # ================== full-batch training ==================
            if compare_full_vs_mini and "discrete_full" in processes:
                process = processes["discrete_full"]

                # full-batch version also updates for every batch (matching frequency)
                # but always uses the full dataset gradient
                for batch_idx, _ in enumerate(train_batches):
                    step_data = defaultdict(lambda: {})

                    # always use full dataset for gradient calculation
                    process.loss_fn = full_loss_fn
                    process.prepare()  # compute full-batch gradient and Hessian
                    process.step()     # update using full-batch gradient

                    # collect data for this step
                    step_data["discrete_full"].update({
                        "loss": process.loss,
                        "grad_norm": process.gradient.norm().item() if process.gradient is not None else 0.0,
                    })
                    if process.eff_eigs is not None:
                        step_data["discrete_full"]["hessian_eigs"] = process.eff_eigs

                    # save full-batch step data
                    data_saver.save(total_step_counter, step_data)
                    total_step_counter += 1

            # ================== central flow training ==================
            if compare_full_vs_mini and "central" in processes:
                process = processes["central"]

                # central flow updates at the same batch frequency for fair comparison
                for batch_idx, _ in enumerate(train_batches):
                    step_data = defaultdict(lambda: {})

                    # central flow uses full-batch data (as continuous flow)
                    process.loss_fn = full_loss_fn
                    process.prepare()  # compute at current location
                    process.step()     # take central flow step

                    # collect data for this step
                    step_data["central"].update({
                        "loss": process.loss,
                        "grad_norm": process.gradient.norm().item() if process.gradient is not None else 0.0,
                    })
                    if process.eff_eigs is not None:
                        step_data["central"]["hessian_eigs"] = process.eff_eigs

                    # save central flow data
                    data_saver.save(total_step_counter, step_data)
                    total_step_counter += 1

            # 每个epoch仍然报告进度
            if compare_full_vs_mini and epoch % 10 == 0:
                if "discrete_full" in processes and "discrete_mini" in processes:
                    w_full = processes["discrete_full"].w
                    w_mini = processes["discrete_mini"].w
                    loss_full = full_loss_fn(w_full)
                    loss_mini = full_loss_fn(w_mini)
                    param_diff = (w_full - w_mini).norm().item()
                    print(f"Epoch {epoch}: diff={param_diff:.4f}, loss_full={loss_full:.4f}, loss_mini={loss_mini:.4f}")


def _create_experiment_folder(expid: int) -> Path:
    """Create the folder where saved data and checkpoints will be stored."""
    experiment_dir = Path(os.environ.get("EXPERIMENT_DIR", "experiments"))
    folder = experiment_dir / expid
    if folder.exists():
        override = input(
            f"Directory {folder} already exists. Do you want to override it? (y/N): "
        ).lower()
        if override == 'y':
            shutil.rmtree(folder)
        else:
            raise ValueError(f"Directory {folder} already exists")
    folder.mkdir(parents=True)
    return folder

if __name__ == "__main__":
    args = tyro.cli(main, config=[tyro.conf.ConsolidateSubcommandArgs])
