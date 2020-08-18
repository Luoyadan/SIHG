from SHIG_trainer import SHIGTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph
import os
import torch
import numpy as np
import random
import optuna
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args, trial=None):
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    tab_printer(args)
    # read data
    edges = read_graph(args)
    trainer = SHIGTrainer(args, edges)
    trainer.setup_dataset()
    # training
    trainer.create_and_train_model(trial)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    if args.metric_to_optimize is 'AUC':
        return trainer.logs["performance"][-1][1]
    elif args.metric_to_optimize is 'F1':
        return trainer.logs["performance"][-1][2]


if __name__ == "__main__":
    # use optuna to find best hyperparameters
    args = parameter_parser()

    if args.auto_ml:
        # maximize evaluation metrics
        study = optuna.create_study(direction="maximize")

        # number of trials
        study.optimize(main, args, n_trials=100)
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print(" Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        main(args)

