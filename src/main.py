"""SGCN runner."""

from sgcn_G import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs
import os
import torch
import numpy as np
import random
import optuna
import pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def main(trial=None):
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()
    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)


    tab_printer(args)
    edges = read_graph(args)
    trainer = SignedGCNTrainer(args, edges)
    trainer.setup_dataset()
    trainer.create_and_train_model(trial)

    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

    if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return trainer.logs["performance"][-1][2] #+ trainer.logs["performance"][-1][2]

if __name__ == "__main__":
    auto_ml = True

    if auto_ml:
        study = optuna.create_study(direction="maximize")
        study.optimize(main, n_trials=1)
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print(" Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        pickle.dump(study, open("params_bitcoin_alpha_F1.pkl", "wb"))
    else:
        main()

