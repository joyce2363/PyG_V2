import numpy as np
import csv
from pygdebias.debiasing import NIFTY
from pygdebias.datasets import Income, Pokec_z, Pokec_n, Bail, Nba
import optuna

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="pokec_n", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--seed', type=str, default="1")
args = parser.parse_args()

if args.dataset == "pokec_n": 
    pokec_n = Pokec_n(args.seed)
    adj, features, idx_train, idx_val, idx_test, labels, sens = (
        pokec_n.adj(),
        pokec_n.features(),
        pokec_n.idx_train(),
        pokec_n.idx_val(),
        pokec_n.idx_test(),
        pokec_n.labels(),
        pokec_n.sens(),
    )
elif args.dataset == "pokec_z": 
    pokec_z = Pokec_z(args.seed)
    adj, features, idx_train, idx_val, idx_test, labels, sens = (
        pokec_z.adj(),
        pokec_z.features(),
        pokec_z.idx_train(),
        pokec_z.idx_val(),
        pokec_z.idx_test(),
        pokec_z.labels(),
        pokec_z.sens(),
    )
elif args.dataset == "nba": 
    nba = Nba(args.seed)
    adj, features, idx_train, idx_val, idx_test, labels, sens = (
        nba.adj(),
        nba.features(),
        nba.idx_train(),
        nba.idx_val(),
        nba.idx_test(),
        nba.labels(),
        nba.sens(),
    )
elif args.dataset == "income": 
    income = Income(args.seed)
    adj, features, idx_train, idx_val, idx_test, labels, sens = (
        income.adj(),
        income.features(),
        income.idx_train(),
        income.idx_val(),
        income.idx_test(),
        income.labels(),
        income.sens(),
    )
elif args.dataset == "bail": 
    bail = Bail(args.seed)
    adj, features, idx_train, idx_val, idx_test, labels, sens = (
        bail.adj(),
        bail.features(),
        bail.idx_train(),
        bail.idx_val(),
        bail.idx_test(),
        bail.labels(),
        bail.sens(),
    )

def objective(trial):
    num_hidden = trial.suggest_categorical("num_hidden", [4, 16, 64, 128, 256])
    num_proj_hidden = trial.suggest_categorical("num_proj_hidden", [4, 16, 64, 128, 256])
    lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001, 0.00001])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    sim_coeff = trial.suggest_categorical("sim_coeff", [0.3, 0.5, 0.7])

    model = NIFTY(
                    adj = adj, 
                    features = features,
                    labels = labels,
                    idx_train = idx_train, 
                    idx_val = idx_val, 
                    idx_test = idx_test, 
                    sens = sens, 
                    sens_idx = -1,
                    num_hidden = num_hidden,
                    num_proj_hidden = num_proj_hidden,
                    lr = lr,
                    weight_decay = weight_decay,
                    sim_coeff = sim_coeff, 
                    )
    model.fit()
    # Evaluate the model.

    (
        ACC,
        # AUCROC,
        F1,
        recall, 
        precision,
        cm,
        ACC_sens0,
        # AUCROC_sens0,
        F1_sens0,
        ACC_sens1,
        # AUCROC_sens1,
        F1_sens1,
        SP,
        EO,
    ) = model.predict()
    print("THIS IS: ", str(args.dataset))
    return ACC
# Create an Optuna study object and specify the optimization direction
study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=100)  # Run 100 trials
best_params = study.best_params
best_value = study.best_value


# Get the best trial
best_trial = study.best_trial

# Print the best trial number and value
print(f"Best trial number: {best_trial.number}")
print(f"Best value: {best_trial.value}")

# Get the best parameters
best_params = best_trial.params
best_params["dataset: "] = args.dataset
best_params["seed: "] = args.seed
best_params["acc: "] = best_trial.value
best_params["model: "] = "NiftyGCN_RETUNE"
# Print the best parameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

filename = 'hyperparameter_nifty_gcn_' + '.csv'

# Writing data to CSV
with open(filename, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=best_params.keys())
    writer.writerow(best_params)

print(f"Data has been written to {filename}")