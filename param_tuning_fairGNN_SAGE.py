from pygdebias.debiasing import FairGNN_ALL
from pygdebias.datasets import Pokec_n, Pokec_z, Nba, Income, Bail
import optuna
import csv

# Available choices: 'Credit', 'German', 'Facebook', 'Pokec_z', 'Pokec_n', 'Nba', 'Twitter', 'Google', 'LCC', 'LCC_small', 'Cora', 'Citeseer', 'Amazon', 'Yelp', 'Epinion', 'Ciao', 'Dblp', 'Filmtrust', 'Lastfm', 'Ml-100k', 'Ml-1m', 'Ml-20m', 'Oklahoma', 'UNC', 'Bail'.
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="nba", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--seed', type=str, default="1")
parser.add_argument('model', type=str, default='gcn')
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
    # Define the hyperparameter search space
    num_hidden = trial.suggest_categorical("num_hidden", [16, 64, 128, 256])
    sim_coeff = trial.suggest_categorical("sim_coeff", [0.3, 0.5, 0.6, 0.7])
    acc = trial.suggest_categorical("acc", [0.2, 0.3, 0.4, 0.5, 0.6, 0.69, 0.7])
    alpha = trial.suggest_categorical("alpha", [1, 3, 4, 5, 6, 7, 10, 20, 40])
    beta = trial.suggest_categorical("beta", [0.1, 0.01, 0.001, 0.0001])
    proj_hidden = trial.suggest_categorical("proj_hidden", [4, 8, 16, 64, 128])
    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4, 1e-5])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-2, 1e-3, 1e-4, 1e-5])

    # Create GNN model with suggested hyperparameters
    model = FairGNN_ALL(
                    adj, 
                    features, 
                    labels, 
                    idx_train, 
                    idx_val, 
                    idx_test, 
                    sens, 
                    nfeat=features.shape[1],
                    num_hidden = num_hidden,
                    alpha = alpha,
                    beta = beta,
                    acc = acc,
                    sim_coeff = sim_coeff, 
                    lr = lr, 
                    weight_decay = weight_decay, 
                    proj_hidden = proj_hidden,
                    )
    model.fit(adj, features, labels, idx_train, idx_val, idx_test, sens, idx_train)
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
    ) = model.predict(idx_test)
    print("THIS IS: ", str(args.dataset))
    return ACC
# Create an Optuna study object and specify the optimization direction
study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=1)  # Run 100 trials
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
best_params["model: "] = "fairGNN_SAGE"
# Print the best parameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

if args.dataset == "pokec_z": 
    filename = 'hyperparameter' + str(args.dataset) + '.csv'
elif args.dataset == "pokec_n":
    filename = 'hyperparameter.csv'
elif args.dataset == "nba": 
    filename = 'hyperparameter' + str(args.dataset) + '.csv'
elif args.dataset == "income": 
    filename = 'hyperparameter' + str(args.dataset) + '.csv'
elif args.dataset == "bail": 
    filename = 'hyperparameter' + str(args.dataset) + '.csv'



# Writing data to CSV
with open(filename, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=best_params.keys())
    writer.writerow(best_params)

print(f"Data has been written to {filename}")