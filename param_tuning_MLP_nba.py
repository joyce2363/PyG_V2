from pygdebias.debiasing import GNN
from pygdebias.datasets import Nba
import optuna
# Available choices: 'Credit', 'German', 'Facebook', 'Pokec_z', 'Pokec_n', 'Nba', 'Twitter', 'Google', 'LCC', 'LCC_small', 'Cora', 'Citeseer', 'Amazon', 'Yelp', 'Epinion', 'Ciao', 'Dblp', 'Filmtrust', 'Lastfm', 'Ml-100k', 'Ml-1m', 'Ml-20m', 'Oklahoma', 'UNC', 'Bail'.
    
nba = Nba()
adj, features, idx_train, idx_val, idx_test, labels, sens = (
    nba.adj(),
    nba.features(),
    nba.idx_train(),
    nba.idx_val(),
    nba.idx_test(),
    nba.labels(),
    nba.sens(),
)

def objective(trial):
    # Define the hyperparameter search space
    num_hidden = trial.suggest_int("num_hidden", 4, 130)
    num_proj_hidden = trial.suggest_int("num_proj_hidden", 4, 130)
    lr = trial.suggest_float("lr", 0.00001, 0.01)
    weight_decay = trial.suggest_float("weight_decay", 0.000001, 0.001)
    sim_coeff = trial.suggest_float("sim_coeff", 0.3, 0.7)
    # lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    # Create GNN model with suggested hyperparameters
    model = GNN(adj, 
                features, 
                labels, 
                idx_train, 
                idx_val, 
                idx_test, 
                sens, 
                idx_train,
                num_hidden = num_hidden, 
                num_proj_hidden = num_proj_hidden,
                lr = lr,
                weight_decay = weight_decay, 
                sim_coeff = sim_coeff
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
    print("THIS IS NBA! for MLP :D ")
    return ACC
# , F1, recall, precision, cm, ACC_sens0, F1_sens0, ACC_sens1, F1_sens1, SP, EO


# print("idx_train: ", len(idx_train))
# print("idx_val: ", len(idx_val))
# print("idx_test: ", len(idx_test))

# print("len of labels: ", len(labels))
# model = GNN(adj, 
#             features, 
#             labels, 
#             idx_train, 
#             idx_val, 
#             idx_test, 
#             sens, 
#             idx_train) #features.shape[1]

# # Train the model.
# model.fit()

# # Evaluate the model.

# (
#     ACC,
#     # AUCROC,
#     F1,
#     recall, 
#     precision,
#     cm,
#     ACC_sens0,
#     # AUCROC_sens0,
#     F1_sens0,
#     ACC_sens1,
#     # AUCROC_sens1,
#     F1_sens1,
#     SP,
#     EO,
# ) = model.predict()

# print("ACC:", ACC)
# # print("AUCROC: ", AUCROC)
# print("F1: ", F1)
# print("recall: ", recall)
# print("precision: ", precision)
# print("confusion_matrix: ", cm)
# print("ACC_sens0:", ACC_sens0)
# # print("AUCROC_sens0: ", AUCROC_sens0)
# print("F1_sens0: ", F1_sens0)
# print("ACC_sens1: ", ACC_sens1)
# # print("AUCROC_sens1: ", AUCROC_sens1)
# print("F1_sens1: ", F1_sens1)
# print("SP: ", SP)
# print("EO:", EO)

# Create an Optuna study object and specify the optimization direction
study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=100)  # Run 100 trials

# Get the best hyperparameters and their value
best_params = study.best_params
best_value = study.best_value


# Get the best trial
best_trial = study.best_trial

# Print the best trial number and value
print(f"Best trial number: {best_trial.number}")
print(f"Best value: {best_trial.value}")

# Get the best parameters
best_params = best_trial.params

# Print the best parameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")