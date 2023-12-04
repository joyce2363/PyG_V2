import numpy as np
import csv
import argparse
from pygdebias.debiasing import FairGNN
from pygdebias.datasets import Income, Pokec_z, Pokec_n, Bail, Nba

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="nba", help='One dataset from income, bail, pokec1, and pokec2.')
args = parser.parse_args()

curr_dict = {}

for i in range(1, 6): 
    seed = i 
    if seed == 1: 
        accuracy = []
        satistical_parity = []
        equal_opportunity = []
    if args.dataset == "income": 
        income = Income(seed)
        # Income.load_income(dataset = "income", seed=seed)
        adj, features, idx_train, idx_val, idx_test, labels, sens = (
            income.adj(),
            income.features(),
            income.idx_train(),
            income.idx_val(),
            income.idx_test(),
            income.labels(),
            income.sens(),
            # income.seed(seed), 
        )
        print("LOADING INCOME")
    elif args.dataset == "pokec_z":
        pokec_z = Pokec_z(seed)
        # Income.load_income(dataset = "income", seed=seed)
        adj, features, idx_train, idx_val, idx_test, labels, sens = (
            pokec_z.adj(),
            pokec_z.features(),
            pokec_z.idx_train(),
            pokec_z.idx_val(),
            pokec_z.idx_test(),
            pokec_z.labels(),
            pokec_z.sens(),
            # income.seed(seed), 
        )
        print("pokec_z")
    elif args.dataset == "pokec_n": 
        pokec_n = Pokec_n(seed)
        adj, features, idx_train, idx_val, idx_test, labels, sens = (
            pokec_n.adj(),
            pokec_n.features(),
            pokec_n.idx_train(),
            pokec_n.idx_val(),
            pokec_n.idx_test(),
            pokec_n.labels(),
            pokec_n.sens(),
            # income.seed(seed), 
        )
        print("pokec_n")
    elif args.dataset == "bail": 
        bail = Bail(seed)
        adj, features, idx_train, idx_val, idx_test, labels, sens = (
            bail.adj(),
            bail.features(),
            bail.idx_train(),
            bail.idx_val(),
            bail.idx_test(),
            bail.labels(),
            bail.sens(),
            # income.seed(seed), 
        )
        print("bail")
    elif args.dataset == "nba": 
        nba = Nba(seed)
        adj, features, idx_train, idx_val, idx_test, labels, sens = (
            nba.adj(),
            nba.features(),
            nba.idx_train(),
            nba.idx_val(),
            nba.idx_test(),
            nba.labels(),
            nba.sens(),
            # income.seed(seed), 
        )
        print("nba")
    print("idx_train: ", len(idx_train))
    print("idx_val: ", len(idx_val))
    print("idx_test: ", len(idx_test))
    print("len of labels: ", len(labels))
    # Initiate the model (with default parameters).
    model = FairGNN(features.shape[1], seed)

    # Train the model.
    model.fit(adj, features, labels, idx_train, idx_val, idx_test, sens, idx_train)

# Evaluate the model.

    (
        ACC,
        AUCROC,
        F1,
        ACC_sens0,
        AUCROC_sens0,
        F1_sens0,
        ACC_sens1,
        AUCROC_sens1,
        F1_sens1,
        SP,
        EO,
    ) = model.predict(idx_test)

    accuracy.append(ACC)
    satistical_parity.append(SP)
    equal_opportunity.append(EO)
    curr_dict['Model'] = ['FairGNN']
    if args.dataset == 'pokec_z': 
        curr_dict['Dataset'] = [str('pokec1')]
    elif args.dataset == 'pokec_n': 
        curr_dict['Dataset'] = [str('pokec2')]
    else: 
        curr_dict['Dataset'] = [str(args.dataset)]

    curr_dict['Average'] = [str(np.mean(accuracy)) + str(' += ') + str(np.var(accuracy))]
    curr_dict['Statistical Parity'] = [str(np.mean(satistical_parity)) + str(' += ') + str(np.var(satistical_parity))]
    curr_dict['Equal Opportunity'] = [str(np.mean(equal_opportunity)) + str(' += ') + str(np.var(equal_opportunity))]


    print("ACC:", ACC)
    print("AUCROC: ", AUCROC)
    print("F1: ", F1)
    print("ACC_sens0:", ACC_sens0)
    print("AUCROC_sens0: ", AUCROC_sens0)
    print("F1_sens0: ", F1_sens0)
    print("ACC_sens1: ", ACC_sens1)
    print("AUCROC_sens1: ", AUCROC_sens1)
    print("F1_sens1: ", F1_sens1)
    print("SP: ", SP)
    print("EO:", EO)

print("average for:", args.dataset , np.mean(accuracy), '+=', np.var(accuracy))
print("statistical parity:", args.dataset, np.mean(satistical_parity), '+=', np.var(satistical_parity))
print("equal Opportunity:", args.dataset, np.mean(equal_opportunity), '+=', np.var(equal_opportunity))

# data.append(curr_dict)


filename = 'output.csv'

# Writing data to CSV
with open(filename, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=curr_dict.keys())


    # Write header
    writer.writeheader()

    # Write rows
    for i in range(0,1):
        row = {key: curr_dict[key] for key in curr_dict}
        writer.writerow(row)

print(f"Data has been written to {filename}")