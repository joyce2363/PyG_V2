import numpy as np
import csv
import argparse
from pygdebias.debiasing import NIFTY_GAT
from pygdebias.datasets import Income, Pokec_z, Pokec_n, Bail, Nba

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="nba", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--num_hidden', nargs='+', type=int, default=[16])
parser.add_argument('--num_proj_hidden', nargs='+', type=int, default=[16])
parser.add_argument('--lr', nargs='+', type=float, default=[0.01])
parser.add_argument('--weight_decay', nargs='+', type=float, default=[1e-5])
parser.add_argument('--sim_coeff', nargs='+', type=float, default=[0.5])
parser.add_argument('--drop_edge_rate_1', nargs='+', type=float, default=[0.1])
parser.add_argument('--drop_edge_rate_2', nargs='+', type=float, default=[0.1])
parser.add_argument('--drop_feature_rate_1', nargs='+', type=float, default=[0.01])
parser.add_argument('--drop_feature_rate_2', nargs='+', type=float, default=[0.01])

parser.add_argument('--model', type=str, default='gat')
parser.add_argument('--encoder', type=str, default='gat')
args = parser.parse_args()

# Available choices: 'Credit', 'German', 'Facebook', 'Pokec_z', 'Pokec_n', 'Nba', 'Twitter', 'Google', 'LCC', 'LCC_small', 'Cora', 'Citeseer', 'Amazon', 'Yelp', 'Epinion', 'Ciao', 'Dblp', 'Filmtrust', 'Lastfm', 'Ml-100k', 'Ml-1m', 'Ml-20m', 'Oklahoma', 'UNC', 'Bail'.
seed = 0
curr_dict = {} 
print(zip(args.num_hidden, args.num_proj_hidden, args.lr, args.weight_decay, args.sim_coeff))
# import random
# import torch
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
for i in range (1,6): 
    seed = i 
    if seed == 1: 
        accuracy = []
        satistical_parity = []
        equal_opportunity = []
    if args.dataset == "income": 
        income = Income(seed)
        adj, features, idx_train, idx_val, idx_test, labels, sens = (
            income.adj(),
            income.features(),
            income.idx_train(),
            income.idx_val(),
            income.idx_test(),
            income.labels(),
            income.sens(),
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

    print("SEED: ", seed)
    model = NIFTY_GAT(
        adj, 
        features, 
        labels, 
        idx_train, 
        idx_val, 
        idx_test, 
        sens, 
        seed,
        num_hidden = args.num_hidden[seed-1],  
        num_proj_hidden = args.num_proj_hidden[seed-1],
        lr = args.lr[seed-1],
        weight_decay = args.weight_decay[seed-1], 
        drop_edge_rate_1 = args.drop_edge_rate_1[seed-1], 
        drop_edge_rate_2 = args.drop_edge_rate_2[seed-1], 
        drop_feature_rate_1 = args.drop_feature_rate_1[seed-1], 
        drop_feature_rate_2 = args.drop_feature_rate_2[seed-1], 
        sim_coeff = args.sim_coeff[seed-1],
        encoder = args.encoder,
    )
        # Train the model.
    model.fit(seed = i, model = args.model, data=args.dataset)

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
    ) = model.predict(seed = i, model = args.model, data=args.dataset)
    accuracy.append(ACC)
    satistical_parity.append(SP)
    equal_opportunity.append(EO)
    print("ACC:", ACC)
        # print("AUCROC: ", AUCROC)
    print("F1: ", F1)
    print("recall: ", recall)
    print("precision: ", precision)
    print("confusion_matrix: ", cm)
    print("ACC_sens0:", ACC_sens0)
        # print("AUCROC_sens0: ", AUCROC_sens0)
    print("F1_sens0: ", F1_sens0)
    print("ACC_sens1: ", ACC_sens1)
        # print("AUCROC_sens1: ", AUCROC_sens1)
    print("F1_sens1: ", F1_sens1)
    print("SP: ", SP)
    print("EO:", EO)
curr_dict['Model'] = 'NIFTY_' + str(args.model)

if args.dataset == 'pokec_z': 
    curr_dict['Dataset'] = str('pokec1')
elif args.dataset == 'pokec_n': 
    curr_dict['Dataset'] = str('pokec2')
else: 
    curr_dict['Dataset'] = str(args.dataset)

curr_dict['Average'] = str(np.round(np.mean(accuracy), decimals=4) *100) + str(' += ') + str(np.round(np.var(accuracy), decimals=4)*100)
curr_dict['Statistical Parity'] = str(np.round(np.mean(satistical_parity), decimals=4)*100) + str(' += ') + str(np.round(np.var(satistical_parity), decimals=4)*100)
curr_dict['Equal Opportunity'] = str(np.round(np.mean(equal_opportunity), decimals=4)*100) + str(' += ') + str(np.round(np.var(equal_opportunity), decimals=4)*100)

print("curr_dict: ", curr_dict)
print("average for:", args.dataset , np.round(np.mean(accuracy), decimals=4) *100, '+=', np.round(np.var(accuracy), decimals=4)*100)
print("statistical parity:", args.dataset, np.round(np.mean(satistical_parity), decimals=4)*100, '+=', np.round(np.var(satistical_parity), decimals=4)*100)
print("equal Opportunity:", args.dataset, np.round(np.mean(equal_opportunity), decimals=4)*100, '+=', np.round(np.var(equal_opportunity), decimals=4)*100)


filename = 'marchRESULTS.csv'

# Writing data to CSV
with open(filename, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=curr_dict.keys())


    # Write header
    # writer.writeheader()

    # Write rows
    for i in range(0,1):
        row = {key: curr_dict[key] for key in curr_dict}
        writer.writerow(row)

print(f"Data has been written to {filename}")