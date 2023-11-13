from pygdebias.debiasing import GNN
from pygdebias.datasets import Pokec_n

# Available choices: 'Credit', 'German', 'Facebook', 'Pokec_z', 'Pokec_n', 'Nba', 'Twitter', 'Google', 'LCC', 'LCC_small', 'Cora', 'Citeseer', 'Amazon', 'Yelp', 'Epinion', 'Ciao', 'Dblp', 'Filmtrust', 'Lastfm', 'Ml-100k', 'Ml-1m', 'Ml-20m', 'Oklahoma', 'UNC', 'Bail'.

pokec_n = Pokec_n()
adj, features, idx_train, idx_val, idx_test, labels, sens = (
    pokec_n.adj(),
    pokec_n.features(),
    pokec_n.idx_train(),
    pokec_n.idx_val(),
    pokec_n.idx_test(),
    pokec_n.labels(),
    pokec_n.sens(),
)

print("idx_train: ", len(idx_train))
print("idx_val: ", len(idx_val))
print("idx_test: ", len(idx_test))

print("len of labels: ", len(labels))
model = GNN(adj, 
            features, 
            labels, 
            idx_train, 
            idx_val, 
            idx_test, 
            sens, 
            idx_train) #features.shape[1]

# Train the model.
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
