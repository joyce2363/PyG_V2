

python test_NIFTY.py --dataset pokec_n
python test_NIFTY.py --dataset pokec_z
python test_NIFTY.py --dataset nba
python test_NIFTY.py --dataset bail
python test_NIFTY.py --dataset income

# run with all the seeds 
# check if it compares to the current hyperparameters + has better results

#NIFTY

python param_tuning_NIFTY_ALL.py --dataset bail --seed 1 --model gat
python param_tuning_NIFTY_ALL.py --dataset bail --seed 2 --model gat
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 3 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 4 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 5 --model sage

python param_tuning_VGCN.py --dataset bail --seed 1 
python param_tuning_VGCN.py --dataset bail --seed 2 
python param_tuning_VGCN.py --dataset bail --seed 3 
python param_tuning_VGCN.py --dataset bail --seed 4 
python param_tuning_VGCN.py --dataset bail --seed 5 



python param_tuning_fairGNN_ALL.py --dataset nba --seed 1 --model sage
python param_tuning_fairGNN_ALL.py --dataset nba --seed 2 --model sage



python param_tuning_NIFTY.py --dataset pokec_z --seed 1
python param_tuning_NIFTY.py --dataset pokec_z --seed 2
python param_tuning_NIFTY.py --dataset pokec_z --seed 3
python param_tuning_NIFTY.py --dataset pokec_z --seed 4
python param_tuning_NIFTY.py --dataset pokec_z --seed 5

python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 1 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 2 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 3 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 4 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 5 --model sage

python param_tuning_NIFTY_ALL.py --dataset income --seed 1 --model sage
python param_tuning_NIFTY_ALL.py --dataset income --seed 2 --model sage
python param_tuning_NIFTY_ALL.py --dataset income --seed 3 --model sage
python param_tuning_NIFTY_ALL.py --dataset income --seed 4 --model sage
python param_tuning_NIFTY_ALL.py --dataset income --seed 5 --model sage

python param_tuning_NIFTY_ALL.py --dataset nba --seed 1 --model gat
python param_tuning_NIFTY_ALL.py --dataset nba --seed 2 --model gat
python param_tuning_NIFTY_ALL.py --dataset nba --seed 3 --model gat
python param_tuning_NIFTY_ALL.py --dataset nba --seed 4 --model gat
python param_tuning_NIFTY_ALL.py --dataset nba --seed 5 --model gat

python param_tuning_NIFTY_ALL.py --dataset bail --seed 1 --model gat
python param_tuning_NIFTY_ALL.py --dataset bail --seed 2 --model gat
python param_tuning_NIFTY_ALL.py --dataset bail --seed 3 --model sage
python param_tuning_NIFTY_ALL.py --dataset bail --seed 4 --model sage
python param_tuning_NIFTY_ALL.py --dataset bail --seed 5 --model sage

python param_tuning_NIFTY_ALL.py --dataset pokec_n --seed 1 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_n --seed 2 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_n --seed 3 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_n --seed 4 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_n --seed 5 --model sage

python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 1 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 2 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 3 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 4 --model sage
python param_tuning_NIFTY_ALL.py --dataset pokec_z --seed 5 --model sage


python param_tuning_VGCN.py --dataset bail --seed 1
python param_tuning_VGCN.py --dataset bail --seed 2
python param_tuning_VGCN.py --dataset bail --seed 3
python param_tuning_VGCN.py --dataset bail --seed 4
python param_tuning_VGCN.py --dataset bail --seed 5

python param_tuning_VGCN.py --dataset nba --seed 2
python param_tuning_VGCN.py --dataset nba --seed 3
python param_tuning_VGCN.py --dataset nba --seed 4
python param_tuning_VGCN.py --dataset nba --seed 5

python param_tuning_VGCN.py --dataset income --seed 1
python param_tuning_VGCN.py --dataset income --seed 2



python param_tuning_VGCN.py --dataset nba --seed 1 --model gat
python param_tuning_VGCN.py --dataset nba --seed 2 --model gat
python param_tuning_VGCN.py --dataset nba --seed 3 --model gat
python param_tuning_VGCN.py --dataset nba --seed 4 --model gat
python param_tuning_VGCN.py --dataset nba --seed 5 --model gat



python param_tuning_fairGNN_ALL.py --dataset income --seed 5 --model gat
python param_tuning_fairGNN_ALL.py --dataset bail --seed 5 --model gat

parser.add_argument('--dataset', type=str, default="nba", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--num_hidden', type=float, default=16)
parser.add_argument('--num_proj_hidden', type=float, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--sim_coeff', type=float, default=0.5 )


    num_hidden = trial.suggest_categorical("num_hidden", [4, 16, 32, 64, 128, 256])
    num_proj_hidden = trial.suggest_categorical("num_proj_hidden", [4, 16, 32, 64, 128, 256])
    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4, 1e-5])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    sim_coeff = trial.suggest_categorical("sim_coeff", [0.3, 0.4, 0.5, 0.6, 0.7])
    drop_edge_rate_1 = trial.suggest_categorical("drop_edge_rate_1",  [0.1, 0.001, 0.0001])
    drop_edge_rate_2 = trial.suggest_categorical("drop_edge_rate_2",  [0.1, 0.001, 0.0001])
    drop_feature_rate_1 = trial.suggest_categorical("drop_feature_rate_1",  [0.1, 0.001, 0.0001])
    drop_feature_rate_2 = trial.suggest_categorical("drop_feature_rate_2",  [0.1, 0.001, 0.0001])


# NIFTY GCN
python test_NIFTY_ALL.py --dataset income --model sage \
    --num_hidden 64 256 128 128 256 \
    --num_proj_hidden 128 32 16 16 64 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.001 0.001 0.0001 0.01 \
    --sim_coeff 0.3 0.4 0.3 0.6 0.4 \
    --drop_edge_rate_1 0.1 0.0001 0.0001 0.001 0.001 \
    --drop_edge_rate_2 0.1 0.0001 0.0001 0.1 0.0001 \
    --drop_feature_rate_1 0.001 0.0001 0.1 0.0001 0.0001 \
    --drop_feature_rate_2 0.1 0.001 0.0001 0.0001 0.001 \

python test_NIFTY_ALL.py --dataset pokec_n --model gat \
    --num_hidden 256 256 4 16 64 \
    --num_proj_hidden 16 32 32 128 256 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.01 0.01 0.00001 0.00001 \
    --sim_coeff 0.7 0.6 0.4 0.3 0.4 \
    --drop_edge_rate_1 0.1 0.01 0.001 0.001 0.001 \
    --drop_edge_rate_2 0.001 0.001 0.0001 0.001 0.001 \
    --drop_feature_rate_1 0.01 0.0001 0.0001 0.1 0.01 \
    --drop_feature_rate_2 0.1 0.0001 0.01 0.0001 0.001 \

python test_NIFTY_ALL.py --dataset pokec_z --model gcn     --num_hidden 256 64 256 128 256     --num_proj_hidden 128 4 32 128 4     --lr 0.0001 0.01 0.0001 0.01 0.01     --weight_decay 0.01 0.000001 0.01 0.01 0.01     --sim_coeff 0.7 0.7 0.7 0.3 0.6     --drop_edge_rate_1 0.1 0.0001 0.001 0.1 0.1    --drop_edge_rate_2 0.001 0.0001 0.1 0.0001 0.1     --drop_feature_rate_1 0.001 0.0001 0.001 0.001 0.0001     --drop_feature_rate_2 0.1 0.001 0.1 0.1 0.1\

python test_NIFTY_ALL.py --dataset income --model gcn \
    --num_hidden 128 64 64 256 128 \
    --num_proj_hidden 16 64 4 256 128 \
    --lr 0.00001 0.01 0.01 0.01 0.00001 \
    --weight_decay 0.01 0.01 0.01 0.01 0.000001 \
    --sim_coeff 0.5 0.6 0.5 0.7 0.6 \
    --drop_edge_rate_1 0.01 0.01 0.01 0.01 0.01 \
    --drop_edge_rate_2 0.01 0.01 0.01 0.01 0.01 \
    --drop_feature_rate_1 0.1 0.1 0.1 0.1 0.1 \
    --drop_feature_rate_2 0.1 0.1 0.1 0.1 0.1 \

python test_NIFTY_ALL.py --dataset income --model gcn \
    --num_hidden 128 \
    --num_proj_hidden 16 \
    --lr 0.00001 \
    --weight_decay 0.01 \
    --sim_coeff 0.5 \
    --drop_edge_rate_1 0.0001 \
    --drop_edge_rate_2 0.0001 \
    --drop_feature_rate_1 0.1 \
    --drop_feature_rate_2 0.0001 \

python test_NIFTY_ALL.py --dataset pokec_n --num_hidden 16 128 16 256 256 \
    --num_proj_hidden 128 256 64 64 128 --lr 0.01 0.01 0.01 0.01 0.001 \
    --weight_decay 0.000001 0.001 0.0001 0.01 0.001 \
    --sim_coeff 0.3 0.3 0.7 0.7 0.3

python test_NIFTY_ALL.py --dataset nba --num_hidden 128 16 256 16 64 \
    --num_proj_hidden 64 128 256 128 256 --lr 0.001 0.001 0.0001 0.001 0.001 \
    --weight_decay 0.05 0.05 0.01 0.05 0.05 \
    --sim_coeff 0.7 0.5 0.3 0.3 0.7

# fairGNN_SAGE
python test_fairGNN_ALL.py --dataset bail --model gat\
    --num_hidden 256 256 64 256 16 \
    --sim_coeff 0.3 0.5 0.3 0.6 0.6 \
    --acc 0.4 0.688 0.69 0.7 0.2 \
    --alpha 6 20 10 1 6 \
    --beta 0.1 0.0001 0.0001 1 1 \
    --proj_hidden 128 128 8 128 64 \
    --lr 0.01 0.01 0.01 0.001 0.01 \
    --weight_decay 0.0001 0.00001 0.01 0.01 0.001

python test_fairGNN_ALL.py --dataset nba --model sage    --num_hidden 128 256 128 16 128     --sim_coeff 0.7 0.3 0.3 0.7 0.6     --acc 0.5 0.2 0.3 0.3 0.69     --alpha 3 1 2 6 3     --beta 0.001 0.01 0.0001 0.001 0.01     --proj_hidden 64 4 8 16 64     --lr 0.0001 0.0001 0.001 0.001 0.0001     --weight_decay 0.0001 0.01 0.0001 0.001 0.00001

python test_fairGNN_ALL.py --dataset bail  --model gcn   --num_hidden 64 256 256 128 128     --sim_coeff 0.6 0.7 0.3 0.5 0.6     --acc 0.69 0.68 0.7 0.68 0.7     --alpha 50 10 4 3 7     --beta 0.1 0.1 0.001 0.01 0.001     --proj_hidden 8 16 16 128 64     --lr 0.01 0.01 0.01 0.01 0.01     --weight_decay 0.00001 0.0001 0.0001 0.00001 0.001

# testing

python test_MLP.py --dataset nba \
    --num_hidden 128 128 128 64 256 \
    --num_proj_hidden 256 4 16 256 4 \
    --lr 0.0001 0.01 0.01 0.001 0.001 \
     --weight_decay 0.05 0.01 0.05 0.0001 0.01\
    --sim_coeff 0.7 0.7 0.5 0.3 0.3 

# ANOTHER


#NIFTY

python test_NIFTY_ALL.py --dataset pokec_n \
    --num_hidden 128 16 64 128 256 \
    --num_proj_hidden 16 4 128 16 256 \
    --lr 0.0001 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.001 0.0001 0.0001 0.0001 \
    --sim_coeff 0.7 0.3 0.3 0.3 0.5

# TOTALLY NEW ONe

256,128,0.001,0.01,0.5,pokec_n,1,0.6263736263736264,gat seed = 1
4,128,0.01,0.05,0.3,pokec_n,2,0.6302521008403361,gat seed = 2
16,16,0.01,0.01,0.7,pokec_n,3,0.6153846153846154,gat seed = 3
16,256,0.01,0.01,0.3,pokec_n,4,0.6296056884292178,gat seed = 4
64,256,0.01,0.002,0.7,pokec_n,5,0.6237879767291532,gat seed = 5

python test_VGCN.py --dataset pokec_n --model gat \
    --num_hidden 256 4 16 16 64 \
    --num_proj_hidden 128 128 16 256 256 \
    --lr 0.001 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.05 0.01 0.01 0.002 \
    --sim_coeff 0.5 0.3 0.7 0.3 0.7 \

16,64,0.01,0.002,0.5,pokec_z,1,0.6532637075718015,GAT seed = 1
4,64,0.01,1e-05,0.5,pokec_z,2,0.6313315926892951,GAT seed = 2
256,128,0.01,0.05,0.7,pokec_z,3,0.6590078328981723,GAT seed = 3
16,256,0.01,0.05,0.3,pokec_z,4,0.6339425587467363,GAT seed = 4
128,128,0.01,0.05,0.3,pokec_z,5,0.6407310704960836,GAT seed = 5

python test_VGCN.py --dataset pokec_z --model gat \
    --num_hidden 16 4 256 16 128 \
    --num_proj_hidden 64 64 128 256 128 \
    --lr 0.01 0.01 0.01 0.01 0.01\
    --weight_decay 0.002 0.00001 0.05 0.05 0.05\
    --sim_coeff 0.5 0.5 0.7 0.3 0.3

BAIL:
16,64,0.001,0.01,0.5,bail,1,0.8931977113795295,gat seed = 1
16,16,0.01,0.01,0.3,bail,2,0.9023098114007205,gat seed = 2
256,16,0.01,0.001,0.3,bail,3,0.882178427632973,gat seed = 3
4,256,0.01,1e-05,0.3,bail,4,0.9143886416613689,gat seed = 4
16,128,0.01,0.01,0.5,bail,5,0.8999788090697182,gat seed = 5

python test_VGCN.py --dataset bail --model gat \
    --num_hidden 16 16 256 4 16 \
    --num_proj_hidden 64 16 16 256 128 \
    --lr 0.001 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.01 0.001 0.00001 0.01\
    --sim_coeff 0.5 0.3 0.3 0.3 0.5