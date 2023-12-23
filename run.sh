

# python test_VGCN.py --dataset income
# python test_VGCN.py --dataset bail
# python test_VGCN.py --dataset pokec_z
# python test_VGCN.py --dataset pokec_n
# python test_VGCN.py --dataset nba

# python test_MLP.py --dataset income
# python test_MLP.py --dataset bail
# python test_MLP.py --dataset pokec_z
# python test_MLP.py --dataset pokec_n
# python test_MLP.py --dataset nba

# python test_fairGNN_2.py --dataset income
# python test_fairGNN_2.py --dataset bail
# python test_fairGNN_2.py --dataset pokec_z
# python test_fairGNN_2.py --dataset pokec_n
# python test_fairGNN_2.py --dataset nba
# ./run.sh | tee output.txt

# try something new to compare
python test_VGCN.py --dataset income
python test_MLP.py --dataset income
python test_fairGNN_2.py --dataset income

python test_VGCN.py --dataset bail
python test_MLP.py --dataset bail
python test_fairGNN_2.py --dataset bail

python test_VGCN.py --dataset pokec_z
python test_MLP.py --dataset pokec_z
python test_fairGNN_2.py --dataset pokec_z

python test_VGCN.py --dataset pokec_n
python test_MLP.py --dataset pokec_n
python test_fairGNN_2.py --dataset pokec_n

python test_VGCN.py --dataset nba
python test_MLP.py --dataset nba
python test_fairGNN_2.py --dataset nba


python test_sage.py --dataset nba
python test_sage.py --dataset nba
python test_sage.py --dataset nba
python test_sage.py --dataset nba
python test_sage.py --dataset nba
python test_sage.py --dataset nba

python test_sage.py --dataset income
python test_sage.py --dataset nba
python test_sage.py --dataset pokec_n
python test_sage.py --dataset pokec_z
python test_sage.py --dataset bail


python test_NIFTY.py --dataset pokec_n
python test_NIFTY.py --dataset pokec_z
python test_NIFTY.py --dataset nba
python test_NIFTY.py --dataset bail
python test_NIFTY.py --dataset income

# run with all the seeds 
# check if it compares to the current hyperparameters + has better results
# if not ask prof
#NIFTY

python param_tuning_fairGNN.py --dataset income --seed 1
python param_tuning_fairGNN.py --dataset income --seed 2
python param_tuning_fairGNN.py --dataset income --seed 3
python param_tuning_fairGNN.py --dataset income --seed 4
python param_tuning_fairGNN.py --dataset income --seed 5

python param_tuning_fairGNN.py --dataset nba --seed 1
python param_tuning_fairGNN.py --dataset nba --seed 2
python param_tuning_fairGNN.py --dataset nba --seed 3
python param_tuning_fairGNN.py --dataset nba --seed 4
python param_tuning_fairGNN.py --dataset nba --seed 5

python param_tuning_fairGNN.py --dataset bail --seed 1
python param_tuning_fairGNN.py --dataset bail --seed 2
python param_tuning_fairGNN.py --dataset bail --seed 3
python param_tuning_fairGNN.py --dataset bail --seed 4
python param_tuning_fairGNN.py --dataset bail --seed 5

python param_tuning_fairGNN.py --dataset pokec_z --seed 1
python param_tuning_fairGNN.py --dataset pokec_z --seed 2
python param_tuning_fairGNN.py --dataset pokec_z --seed 3
python param_tuning_fairGNN.py --dataset pokec_z --seed 4
python param_tuning_fairGNN.py --dataset pokec_z --seed 5


python param_tuning_fairGNN.py --dataset pokec_n --seed 1
python param_tuning_fairGNN.py --dataset pokec_n --seed 2
python param_tuning_fairGNN.py --dataset pokec_n --seed 3
python param_tuning_fairGNN.py --dataset pokec_n --seed 4
python param_tuning_fairGNN.py --dataset pokec_n --seed 5
#NIFTY
python param_tuning_NIFTY.py --dataset pokec_n --seed 1
python param_tuning_NIFTY.py --dataset pokec_n --seed 2
python param_tuning_NIFTY.py --dataset pokec_n --seed 3
python param_tuning_NIFTY.py --dataset pokec_n --seed 4
python param_tuning_NIFTY.py --dataset pokec_n --seed 5

python param_tuning_NIFTY.py --dataset pokec_z --seed 1
python param_tuning_NIFTY.py --dataset pokec_z --seed 2
python param_tuning_NIFTY.py --dataset pokec_z --seed 3
python param_tuning_NIFTY.py --dataset pokec_z --seed 4
python param_tuning_NIFTY.py --dataset pokec_z --seed 5

python param_tuning_NIFTY.py --dataset bail --seed 1
python param_tuning_NIFTY.py --dataset bail --seed 2
python param_tuning_NIFTY.py --dataset bail --seed 3
python param_tuning_NIFTY.py --dataset bail --seed 4
python param_tuning_NIFTY.py --dataset bail --seed 5

python param_tuning_NIFTY.py --dataset nba --seed 1
python param_tuning_NIFTY.py --dataset nba --seed 2
python param_tuning_NIFTY.py --dataset nba --seed 3
python param_tuning_NIFTY.py --dataset nba --seed 4
python param_tuning_NIFTY.py --dataset nba --seed 5

python param_tuning_NIFTY.py --dataset income --seed 1
python param_tuning_NIFTY.py --dataset income --seed 2
python param_tuning_NIFTY.py --dataset income --seed 3
python param_tuning_NIFTY.py --dataset income --seed 4
python param_tuning_NIFTY.py --dataset income --seed 5

parser.add_argument('--dataset', type=str, default="nba", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--num_hidden', type=float, default=16)
parser.add_argument('--num_proj_hidden', type=float, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--sim_coeff', type=float, default=0.5 )


# NIFTY GCN
python test_NIFTY.py --dataset pokec_n --num_hidden 128 256 256 256 256 \
    --num_proj_hidden 64 16 4 16 128 --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.01 0.000001 0.01 0.0001 --sim_coeff 0.5 0.5 0.3 0.5 0.5

python test_NIFTY.py --dataset bail --num_hidden 128 256 16 256 16 \
    --num_proj_hidden 128 64 128 64 256 --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.0001 0.000001 0.01 0.000001 0.00001 --sim_coeff 0.5 0.5 0.5 0.5 0.5
    
python test_NIFTY.py --dataset pokec_z --num_hidden 256 16 128 128 16 \
    --num_proj_hidden 128 256 4 16 128 --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.000001 0.01 0.01 0.001 --sim_coeff 0.3 0.3 0.3 0.5 0.5

python test_NIFTY.py --dataset income --num_hidden 128 256 128 256 128 \
    --num_proj_hidden 256 64 16 64 128 --lr 0.00001 0.01 0.00001 0.01 0.00001 \
    --weight_decay 0.001 0.01 0.0001 0.01 0.001 --sim_coeff 0.3 0.3 0.3 0.3 0.5

python test_NIFTY.py --dataset nba --num_hidden 16 16 128 16 64 \
    --num_proj_hidden 4 16 64 128 16 --lr 0.01 0.01 0.001 0.01 0.01 \
    --weight_decay 0.001 0.001 0.000001 0.000001 0.01 --sim_coeff 0.5 0.3 0.7 0.3 0.5


# for GAT

python test_NIFTY_GAT.py --dataset pokec_n --num_hidden 64 256 64 128 16 \
    --num_proj_hidden 256 4 256 256 4 --lr 0.01 0.01 0.01 0.0001 0.01 \
    --weight_decay 0.01 0.01 0.001 0.0001 0.001 --sim_coeff 0.5 0.5 0.5 0.3 0.7

python test_NIFTY_GAT.py --dataset bail --num_hidden 128 64 64 256 16 \
    --num_proj_hidden 64 64 256 256 128 --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.00001 0.0001 0.01 0.0001 0.001 --sim_coeff 0.7 0.7 0.7 0.3 0.5
    
python test_NIFTY_GAT.py --dataset pokec_z --num_hidden 64 256 64 128 64 \
    --num_proj_hidden 128 128 128 4 64 --lr 0.01 0.001 0.01 0.01 0.01 \
    --weight_decay 0.01 0.0001 0.001 0.01 0.01 --sim_coeff 0.5 0.3 0.5 0.7 0.7

python test_NIFTY_GAT.py --dataset income --num_hidden 128 256 256 256 256 \
    --num_proj_hidden 256 16 256 128 64 --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.001 0.00001 0.001 0.001 0.000001 --sim_coeff 0.5 0.3 0.3 0.3 0.3

python test_NIFTY_GAT.py --dataset nba --num_hidden 64 128 128 64 64 \
    --num_proj_hidden 256 64 128 16 16 --lr 0.001 0.0001 0.001 0.001 0.001 \
    --weight_decay 0.000001 0.0001 0.0001 0.001 0.0001 --sim_coeff 0.3 0.3 0.5 0.5 0.3
