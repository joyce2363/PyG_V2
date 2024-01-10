

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

python param_tuning_fairGNN_ALL.py --dataset income --seed 1
python param_tuning_fairGNN_ALL.py --dataset income --seed 2
python param_tuning_fairGNN_ALL.py --dataset income --seed 3
python param_tuning_fairGNN_ALL.py --dataset income --seed 4
python param_tuning_fairGNN_ALL.py --dataset income --seed 5

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

python param_tuning_fairGNN_ALL.py --dataset bail --seed 1 --model gcn
python param_tuning_fairGNN_ALL.py --dataset bail --seed 2 --model gcn
python param_tuning_fairGNN_ALL.py --dataset bail --seed 3 --model gcn
python param_tuning_fairGNN_ALL.py --dataset bail --seed 4 --model gcn
python param_tuning_fairGNN_ALL.py --dataset bail --seed 5 --model gcn

python param_tuning_fairGNN_ALL.py --dataset income --seed 1 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 2 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 3 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 4 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 5 --model gcn

python param_tuning_fairGNN_ALL.py --dataset bail --seed 1 --model sage
python param_tuning_fairGNN_ALL.py --dataset bail --seed 2 --model sage
python param_tuning_fairGNN_ALL.py --dataset bail --seed 3 --model sage
python param_tuning_fairGNN_ALL.py --dataset bail --seed 4 --model sage
python param_tuning_fairGNN_ALL.py --dataset bail --seed 5 --model sage



python param_tuning_fairGNN_ALL.py --dataset nba --seed 1 --model gcn
python param_tuning_fairGNN_ALL.py --dataset nba --seed 2 --model gcn
python param_tuning_fairGNN_ALL.py --dataset nba --seed 3 --model gcn
python param_tuning_fairGNN_ALL.py --dataset nba --seed 4 --model gcn
python param_tuning_fairGNN_ALL.py --dataset nba --seed 5 --model gcn

python param_tuning_fairGNN_ALL.py --dataset nba --seed 1 --model gat
python param_tuning_fairGNN_ALL.py --dataset nba --seed 2 --model gat
python param_tuning_fairGNN_ALL.py --dataset nba --seed 3 --model gat
python param_tuning_fairGNN_ALL.py --dataset nba --seed 4 --model gat
python param_tuning_fairGNN_ALL.py --dataset nba --seed 5 --model gat

python param_tuning_fairGNN_ALL.py --dataset nba --seed 1 --model sage
python param_tuning_fairGNN_ALL.py --dataset nba --seed 2 --model sage
python param_tuning_fairGNN_ALL.py --dataset nba --seed 3 --model sage
python param_tuning_fairGNN_ALL.py --dataset nba --seed 4 --model sage
python param_tuning_fairGNN_ALL.py --dataset nba --seed 5 --model sage



python param_tuning_fairGNN_ALL.py --dataset income --seed 1 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 2 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 3 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 4 --model gcn
python param_tuning_fairGNN_ALL.py --dataset income --seed 5 --model gcn

python param_tuning_fairGNN_ALL.py --dataset income --seed 1 --model gat
python param_tuning_fairGNN_ALL.py --dataset income --seed 2 --model gat
python param_tuning_fairGNN_ALL.py --dataset income --seed 3 --model gat
python param_tuning_fairGNN_ALL.py --dataset income --seed 4 --model gat
python param_tuning_fairGNN_ALL.py --dataset income --seed 5 --model gat

python param_tuning_fairGNN_ALL.py --dataset income --seed 1 --model sage
python param_tuning_fairGNN_ALL.py --dataset income --seed 2 --model sage
python param_tuning_fairGNN_ALL.py --dataset income --seed 3 --model sage
python param_tuning_fairGNN_ALL.py --dataset income --seed 4 --model sage
python param_tuning_fairGNN_ALL.py --dataset income --seed 5 --model sage



python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 1 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 2 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 3 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 4 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 5 --model gcn

python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 1 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 2 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 3 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 4 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 5 --model gat

python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 1 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 2 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 3 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 4 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_z --seed 5 --model sage



python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 1 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 2 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 3 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 4 --model gcn
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 5 --model gcn

python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 1 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 2 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 3 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 4 --model gat
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 5 --model gat

python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 1 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 2 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 3 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 4 --model sage
python param_tuning_fairGNN_ALL.py --dataset pokec_n --seed 5 --model sage

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

python test_NIFTY.py --dataset pokec_z --num_hidden 256 16 128 128 16 \
    --num_proj_hidden 128 256 4 16 128 --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.000001 0.01 0.01 0.001 --sim_coeff 0.3 0.3 0.3 0.5 0.5

# fairGNN_SAGE
python test_fairGNN_SAGE.py --dataset bail --num_hidden 64 256 16 16 256 \
    --sim_coeff 0.5 0.5 0.7 0.3 0.5 --acc 0.7 0.2 0.3 0.5 0.7 --alpha 40 1 10 160 1 \
    --beta 1 20 20 80 380 --proj_hidden 128 16 64 256 128 --lr 0.00001 0.00001 0.01 0.001 0.00001 \
    --weight_decay 0.05 0.05 0.001 0.001 0.001

python test_fairGNN_SAGE.py --dataset income --num_hidden 64 256 16 16 256 \
    --sim_coeff 0.5 0.5 0.7 0.3 0.5 --acc 0.7 0.2 0.3 0.5 0.7 --alpha 40 1 10 160 1 \
    --beta 1 20 20 80 380 --proj_hidden 128 16 64 256 128 --lr 0.00001 0.00001 0.01 0.001 0.00001 \
    --weight_decay 0.05 0.05 0.001 0.001 0.001

python test_fairGNN_SAGE.py --dataset nba --num_hidden 64 128 16 128 256 \
    --sim_coeff 0.5 0.7 0.3 0.7 0.7 --acc 0.4 0.2 0.6 0.5 0.6 \
    --alpha 380 80 40 80 80 \
    --beta 80 380 380 1 1 --proj_hidden 16 16 4 256 64 --lr 0.01 0.001 0.00001 0.0001 0.00001 \
    --weight_decay 0.001 0.05 0.05 0.05 0.001

# testing
python test_fairGNN_ALL.py --dataset nba --model gat --num_hidden 64 64 16 256 128\
    --sim_coeff 0.7 0.7 0.5 0.7 0.7 --acc 0.3 0.3 0.5 0.6 0.4\
     --alpha 1 1 10 10 1\
    --beta 1 1 1 10 1 --proj_hidden 128 256 128 128 64 \
     --lr 0.001 0.001 0.001 0.001 0.01 \
    --weight_decay 0.05 0.05 0.01 0.05 0.01

python test_fairGNN_ALL.py --dataset bail --model gat --num_hidden 128 64 256 128 64\
    --sim_coeff 0.7 0.5 0.5 0.7 0.7 --acc 0.6 0.3 0.2 0.5 0.7\
     --alpha 1 1 80 20 1\
    --beta 10 10 10 160 10 --proj_hidden 256 4 256 128 256 \
     --lr 0.01 0.001 0.01 0.001 0.01 \
    --weight_decay 0.05 0.05 0.05 0.05 0.05

python test_fairGNN_ALL.py --dataset income --model gat --num_hidden 16 128 256 64 256\
    --sim_coeff 0.7 0.7 0.7 0.3 0.3 --acc 0.2 0.2 0.3 0.5 0.2\
     --alpha 20 20 80 10 1\
    --beta 1 1 1 20 1 --proj_hidden 64 16 64 128 16 \
     --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.001 0.0001 0.002 0.05 0.002

python test_fairGNN_ALL.py --dataset nba --model gcn --num_hidden 64 256 16 64 16\
    --sim_coeff 0.3 0.5 0.5 0.7 0.5 --acc 0.3 0.5 0.5 0.6 0.5 \
     --alpha 10 380 20 80 1 \
    --beta 40 80 40 40 20 --proj_hidden 16 256 16 4 16 \
     --lr 0.00001 0.0001 0.001 0.001 0.01 \
    --weight_decay 0.01 0.05 0.05 0.002 0.001

python test_fairGNN_ALL.py --dataset bail --model gcn --num_hidden 64 128 256 256 128\
    --sim_coeff 0.3 0.7 0.3 0.3 0.7 --acc 0.6 0.6 0.2 0.6 0.3 \
     --alpha 380 20 1 80 380 \
    --beta 40 1 160 10 40 --proj_hidden 64 128 256 256 16 \
     --lr  0.01 0.01 0.00001 0.001 0.001\
    --weight_decay 0.01 0.002 0.0001 0.05 0.01

python test_fairGNN_ALL.py --dataset income --model gcn --num_hidden 256 64 256 256 16\
    --sim_coeff 0.5 0.5 0.5 0.7 0.3 --acc 0.5 0.4 0.4 0.7 0.3 \
     --alpha 10 380 80 40 1\
    --beta 80 20 380 10 10 --proj_hidden 64 64 4 16 256\
     --lr 0.001 0.00001 0.01 0.001 0.0001 \
    --weight_decay 0.0001 0.002 0.002 0.05 0.05

python test_fairGNN_ALL.py --dataset income --model gcn \
    --num_hidden 256 \
    --sim_coeff 0.5 --acc 0.5 \
    --alpha 10 \
    --beta 80 \
    --proj_hidden 64 \
    --lr 0.001 \
    --weight_decay 0.0001


# ANOTHER

python test_fairGNN_ALL.py --dataset pokec_z --model gcn \
    --num_hidden 128 18 128 256 128 128 \
    --sim_coeff 0.7 0.6 0.5 0.6 0.3 \
    --acc 0.6 0.4 0.5 0.2 0.2 \
     --alpha 20 10 5 6 10 \
    --beta 0.1 0.0001 0.001 0.001 0.1 \
     --proj_hidden 8 8 16 64 16 \
     --lr 0.01 0.01 0.01 0.01 0.01\
    --weight_decay 0.01 0.01 0.00001 0.01 0.01

python test_fairGNN_ALL.py --dataset pokec_z --model gat \
    --num_hidden 128 64 128 128 128\
    --sim_coeff 0.5 0.3 0.5 0.3 0.6\
     --acc 0.2 0.4 0.4 0.6 0.4 \
     --alpha 10 6 20 7 20 \
    --beta 0.001 0.1 0.001 0.0001 0.001 \
     --proj_hidden 128 16 128 128 8\
     --lr 0.01 0.01 0.01 0.00001 0.01 \
    --weight_decay 0.0001 0.001 0.00001 0.01 0.00001

python test_fairGNN_ALL.py --dataset pokec_z --model sage \
    --num_hidden 128 128 256 64 64 \
    --sim_coeff 0.6 0.5 0.5 0.7 0.7\
     --acc 0.3 0.5 0.2 0.6 0.3 \
    --alpha 5 40 10 20 3 \
    --beta 0.1 0.0001 0.01 0.0001 0.001 \
    --proj_hidden 4 16 64 4 8 \
    --lr 0.01 0.01 0.001 0.01 0.01 \
    --weight_decay 0.01 0.00001 0.001 0.0001 0.01

# ANOTHER

python test_fairGNN_ALL.py --dataset nba --model gcn \
    --num_hidden 64 16 128 16 16 \
    --sim_coeff 0.5 0.6 0.6 0.7 0.7 \
    --acc 0.5 0.69 0.2 0.5 0.69 \
     --alpha 1 4 6 20 4 \
    --beta 0.1 0.001 0.01 0.001 0.1 \
     --proj_hidden 16 64 64 8 8 \
     --lr 0.001 0.01 0.001 0.01 0.01 \
    --weight_decay 0.00001 0.001 0.01 0.01 0.01

python test_fairGNN_ALL.py --dataset nba --model gat \
    --num_hidden 16 256 256 16 16 \
    --sim_coeff 0.5 0.5 0.7 0.3 0.5 \
    --acc 0.7 0.2 0.5 0.5 0.6 \
     --alpha 4 6 10 1 6 \
    --beta 0.0001 0.01 0.0001 0.01 0.1 \
     --proj_hidden 64 64 4 8 16 \
     --lr 0.001 0.0001 0.0001 0.001 0.001 \
    --weight_decay 0.01 0.01 0.01 0.00001 0.01

python test_fairGNN_ALL.py --dataset nba --model sage \
    --num_hidden 64 256 256 256 128 \
    --sim_coeff 0.3 0.6 0.6 0.5 0.5 \
    --acc 0.4 0.5 0.7 0.3 0.3 \
     --alpha 40 40 10 6 4 \
    --beta 0.001 0.0001 0.001 0.01 0.001 \
     --proj_hidden 128 64 4 4 128 \
     --lr 0.0001 0.01 0.0001 0.0001 0.001 \
    --weight_decay 0.00001 0.01 0.00001 0.001 0.001


# ANOTHER
python test_fairGNN_ALL.py --dataset pokec_n --model gcn \
    --num_hidden 256 64 64 128 16  \
    --sim_coeff 0.6 0.6 0.7 0.3 0.6 \
    --acc 0.4 0.4 0.3 0.5 0.5 \
     --alpha 40 40 20 6 3 \
    --beta 0.0001 0.01 0.001 0.0001 0.001\
     --proj_hidden 8 64 128 128 4 \
     --lr 0.01 0.01 0.01 0.01 0.01\
    --weight_decay 0.0001 0.01 0.001 0.0001 0.01

python test_fairGNN_ALL.py --dataset pokec_n --model gat \
    --num_hidden 64 16 64 128 256 \
    --sim_coeff 0.7 0.3 0.5 0.5 0.6 \
    --acc 0.3 0.3 0.3 0.3 0.4 \
     --alpha 10 7 7 1 7 \
    --beta 0.1 0.0001 0.0001 0.0001 0.1 \
     --proj_hidden 64 16 8 64 128 \
     --lr 0.01 0.01 0.0001 0.01 0.01\
    --weight_decay 0.00001 0.01 0.001 0.0001 0.0001

python test_fairGNN_ALL.py --dataset pokec_n --model sage \
    --num_hidden 128 64 128 256 64 \
    --sim_coeff 0.6 0.6 0.5 0.5 0.7 \
    --acc 0.5 0.5 0.3 0.5 0.5 \
     --alpha 40 7 40 6 4 \
    --beta 0.0001 0.001 0.0001 0.1 0.001\
     --proj_hidden 64 64 64 4 64  \
     --lr 0.01 0.01 0.01 0.01 0.001\
    --weight_decay 0.00001 0.00001 0.00001 0.01 0.001
# BAIL ANOTHER 
python test_fairGNN_ALL.py --dataset bail --model gcn \
    --num_hidden 64 256 256 128 256  \
    --sim_coeff 0.5 0.6 0.6 0.6 0.6 \
    --acc 0.69 0.69 0.4 0.7 0.69 \
     --alpha 6 40 4 6 6 \
    --beta 0.0001 0.001 0.001 0.1 0.001 \
     --proj_hidden 8 4 8 16 128 \
     --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.0001 0.00001 0.0001 0.0001 0.00001

python test_fairGNN_ALL.py --dataset bail --model gat \
    --num_hidden  16 16 256 16 16 \
    --sim_coeff 0.7 0.6 0.6 0.5 0.6 \
    --acc 0.5 0.4 0.2 0.5 0.69\
     --alpha 5 7 7 20 20 \
    --beta  0.0001 0.01 0.0001 0.001 0.001\
     --proj_hidden 4 64 4 8 4 \
     --lr  0.01 0.01 0.01 0.01 0.01\
    --weight_decay 0.01 0.01 0.01 0.01 0.0001

python test_fairGNN_ALL.py --dataset bail --model sage \
    --num_hidden  256 256 128 256 256  \
    --sim_coeff 0.6 0.3 0.7 0.6 0.3 \
    --acc 0.3 0.6 0.4 0.6 0.7 \
     --alpha 10 40 40 20 40\
    --beta 0.1 0.001 0.1 0.0001 0.1 \
     --proj_hidden 64 128 128 4 64 \
     --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.01 0.01 0.01 0.01

#NIFTY

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






# TOTALLY NEW ONe

python test_fairGNN_ALL.py --dataset nba --model gcn \
    --num_hidden 64 16 128 128 256\
    --sim_coeff 0.5 0.6 0.3 0.5 0.5\
    --acc 0.2 0.5 0.6 0.6 0.7\
     --alpha 5 40 7 6 10\
    --beta 0.001 0.01 0.01 0.0001 0.0001\
     --proj_hidden 4 128 16 64 64 \
     --lr 0.01 0.001 0.001 0.001 0.01\
    --weight_decay 0.001 0.001 0.001 0.001 0.01

python test_fairGNN_ALL.py --dataset nba --model gat \
    --num_hidden 16 128 256 256 64\
    --sim_coeff 0.5 0.7 0.5 0.7 0.3\
    --acc 0.7 0.6 0.4 0.6 0.7\
     --alpha 6 5 1 4 5\
    --beta 0.1 0.1 0.001 0.0001 0.01\
     --proj_hidden 8 128 16 8 16\
     --lr 0.01 0.0001 0.0001 0.0001 0.0001\
    --weight_decay 0.01 0.00001 0.0001 0.0001 0.01

python test_fairGNN_ALL.py --dataset nba --model sage \
    --num_hidden 16 128 256 128 16\
    --sim_coeff 0.7 0.6 0.5 0.7 0.6\
    --acc 0.3 0.4 0.4 0.7 0.5\
     --alpha 40 20 4 4 4\
    --beta 0.0001 0.0001 0.001 0.001 0.001\
     --proj_hidden 64 4 8 8 64 \
     --lr 0.0001 0.001 0.0001 0.001 0.0001\
    --weight_decay 0.0001 0.0001 0.01 0.001 0.001

python test_fairGNN_ALL.py --dataset pokec_z --model gcn \
    --num_hidden 64 128 256 128 64\
    --sim_coeff 0.5 0.3 0.3 0.3 0.7\
    --acc 0.4 0.3 0.2 0.4 0.6\
     --alpha 40 40 4 10 6\
    --beta 0.01 0.001 0.001 0.01 0.0001\
     --proj_hidden  4 16 4 128 4\
     --lr 0.01 0.01 0.001 0.01 0.01\
    --weight_decay 0.01 0.01 0.01 0.0001 0.0001