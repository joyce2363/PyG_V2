

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

# NIFTY Methods 
# this is running 5 times, for seed 1, 2, 3, 4, 5 
# it should output the average result for all 5 seeds
python test_NIFTY_ALL.py --dataset income \
    --model sage \
    --num_hidden 64 256 128 128 256 \
    --num_proj_hidden 128 32 16 16 64 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.001 0.001 0.0001 0.01 \
    --sim_coeff 0.3 0.4 0.3 0.6 0.4 \
    --drop_edge_rate_1 0.1 0.0001 0.0001 0.001 0.001 \
    --drop_edge_rate_2 0.1 0.0001 0.0001 0.1 0.0001 \
    --drop_feature_rate_1 0.001 0.0001 0.1 0.0001 0.0001 \
    --drop_feature_rate_2 0.1 0.001 0.0001 0.0001 0.001 \

python test_NIFTY_ALL.py --dataset pokec_n \
    --model gat \
    --num_hidden 256 256 4 16 64 \
    --num_proj_hidden 16 32 32 128 256 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.01 0.01 0.00001 0.00001 \
    --sim_coeff 0.7 0.6 0.4 0.3 0.4 \
    --drop_edge_rate_1 0.1 0.01 0.001 0.001 0.001 \
    --drop_edge_rate_2 0.001 0.001 0.0001 0.001 0.001 \
    --drop_feature_rate_1 0.01 0.0001 0.0001 0.1 0.01 \
    --drop_feature_rate_2 0.1 0.0001 0.01 0.0001 0.001 \


python test_NIFTY_ALL.py --dataset pokec_n \
    --model gat \
    --num_hidden 256 \
    --num_proj_hidden 16 \
    --lr 0.01 \
    --weight_decay 0.01 \
    --sim_coeff 0.7 \
    --drop_edge_rate_1 0.1 \
    --drop_edge_rate_2 0.001 \
    --drop_feature_rate_1 0.01 \
    --drop_feature_rate_2 0.1 \

python test_NIFTY_ALL.py --dataset pokec_z \
    --model gcn \
    --num_hidden 256 64 256 128 256 \
    --num_proj_hidden 128 4 32 128 4 \
    --lr 0.0001 0.01 0.0001 0.01 0.01 \
    --weight_decay 0.01 0.000001 0.01 0.01 0.01 \
    --sim_coeff 0.7 0.7 0.7 0.3 0.6 \
    --drop_edge_rate_1 0.1 0.0001 0.001 0.1 0.1 \
    --drop_edge_rate_2 0.001 0.0001 0.1 0.0001 0.1 \
    --drop_feature_rate_1 0.001 0.0001 0.001 0.001 0.0001 \
    --drop_feature_rate_2 0.1 0.001 0.1 0.1 0.1 \

python test_NIFTY_ALL.py --dataset income \
    --model gcn \
    --num_hidden 128 64 64 256 128 \
    --num_proj_hidden 16 64 4 256 128 \
    --lr 0.00001 0.01 0.01 0.01 0.00001 \
    --weight_decay 0.01 0.01 0.01 0.01 0.000001 \
    --sim_coeff 0.5 0.6 0.5 0.7 0.6 \
    --drop_edge_rate_1 0.01 0.01 0.01 0.01 0.01 \
    --drop_edge_rate_2 0.01 0.01 0.01 0.01 0.01 \
    --drop_feature_rate_1 0.1 0.1 0.1 0.1 0.1 \
    --drop_feature_rate_2 0.1 0.1 0.1 0.1 0.1 \

# if you want to run only one seed
# python test_NIFTY_ALL.py --dataset income --model gcn \
#     --num_hidden 128 \
#     --num_proj_hidden 16 \
#     --lr 0.00001 \
#     --weight_decay 0.01 \
#     --sim_coeff 0.5 \
#     --drop_edge_rate_1 0.0001 \
#     --drop_edge_rate_2 0.0001 \
#     --drop_feature_rate_1 0.1 \
#     --drop_feature_rate_2 0.0001 \


# fairGNN methods
python test_fairGNN_ALL.py --dataset bail \
    --model gat \
    --num_hidden 256 256 64 256 16 \
    --sim_coeff 0.3 0.5 0.3 0.6 0.6 \
    --acc 0.4 0.688 0.69 0.7 0.2 \
    --alpha 6 20 10 1 6 \
    --beta 0.1 0.0001 0.0001 1 1 \
    --proj_hidden 128 128 8 128 64 \
    --lr 0.01 0.01 0.01 0.001 0.01 \
    --weight_decay 0.0001 0.00001 0.01 0.01 0.001

python test_fairGNN_ALL.py --dataset nba \
    --model sage \
    --num_hidden 128 256 128 16 128 \
    --sim_coeff 0.7 0.3 0.3 0.7 0.6 \
    --acc 0.5 0.2 0.3 0.3 0.69 \
    --alpha 3 1 2 6 3 \
    --beta 0.001 0.01 0.0001 0.001 0.01 \
    --proj_hidden 64 4 8 16 64 \
    --lr 0.0001 0.0001 0.001 0.001 0.0001 \
    --weight_decay 0.0001 0.01 0.0001 0.001 0.00001

python test_fairGNN_ALL.py --dataset bail \
    --model gcn \
    --num_hidden 64 256 256 128 128 \
    --sim_coeff 0.6 0.7 0.3 0.5 0.6 \
    --acc 0.69 0.68 0.7 0.68 0.7 \
    --alpha 50 10 4 3 7 \
    --beta 0.1 0.1 0.001 0.01 0.001 \
    --proj_hidden 8 16 16 128 64 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.00001 0.0001 0.0001 0.00001 0.001

# MLP
python test_MLP.py --dataset nba \
    --num_hidden 128 128 128 64 256 \
    --num_proj_hidden 256 4 16 256 4 \
    --lr 0.0001 0.01 0.01 0.001 0.001 \
    --weight_decay 0.05 0.01 0.05 0.0001 0.01\
    --sim_coeff 0.7 0.7 0.5 0.3 0.3 
