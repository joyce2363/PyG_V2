# # NIFTY GCN
# python test_NIFTY_ALL.py --dataset income \
#     --model gcn \
#     --encoder gcn \
#     --num_hidden 4 256 128 64 256 \
#     --num_proj_hidden 64 4 256 16 16 \
#     --lr 0.00001 0.00001 0.01 0.00001 0.01 \
#     --weight_decay 0.00001 0.001 0.001 0.01 0.01 \
#     --sim_coeff 0.4 0.7 0.7 0.3 0.7 \
#     --drop_edge_rate_1 0.0001 0.0001 0.1 0.01 0.0001 \
#     --drop_edge_rate_2 0.0001 0.001 0.001 0.01 0.1 \
#     --drop_feature_rate_1 0.0001 0.1 0.01 0.001 0.01 \
#     --drop_feature_rate_2 0.0001 0.1 0.0001 0.01 0.001 \

# python test_NIFTY_ALL.py --dataset nba \
#     --model gcn \
#     --encoder gcn \
#     --num_hidden 4 64 4 16 4 \
#     --num_proj_hidden 128 64 256 32 64 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.000001 0.0001 0.0001 0.01 0.0001 \
#     --sim_coeff 0.4 0.7 0.7 0.5 0.5 \
#     --drop_edge_rate_1 0.01 0.0001 0.01 0.01 0.001 \
#     --drop_edge_rate_2 0.0001 0.01 0.001 0.001 0.1 \
#     --drop_feature_rate_1 0.001 0.001 0.001 0.01 0.1 \
#     --drop_feature_rate_2 0.01 0.001 0.1 0.01 0.001 \

# python test_NIFTY_ALL.py --dataset bail \
#     --model gcn \
#     --encoder gcn \
#     --num_hidden 256 256 64 128 256 \
#     --num_proj_hidden 64 256 256 16 32 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.0001 0.000001 0.0001 0.000001 0.0001 \
#     --sim_coeff 0.6 0.7 0.4 0.6 0.7 \
#     --drop_edge_rate_1 0.01 0.0001 0.1 0.0001 0.0001 \
#     --drop_edge_rate_2 0.01 0.0001 0.001 0.01 0.001 \
#     --drop_feature_rate_1 0.01 0.0001 0.001 0.01 0.001 \
#     --drop_feature_rate_2 0.0001 0.001 0.01 0.0001 0.01 \

# python test_NIFTY_ALL.py --dataset pokec_n \
#     --model gcn \
#     --encoder gcn \
#     --num_hidden 256 128 128 32 64 \
#     --num_proj_hidden 64 4 128 4 128 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.01 0.00001 0.000001 0.0001 0.000001 \
#     --sim_coeff 0.7 0.4 0.6 0.3 0.4 \
#     --drop_edge_rate_1 0.0001 0.1 0.001 0.1 0.0001 \
#     --drop_edge_rate_2 0.1 0.001 0.0001 0.0001 0.01 \
#     --drop_feature_rate_1 0.01 0.1 0.001 0.001 0.001 \
#     --drop_feature_rate_2 0.01 0.0001 0.01 0.1 0.0001 \

# python test_NIFTY_ALL.py --dataset pokec_z \
#     --model gcn \
#     --encoder gcn \
#     --num_hidden 256 4 32 256 64 \
#     --num_proj_hidden 128 256 256 64 16 \
#     --lr 0.01 0.001 0.01 0.01 0.01 \
#     --weight_decay 0.01 0.0001 0.01 0.01 0.01 \
#     --sim_coeff 0.3 0.6 0.6 0.7 0.7 \
#     --drop_edge_rate_1 0.1 0.1 0.01 0.001 0.0001 \
#     --drop_edge_rate_2 0.001 0.0001 0.01 0.01 0.0001 \
#     --drop_feature_rate_1 0.0001 0.001 0.0001 0.01 0.1 \
#     --drop_feature_rate_2 0.1 0.1 0.1 0.0001 0.1 \


# # NIFTY GAT
# python test_NIFTY_ALL.py --dataset income \
#     --model gat \
#     --encoder gat \
#     --num_hidden 128 256 256 64 256 \
#     --num_proj_hidden 32 4 32 32 4 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.0001 0.0001 0.001 0.001 0.0001 \
#     --sim_coeff 0.5 0.4 0.3 0.6 0.3 \
#     --drop_edge_rate_1 0.001 0.1 0.001 0.01 0.0001 \
#     --drop_edge_rate_2 0.1 0.01 0.0001 0.01 0.0001 \
#     --drop_feature_rate_1 0.01 0.0001 0.01 0.001 0.001 \
#     --drop_feature_rate_2 0.001 0.1 0.001 0.001 0.1\ 

# python test_NIFTY_ALL.py --dataset nba \
#     --model gat \
#     --encoder gat \
#     --num_hidden 256 32 128 64 64 \
#     --num_proj_hidden 128 128 64 64 64 \
#     --lr 0.001 0.001 0.001 0.001 0.001 \
#     --weight_decay 0.000001 0.0001 0.01 0.01 0.01 \
#     --sim_coeff 0.3 0.5 0.3 0.3 0.5  \
#     --drop_edge_rate_1 0.0001 0.0001 0.1 0.1 0.01 \
#     --drop_edge_rate_2 0.0001 0.01 0.01 0.01 0.001 \
#     --drop_feature_rate_1 0.01 0.01 0.0001 0.01 0.01 \
#     --drop_feature_rate_2 0.001 0.001 0.1 0.01 0.01\ 


# python test_NIFTY_ALL.py --dataset bail \
#     --model gat \
#     --encoder gat \
#     --num_hidden 32 64 16 128 16 \
#     --num_proj_hidden 256 32 64 128 256 \
#     --lr 0.01 0.001 0.01 0.01 0.01\
#     --weight_decay 0.00001 0.01 0.0001 0.0001 0.0001 \
#     --sim_coeff 0.5 0.3 0.5 0.7 0.3 \
#     --drop_edge_rate_1 0.1 0.001 0.001 0.001 0.01 \
#     --drop_edge_rate_2 0.001 0.1 0.01 0.001 0.001 \
#     --drop_feature_rate_1 0.0001 0.0001 0.001 0.01 0.1 \
#     --drop_feature_rate_2 0.0001 0.001 0.01 0.001 0.01\ 

# python test_NIFTY_ALL.py --dataset pokec_n \
#     --model gat \
#     --encoder gat \
#     --num_hidden 256 256 4 16 64 \
#     --num_proj_hidden 16 32 32 128 256 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.01 0.01 0.01 0.00001 0.00001 \
#     --sim_coeff 0.7 0.6 0.4 0.3 0.4 \
#     --drop_edge_rate_1 0.1 0.01 0.001 0.001 0.001 \
#     --drop_edge_rate_2 0.001 0.001 0.0001 0.001 0.001 \
#     --drop_feature_rate_1 0.01 0.0001 0.0001 0.1 0.01 \
#     --drop_feature_rate_2 0.1 0.0001 0.01 0.0001 0.001\ 

# python test_NIFTY_ALL.py --dataset pokec_z \
#     --model gat \
#     --encoder gat \
#     --num_hidden 16 64 32 64 16 \
#     --num_proj_hidden 16 32 16 64 128 \
#     --lr 0.01 0.0001 0.001 0.01 0.01 \
#     --weight_decay 0.000001 0.01 0.00001 0.01 0.000001 \
#     --sim_coeff 0.7 0.4 0.5 0.6 0.7 \
#     --drop_edge_rate_1 0.1 0.1 0.0001 0.001 0.0001 \
#     --drop_edge_rate_2 0.01 0.001 0.0001 0.01 0.01 \
#     --drop_feature_rate_1 0.01 0.001 0.01 0.01 0.01 \
#     --drop_feature_rate_2 0.001 0.0001 0.001 0.01 0.01\ 

# # NIFTY SAGE
# python test_NIFTY_ALL.py --dataset nba \
#     --model sage \
#     --encoder sage \
#     --num_hidden 256 128 4 256 16 \
#     --num_proj_hidden 4 64 16 128 32 \
#     --lr 0.01 0.001 0.01 0.01 0.001 \
#     --weight_decay 0.0001 0.0001 0.00001 0.001 0.000001 \
#     --sim_coeff 0.3 0.4 0.6 0.5 0.4 \
#     --drop_edge_rate_1 0.001 0.001 0.001 0.0001 0.0001 \
#     --drop_edge_rate_2 0.1 0.001 0.001 0.0001 0.01 \
#     --drop_feature_rate_1 0.01 0.0001 0.01 0.1 0.01 \
#     --drop_feature_rate_2 0.001 0.001 0.001 0.0001 0.001\

# python test_NIFTY_ALL.py --dataset income \
#     --model sage \
#     --encoder sage \
#     --num_hidden 256 256 64 128 256 \
#     --num_proj_hidden 256 16 128 128 32 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.001 0.001 0.01 0.01 0.01 \
#     --sim_coeff 0.4 0.5 0.3 0.3 0.4 \
#     --drop_edge_rate_1 0.0001 0.01 0.01 0.001 0.01 \
#     --drop_edge_rate_2 0.0001 0.01 0.01 0.0001 0.01 \
#     --drop_feature_rate_1 0.001 0.0001 0.1 0.001 0.0001 \
#     --drop_feature_rate_2 0.01 0.001 0.01 0.01 0.0001\

# python test_NIFTY_ALL.py --dataset bail \
#     --model sage \
#     --encoder sage \
#     --num_hidden 4 64 4 4 64 \
#     --num_proj_hidden 32 4 32 4 32 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.00001 0.01 0.001 0.01 0.001 \
#     --sim_coeff 0.3 0.7 0.5 0.3 0.6 \
#     --drop_edge_rate_1 0.1 0.001 0.1 0.01 0.0001 \
#     --drop_edge_rate_2 0.001 0.01 0.1 0.1 0.1\
#     --drop_feature_rate_1 0.1 0.1 0.0001 0.0001 0.01 \
#     --drop_feature_rate_2 0.01 0.01 0.01 0.0001 0.01\

# python test_NIFTY_ALL.py --dataset pokec_n \
#     --model sage \
#     --encoder sage \
#     --num_hidden 256 16 32 128 128 \
#     --num_proj_hidden 4 16 4 128 32 \
#     --lr 0.01 0.001 0.01 0.01 0.01 \
#     --weight_decay 0.01 0.001 0.01 0.00001 0.0001 \
#     --sim_coeff 0.7 0.5 0.7 0.6 0.7 \
#     --drop_edge_rate_1 0.001 0.01 0.01 0.01 0.0001 \
#     --drop_edge_rate_2 0.0001 0.0001 0.01 0.1 0.0001 \
#     --drop_feature_rate_1 0.001 0.01 0.0001 0.001 0.0001 \
#     --drop_feature_rate_2 0.001 0.001 0.0001 0.0001 0.01\

# python test_NIFTY_ALL.py --dataset pokec_z \
#     --model sage \
#     --encoder sage \
#     --num_hidden 256 32 256 32 32 \
#     --num_proj_hidden 128 64 32 64 4 \
#     --lr 0.0001 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.01 0.01 0.000001 0.0001 0.001 \
#     --sim_coeff 0.7 0.7 0.6 0.7 0.6 \
#     --drop_edge_rate_1 0.1 0.0001 0.0001 0.0001 0.0001 \
#     --drop_edge_rate_2 0.1 0.0001 0.01 0.0001 0.001\
#     --drop_feature_rate_1 0.0001 0.1 0.0001 0.0001 0.0001 \
#     --drop_feature_rate_2 0.1 0.01 0.0001 0.01 0.0001\  


# # FairGAT
# python test_fairGNN_ALL.py --dataset nba \
#     --model gat \
#     --num_hidden 128 256 128 16 128 \
#     --sim_coeff 0.7 0.3 0.3 0.7 0.6 \
#     --acc 0.5 0.2 0.3 0.3 0.69 \
#     --alpha 3 1 2 6 3 \
#     --beta 0.001 0.01 0.0001 0.001 0.01 \
#     --proj_hidden 64 4 8 16 64 \
#     --lr 0.0001 0.0001 0.001 0.001 0.0001 \
#     --weight_decay 0.0001 0.01 0.0001 0.001 0.00001 

# python test_fairGNN_ALL.py --dataset pokec_n \
#     --model gat \
#     --num_hidden 128 16 256 128 128 \
#     --sim_coeff 0.5 0.5 0.5 0.6 0.5 \
#     --acc 0.2 0.6 0.4 0.4 0.2 \
#     --alpha 5 5 100 2 40 \
#     --beta 0.01 0.0001 0.0001 0.01 0.1 \
#     --proj_hidden 16 8 4 128 128 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.0001 0.00001 0.00001 0.00001 0.00001

# python test_fairGNN_ALL.py --dataset pokec_z \
#     --model gat \
#     --num_hidden 256 64 128 256 64 \
#     --sim_coeff 0.7 0.6 0.6 0.5 0.3 \
#     --acc 0.3 0.5 0.6 0.5 0.4 \
#     --alpha 40 100 2 100 100 \
#     --beta 0.001 0.01 1 0.0001 1 \
#     --proj_hidden 8 8 128 8 64 \
#     --lr 0.001 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.0001 0.00001 0.001 0.01 0.01

# python test_fairGNN_ALL.py --dataset income \
#     --model gat \
#     --num_hidden 256 128 256 128 64 \
#     --sim_coeff 0.5 0.7 0.3 0.7 0.5 \
#     --acc 0.688 0.6 0.69 0.688 0.68 \
#     --alpha 1 50 3 3 50 \
#     --beta 0.0001 0.001 0.0001 0.001 0.001 \
#     --proj_hidden 64 16 4 8 16 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.01 0.001 0.00001 0.0001 0.00001

# python test_fairGNN_ALL.py --dataset bail \
#     --model gat \
#     --num_hidden 256 256 64 256 16 \
#     --sim_coeff 0.3 0.5 0.3 0.6 0.6 \
#     --acc 0.4 0.688 0.69 0.7 0.2 \
#     --alpha 6 20 10 1 6 \
#     --beta 0.1 0.0001 0.0001 1 1 \
#     --proj_hidden 128 128 8 128 64 \
#     --lr 0.01 0.01 0.01 0.001 0.01 \
#     --weight_decay 0.0001 0.00001 0.01 0.01 0.001

# # Fair GCN
# python test_fairGNN_ALL.py --dataset nba \
#     --model gcn \
#     --num_hidden 256 64 256 16 128 \
#     --sim_coeff 0.3 0.3 0.6 0.3 0.6 \
#     --acc 0.69 0.3 0.7 0.4 0.68 \
#     --alpha 3 3 1 5 2 \
#     --beta 0.01 1 1 0.1 0.1 \
#     --proj_hidden 16 4 4 128 8 \
#     --lr 0.01 0.01 0.001 0.01 0.001 \
#     --weight_decay 0.0001 0.0001 0.001 0.01 0.01

# python test_fairGNN_ALL.py --dataset income \
#     --model gcn \
#     --num_hidden 64 256 128 64 256 \
#     --sim_coeff 0.5 0.5 0.3 0.6 0.3 \
#     --acc 0.3 0.7 0.5 0.8 0.69 \
#     --alpha 10 50 1 1 2 \
#     --beta 0.1 1 1 1 0.01 \
#     --proj_hidden 64 4 4 128 4 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.001 0.0001 0.0001 0.0001 0.00001

# python test_fairGNN_ALL.py --dataset pokec_n \
#     --model gcn \
#     --num_hidden 16 128 256 256 256 \
#     --sim_coeff 0.5 0.3 0.5 0.5 0.6 \
#     --acc 0.6 0.3 0.4 0.5 0.5 \
#     --alpha 20 20 5 20 100 \
#     --beta 0.1 0.0001 0.001 0.01 0.1 \
#     --proj_hidden 128 128 64 128 16 \
#     --lr 0.01 0.01 0.01 0.001 0.01 \
#     --weight_decay 0.001 0.00001 0.01 0.001 0.00001

# python test_fairGNN_ALL.py --dataset pokec_z \
#     --model gcn \
#     --num_hidden 256 256 64 256 128 \
#     --sim_coeff 0.7 0.3 0.5 0.3 0.3 \
#     --acc 0.6 0.5 0.3 0.5 0.3 \
#     --alpha 5 20 100 6 40 \
#     --beta 1 0.01 0.001 1 0.001 \
#     --proj_hidden 16 8 64 128 4 \
#     --lr 0.01 0.001 0.01 0.01 0.01 \
#     --weight_decay 0.0001 0.0001 0.01 0.00001 0.01

# python test_fairGNN_ALL.py --dataset bail \
#     --model gcn \
#     --num_hidden 64 256 256 128 128 \
#     --sim_coeff 0.6 0.7 0.3 0.5 0.6 \
#     --acc 0.69 0.68 0.7 0.68 0.7 \
#     --alpha 50 10 4 3 7 \
#     --beta 0.1 0.1 0.001 0.01 0.001 \
#     --proj_hidden 8 16 16 128 64 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.00001 0.0001 0.0001 0.00001 0.001

# # fairSAGE
# python test_fairGNN_ALL.py --dataset nba \
#     --model sage \
#     --num_hidden 16 64 256 16 16 \
#     --sim_coeff 0.7 0.3 0.7 0.3 0.7 \
#     --acc 0.6 0.6 0.7 0.69 0.2 \
#     --alpha 100 7 3 4 10 \
#     --beta 0.01 1 0.1 1 0.0001 \
#     --proj_hidden 4 8 64 16 128 \
#     --lr 0.01 0.0001 0.001 0.001 0.0001 \
#     --weight_decay 0.01 0.0001 0.01 0.001 0.0001

# python test_fairGNN_ALL.py --dataset income \
#     --model sage \
#     --num_hidden 128 128 64 128 256 \
#     --sim_coeff 0.3 0.3 0.3 0.5 0.3 \
#     --acc 0.8 0.4 0.5 0.7 0.69 \
#     --alpha 6 1 5 4 4 \
#     --beta 0.0001 0.1 0.01 0.01 0.01 \
#     --proj_hidden 64 64 64 128 8 \
#     --lr 0.01 0.01 0.01 0.01 0.01 \
#     --weight_decay 0.00001 0.001 0.01 0.0001 0.001

# python test_fairGNN_ALL.py --dataset bail \
#     --model sage \
#     --num_hidden 256 256 128 256 256 \
#     --sim_coeff 0.6 0.6 0.6 0.7 0.6 \
#     --acc 0.6 0.8 0.69 0.6 0.5 \
#     --alpha 5 10 6 1 6 \
#     --beta 1 1 0.0001 1 0.01 \
#     --proj_hidden 64 8 8 4 8 \
#     --lr 0.01 0.01 0.01 0.01 0.001 \
#     --weight_decay 0.01 0.01 0.001 0.01 0.01

# python test_fairGNN_ALL.py --dataset pokec_z \
#     --model sage \
#     --num_hidden 128 256 128 128 64 \
#     --sim_coeff 0.3 0.7 0.5 0.5 0.7 \
#     --acc 0.4 0.2 0.2 0.5 0.2 \
#     --alpha 7 40 4 6 8 \
#     --beta 0.01 1 0.001 0.001 0.01 \
#     --proj_hidden 128 4 8 128 8 \
#     --lr 0.01 0.01 0.001 0.01 0.01 \
#     --weight_decay 0.01 0.01 0.001 0.001 0.01

# python test_fairGNN_ALL.py --dataset pokec_n \
#     --model sage \
#     --num_hidden 128 256 256 128 256 \
#     --sim_coeff 0.7 0.6 0.3 0.6 0.7 \
#     --acc 0.4 0.2 0.6 0.5 0.3 \
#     --alpha 10 20 10 20 10 \
#     --beta 0.1 0.0001 1 0.01 0.0001 \
#     --proj_hidden 8 8 64 8 4 \
#     --lr 0.01 0.001 0.01 0.01 0.001 \
#     --weight_decay 0.00001 0.001 0.001 0.01 0.01

# SAGE
python test_VGCN.py --dataset income \
    --model sage \
    --encoder sage\
    --num_hidden 4 16 256 256 256 \
    --num_proj_hidden 256 4 128 16 4 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.05 0.05 0.002 0.002 0.002 \
    --sim_coeff 0.5 0.3 0.3 0.7 0.3\ 

python test_VGCN.py --dataset nba \
    --model sage \
    --encoder sage\
    --num_hidden 256 256 16 256 16 \
    --num_proj_hidden 128 256 64 64 128 \
    --lr 0.0001 0.001 0.001 0.0001 0.001 \
    --weight_decay 0.01 0.05 0.01 0.00001 0.05 \
    --sim_coeff 0.5 0.5 0.7 0.7 0.5\ 

python test_VGCN.py --dataset bail \
    --model sage \
    --encoder sage\
    --num_hidden 64 16 128 4 4\
    --num_proj_hidden 256 64 4 16 128\
    --lr 0.001 0.001 0.001 0.01 0.001\
    --weight_decay 0.01 0.01 0.002 0.05 0.01\
    --sim_coeff 0.3 0.5 0.5 0.7 0.7\ 

python test_VGCN.py --dataset pokec_n \
    --model sage \
    --encoder sage\
    --num_hidden 16 128 128 16 256 \
    --num_proj_hidden 128 64 64 128 128\
    --lr 0.01 0.01 0.01 0.01 0.01\
    --weight_decay 0.05 0.05 0.001 0.00001 0.01 \
    --sim_coeff 0.7 0.3 0.7 0.3 0.7\ 

python test_VGCN.py --dataset pokec_z \
    --model sage \
    --encoder sage\
    --num_hidden 16 4 256 16 256 \
    --num_proj_hidden 64 64 128 256 16 \
    --lr 0.01 0.01 0.01 0.01 0.001 \
    --weight_decay 0.002 0.00001 0.05 0.05 0.0001 \
    --sim_coeff 0.5 0.5 0.7 0.3 0.3\ 

# GCN
python test_VGCN.py --dataset nba \
    --model gcn \
    --encoder gcn \
    --num_hidden 64 16 256 64 256 \
    --num_proj_hidden 256 128 64 256 64 \
    --lr 0.001 0.001 0.01 0.01 0.001 \
    --weight_decay 0.0001 0.001 0.001 0.002 0.01 \
    --sim_coeff 0.7 0.7 0.5 0.3 0.3\ 

python test_VGCN.py --dataset income \
    --model gcn \
    --encoder gcn \
    --num_hidden 16 256 4 64 256 \
    --num_proj_hidden 16 4 128 4 128 \
    --lr 0.00001 0.01 0.0001 0.00001 0.01 \
    --weight_decay 0.001 0.00001 0.00001 0.00001 0.00001 \
    --sim_coeff 0.7 0.7 0.7 0.3 0.5\ 

python test_VGCN.py --dataset bail \
    --model gcn \
    --encoder gcn \
    --num_hidden 265 16 256 256 256 \
    --num_proj_hidden 16 128 16 64 4 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.0001 0.001 0.001 0.001 0.001 \
    --sim_coeff 0.7 0.3 0.3 0.5 0.7\ 

python test_VGCN.py --dataset pokec_z \
    --model gcn \
    --encoder gcn \
    --num_hidden 128 128 64 128 64 \
    --num_proj_hidden 64 256 128 128 256 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.05 0.002 0.05 0.002 0.002 \
    --sim_coeff 0.3 0.7 0.3 0.3 0.7\ 

python test_VGCN.py --dataset pokec_n \
    --model gcn \
    --encoder gcn \
    --num_hidden 4 64 128 16 128 \
    --num_proj_hidden 256 4 4 64 64 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.05 0.002 0.05 0.05 0.05 \
    --sim_coeff 0.3 0.5 0.7 0.5 0.3\ 

# GAT
python test_VGCN.py --dataset nba \
    --model gat \
    --encoder gat \
    --num_hidden 64 128 16 64 64 \
    --num_proj_hidden 6 64 256 128 641 \
    --lr 0.001 0.001 0.001 0.001 0.001 \
    --weight_decay 0.05 0.05 0.01 0.05 0.05 \
    --sim_coeff 0.3 0.3 0.5 0.7 0.3\ 

python test_VGCN.py --dataset income \
    --model gat \
    --encoder gat \
    --num_hidden 64 256 256 256 256 \
    --num_proj_hidden 16 16 128 16 4 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.001 0.001 0.002 0.002 0.002 \
    --sim_coeff 0.5 0.3 0.3 0.7 0.3\ 

python test_VGCN.py --dataset pokec_z \
    --model gat \
    --encoder gat \
    --num_hidden 16 4 256 16 128 \
    --num_proj_hidden 64 64 128 256 128 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.0002 0.00001 0.05 0.05 0.05 \
    --sim_coeff 0.5 0.5 0.7 0.3 0.3\ 

python test_VGCN.py --dataset pokec_n \
    --model gat \
    --encoder gat \
    --num_hidden 256 4 16 16 64 \
    --num_proj_hidden 128 128 16 256 256 \
    --lr 0.001 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.05 0.01 0.01 0.002 \
    --sim_coeff 0.5 0.3 0.7 0.3 0.7\ 

python test_VGCN.py --dataset bail \
    --model gat \
    --encoder gat \
    --num_hidden 16 16 256 4 16 \
    --num_proj_hidden 64 16 16 256 128 \
    --lr 0.001 0.01 0.01 0.01 0.01 \
    --weight_decay 0.01 0.01 0.001 0.00001 0.01 \
    --sim_coeff 0.5 0.3 0.3 0.3 0.5\ 

# MLP
python test_MLP.py --dataset nba \
    --num_hidden 128 128 128 64 256 \
    --num_proj_hidden 256 4 16 256 4 \
    --lr 0.0001 0.01 0.01 0.001 0.001 \
    --weight_decay 0.05 0.01 0.05 0.0001 0.01 \
    --sim_coeff 0.5 0.5 0.7 0.5 0.3\ 

python test_MLP.py --dataset income \
    --num_hidden 64 256 128 128 256 \
    --num_proj_hidden 128 256 16 256 16 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.0001 0.002 0.00001 0.00001 0.002 \
    --sim_coeff 0.7 0.3 0.7 0.7 0.5\

python test_MLP.py --dataset bail \
    --num_hidden 4 4 16 4 16 \
    --num_proj_hidden 4 4 16 64 64 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.001 0.001 0.002 0.002 0.002 \
    --sim_coeff 0.3 0.7 0.3 0.5 0.5\ 

python test_MLP.py --dataset pokec_n \
    --num_hidden 256 256 256 128 256 \
    --num_proj_hidden 4 16 16 256 128 \
    --lr 0.01 0.01 0.01 0.0001 0.01 \
    --weight_decay 0.05 0.05 0.001 0.05 0.05 \
    --sim_coeff 0.7 0.7 0.5 0.7 0.7\

python test_MLP.py --dataset pokec_z \
    --num_hidden 64 256 256 256 256 \
    --num_proj_hidden 64 128 16 256 64 \
    --lr 0.001 0.01 0.001 0.0001 0.01 \
    --weight_decay 0.05 0.05 0.05 0.05 0.05 \
    --sim_coeff 0.7 0.3 0.5 0.7 0.5\  

