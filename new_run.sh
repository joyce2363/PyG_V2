
# NIFTY GAT
python test_NIFTY_ALL.py --dataset income \
    --model gat \
    --num_hidden 64 64 256 256 32 \
    --num_proj_hidden 4 16 32 4 128 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.0001 0.001 0.0001 0.0001 0.00001 \
    --sim_coeff 0.3 0.3 0.4 0.3 0.3 \
    --drop_edge_rate_1 0.001 0.1 0.001 0.1 0.0001 \
    --drop_edge_rate_2 0.001 0.001 0.001 0.0001 0.001 \
    --drop_feature_rate_1 0.0001 0.001 0.0001 0.0001 0.0001 \
    --drop_feature_rate_2 0.001 0.001 0.001 0.1 0.1 \

python test_NIFTY_ALL.py --dataset nba \
    --model gat \
    --num_hidden 256 256 128 32 256 \
    --num_proj_hidden 128 128 64 16 128 \
    --lr 0.001 0.0001 0.001 0.001 0.0001 \
    --weight_decay 0.01 0.001 0.0001 0.01 0.0001 \
    --sim_coeff 0.3 0.4 0.4 0.3 0.6 \
    --drop_edge_rate_1 0.1 0.0001 0.0001 0.1 0.0001 \
    --drop_edge_rate_2 0.001 0.1 0.001 0.001 0.001 \
    --drop_feature_rate_1 0.0001 0.001 0.001 0.001 0.0001 \
    --drop_feature_rate_2 0.0001 0.001 0.1 0.001 0.1 \


python test_NIFTY_ALL.py --dataset pokec_n \
    --model gat \
    --num_hidden 128 16 256 16 16 \
    --num_proj_hidden 128 128 4 128 16 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.000001 0.01 0.01 0.0001 0.0001 \
    --sim_coeff 0.5 0.6 0.4 0.6 0.5 \
    --drop_edge_rate_1 0.0001 0.0001 0.001 0.001 0.0001 \
    --drop_edge_rate_2 0.001 0.1 0.0001 0.0001 0.0001 \
    --drop_feature_rate_1 0.0001 0.0001 0.0001 0.001 0.001 \
    --drop_feature_rate_2 0.001 0.1 0.1 0.0001 0.0001 \

python test_NIFTY_ALL.py --dataset bail \
    --model gat \
    --num_hidden 64 16 32 256 16 \
    --num_proj_hidden 64 64 32 256 32 \
    --lr 0.01 0.01 0.01 0.01 0.01 \
    --weight_decay 0.00001 0.0001 0.01 0.00001 0.001 \
    --sim_coeff 0.7 0.5 0.3 0.5 0.7 \
    --drop_edge_rate_1 0.001 0.1 0.0001 0.001 0.1 \
    --drop_edge_rate_2 0.1 0.001 0.1 0.0001 0.1 \
    --drop_feature_rate_1 0.001 0.001 0.0001 0.001 0.0001 \
    --drop_feature_rate_2 0.001 0.001 0.001 0.0001 0.1 \

python test_NIFTY_ALL.py --dataset pokec_z \
    --model gat \
    --num_hidden 256 256 32 256 128 \
    --num_proj_hidden 64 256 128 32 4 \
    --lr 0.01 0.01 0.01 0.01 0.001 \
    --weight_decay 0.01 0.01 0.0001 0.01 0.00001 \
    --sim_coeff 0.5 0.6 0.4 0.5 0.5 \
    --drop_edge_rate_1 0.0001 0.1 0.0001 0.1 0.001 \
    --drop_edge_rate_2 0.0001 0.0001 0.001 0.1 0.001 \
    --drop_feature_rate_1 0.0001 0.001 0.1 0.0001 0.1 \
    --drop_feature_rate_2 0.0001 0.1 0.1 0.1 0.0001 \

# NIFTY SAGE
python test_NIFTY_ALL.py --dataset bail \
    --model sage \
    --num_hidden 16 4 4 16 16 \
    --num_proj_hidden 4 256 64 256 128 \
    --lr 0.001 0.01 0.01 0.01 0.01 \
    --weight_decay 0.001 0.00001 0.01 0.00001 0.0001 \
    --sim_coeff 0.5 0.4 0.5 0.3 0.7 \
    --drop_edge_rate_1 0.001 0.0001 0.001 0.0001 0.001 \
    --drop_edge_rate_2 0.001 0.001 0.001 0.001 0.1 \
    --drop_feature_rate_1 0.0001 0.001 0.0001 0.001 0.0001 \
    --drop_feature_rate_2 0.0001 0.001 0.001 0.001 0.0001 \

python test_NIFTY_ALL.py --dataset pokec_n \
    --model sage \
    --num_hidden 32 16 64 256 256 \
    --num_proj_hidden 32 64 128 16 32 \
    --lr 0.01 0.01 0.01 0.001 0.001 \
    --weight_decay 0.00001 0.0001 0.00001 0.0001 0.0001 \
    --sim_coeff 0.6 0.6 0.7 0.6 0.7 \
    --drop_edge_rate_1 0.001 0.001 0.0001 0.001 0.1 \
    --drop_edge_rate_2 0.1 0.1 0.0001 0.0001 0.1 \
    --drop_feature_rate_1 0.001 0.1 0.1 0.0001 0.001 \
    --drop_feature_rate_2 0.1 0.0001 0.1 0.1 0.1 \

python test_NIFTY_ALL.py --dataset pokec_z \
    --model sage \
    --num_hidden 256 16 256 128 32 \
    --num_proj_hidden 128 128 256 64 16 \
    --lr 0.01 0.01 0.01 0.001 0.01 \
    --weight_decay 0.01 0.00001 0.0001 0.0001 0.01 \
    --sim_coeff 0.5 0.6 0.7 0.7 0.7 \
    --drop_edge_rate_1 0.001 0.0001 0.0001 0.1 0.1 \
    --drop_edge_rate_2 0.1 0.001 0.0001 0.1 0.1 \
    --drop_feature_rate_1 0.1 0.001 0.001 0.0001 0.0001 \
    --drop_feature_rate_2 0.001 0.1 0.001 0.1 0.0001 \  


