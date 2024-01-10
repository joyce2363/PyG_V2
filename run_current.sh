python test_fairGNN_ALL.py --dataset pokec_z --model gcn \
    --num_hidden 128 128 64 64 64\
    --sim_coeff 0.7 0.3 0.7 0.5 0.6\
    --acc 0.4 0.5 0.4 0.5 0.4\
     --alpha 5 1 6 20 5\
    --beta 0.001 0.0001 0.001 0.01 0.01\
     --proj_hidden 16 128 128 128 16\
     --lr 0.01 0.00001 0.001 0.01 0.01\
    --weight_decay 0.01 0.001 0.001 0.01 0.00001

python test_fairGNN_ALL.py --dataset bail --model gat \
    --num_hidden \
    --sim_coeff \
    --acc \
     --alpha \
    --beta \
     --proj_hidden \
     --lr \
    --weight_decay 

python test_fairGNN_ALL.py --dataset bail --model gcn \
    --num_hidden 64 128 256 256 256\
    --sim_coeff 0.3 0.5 0.3 0.5 0.3\
    --acc 0.4 0.5 0.6 0.4 0.6\
     --alpha 6 10 3 10 10\
    --beta 0.001 0.001 0.01 0.001 0.1\
     --proj_hidden 4 16 128 16 64\
     --lr 0.01 0.01 0.01 0.01 0.01\
    --weight_decay 0.00001 0.001 0.00001 0.00001 0.0001

python test_fairGNN_ALL.py --dataset pokec_z --model sage \
    --num_hidden 64 16 128 256 128\
    --sim_coeff 0.7 0.6 0.3 0.6 0.3\
    --acc 0.5 0.5 0.3 0.3 0.2\
     --alpha 10 20 3 40 6\
    --beta 0.0001 0.01 0.0001 0.0001 0.0001\
     --proj_hidden 128 4 128 8 64\
     --lr 0.01 0.01 0.01 0.01 0.01\
    --weight_decay 0.01 0.001 0.0001 0.0001 0.0001

# python test_NIFTY_GAT.py --dataset income \
#     --num_hidden 256 16 16 128 16 \
#     --num_proj_hidden 4 4 4 256 256\
#      --lr 0.01 0.01 0.0001 0.01 0.01\
#     --weight_decay 0.0001 0.0001 0.000001 0.001 0.00001\
#     --sim_coeff 0.7 0.3 0.5 0.3 0.5