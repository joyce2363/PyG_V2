

# python test_VGCN.py --dataset income
# python test_VGCN.py --dataset bail
# python test_VGCN.py --dataset pokec_z
# python test_VGCN.py --dataset pokec_n
# python test_VGCN.py --dataset nba

python test_MLP.py --dataset income
python test_MLP.py --dataset bail
python test_MLP.py --dataset pokec_z
python test_MLP.py --dataset pokec_n
python test_MLP.py --dataset nba

python test_fairGNN_2.py --dataset income
python test_fairGNN_2.py --dataset bail
python test_fairGNN_2.py --dataset pokec_z
python test_fairGNN_2.py --dataset pokec_n
python test_fairGNN_2.py --dataset nba
# ./run.sh | tee output.txt