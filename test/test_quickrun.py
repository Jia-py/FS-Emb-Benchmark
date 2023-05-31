import sys
sys.path.append("/root/code/FS-Emb-Benchmark")
from recstudio import quickstart
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DCN', help='model name: DCN, PNN, AutoInt, FM, DeepFM, WideDeep, xDeepFM')
parser.add_argument('--dataset', type=str, default='avazu', help='dataset name: ml-100k, ml-1m, yelp')
parser.add_argument('--feature_selection_method', type=str, default='No_Selection', help='feature selection method, AutoField, AdaFS, Lasso, LPFS, optFS, GBDT, No_Selection')
parser.add_argument('--run_mode', type=str, default='light', help='tune, light')
args = parser.parse_args()

data_config_path = f"/root/code/FS-Emb-Benchmark/recstudio/data/config/{args.dataset}.yaml"

quickstart.run(model=args.model, dataset=args.dataset, data_config_path=data_config_path, feature_selection_method=args.feature_selection_method, run_mode=args.run_mode)