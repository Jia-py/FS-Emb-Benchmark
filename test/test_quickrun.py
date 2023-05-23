import sys
sys.path.append("/root/code/FS-Emb-Benchmark")
from recstudio import quickstart
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DeepFM', help='model name: DCN, PNN, AutoInt, FM, DeepFM, WideDeep, xDeepFM')
parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset name: ml-100k, ml-1m')
parser.add_argument('--feature_selection_method', type=str, default='AutoField', help='feature selection method, AutoField, AdaFS, Lasso, LPFS, optFS, GBDT, No_Selection')
parser.add_argument('--run_mode', type=str, default='tune', help='tune, light')
args = parser.parse_args()

quickstart.run(model=args.model, dataset=args.dataset, feature_selection_method=args.feature_selection_method, run_mode=args.run_mode)