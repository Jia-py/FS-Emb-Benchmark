import sys
sys.path.append("/root/code/FS-Emb-Benchmark")
from recstudio import quickstart

# feature selection: AutoField, AdaFS, Lasso, LPFS, optFS, GBDT
quickstart.run(model='DCN', dataset='ml-100k', feature_selection_method='AutoField', run_mode='tune')


import recstudio.data as recdata

print(recdata.supported_dataset)