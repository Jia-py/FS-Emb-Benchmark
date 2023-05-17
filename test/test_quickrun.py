import sys
import nni
sys.path.append("/root/code/FS-Emb-Benchmark")
from recstudio import quickstart

params = nni.get_next_parameter()
quickstart.run(model='DCN', dataset='ml-100k', feature_selection_method='AutoField', use_nni=True, nni_params=params)