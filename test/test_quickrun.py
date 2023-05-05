import sys
sys.path.append("./")
from recstudio import quickstart

quickstart.run(model='DCN', dataset='ml-100k', feature_selection_method='AutoField', use_nni=True)

import recstudio.data as recdata

print(recdata.supported_dataset)