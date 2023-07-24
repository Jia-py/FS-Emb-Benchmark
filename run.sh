
# feature_selection_method=optFS
python test/test_quickrun.py --model=DCN --dataset=yelp --feature_selection_method=optFS >log/DCN-yelp-optFS.log 2>&1
python test/test_quickrun.py --model=PNN --dataset=yelp --feature_selection_method=optFS >log/PNN-yelp-optFS.log 2>&1
python test/test_quickrun.py --model=AutoInt --dataset=yelp --feature_selection_method=optFS >log/AutoInt-yelp-optFS.log 2>&1
python test/test_quickrun.py --model=FM --dataset=yelp --feature_selection_method=optFS >log/FM-yelp-optFS.log 2>&1
python test/test_quickrun.py --model=DeepFM --dataset=yelp --feature_selection_method=optFS >log/DeepFM-yelp-optFS.log 2>&1
python test/test_quickrun.py --model=WideDeep --dataset=yelp --feature_selection_method=optFS >log/WideDeep-yelp-optFS.log 2>&1
python test/test_quickrun.py --model=xDeepFM --dataset=yelp --feature_selection_method=optFS >log/xDeepFM-yelp-optFS.log 2>&1