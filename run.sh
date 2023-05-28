python test/test_quickrun.py --model=DCN --dataset=yelp --feature_selection_method=AutoField >log/DCN-yelp-AutoField.log 2>&1
python test/test_quickrun.py --model=PNN --dataset=yelp --feature_selection_method=AutoField >log/PNN-yelp-AutoField.log 2>&1
# model AutoInt
python test/test_quickrun.py --model=AutoInt --dataset=yelp --feature_selection_method=AutoField >log/AutoInt-yelp-AutoField.log 2>&1
# FM
python test/test_quickrun.py --model=FM --dataset=yelp --feature_selection_method=AutoField >log/FM-yelp-AutoField.log 2>&1
# DeepFM
python test/test_quickrun.py --model=DeepFM --dataset=yelp --feature_selection_method=AutoField >log/DeepFM-yelp-AutoField.log 2>&1
# WideDeep
python test/test_quickrun.py --model=WideDeep --dataset=yelp --feature_selection_method=AutoField >log/WideDeep-yelp-AutoField.log 2>&1
# xDeepFM
python test/test_quickrun.py --model=xDeepFM --dataset=yelp --feature_selection_method=AutoField >log/xDeepFM-yelp-AutoField.log 2>&1

python test/test_quickrun.py --model=DCN --dataset=yelp --feature_selection_method=AdaFS >log/DCN-yelp-AdaFS.log 2>&1
python test/test_quickrun.py --model=PNN --dataset=yelp --feature_selection_method=AdaFS >log/PNN-yelp-AdaFS.log 2>&1
# model AutoInt
python test/test_quickrun.py --model=AutoInt --dataset=yelp --feature_selection_method=AdaFS >log/AutoInt-yelp-AdaFS.log 2>&1
# FM
python test/test_quickrun.py --model=FM --dataset=yelp --feature_selection_method=AdaFS >log/FM-yelp-AdaFS.log 2>&1
# DeepFM
python test/test_quickrun.py --model=DeepFM --dataset=yelp --feature_selection_method=AdaFS >log/DeepFM-yelp-AdaFS.log 2>&1
# WideDeep
python test/test_quickrun.py --model=WideDeep --dataset=yelp --feature_selection_method=AdaFS >log/WideDeep-yelp-AdaFS.log 2>&1
# xDeepFM
python test/test_quickrun.py --model=xDeepFM --dataset=yelp --feature_selection_method=AdaFS >log/xDeepFM-yelp-AdaFS.log 2>&1

# feature_selection_method=Lasso
python test/test_quickrun.py --model=DCN --dataset=yelp --feature_selection_method=Lasso >log/DCN-yelp-Lasso.log 2>&1
python test/test_quickrun.py --model=PNN --dataset=yelp --feature_selection_method=Lasso >log/PNN-yelp-Lasso.log 2>&1
python test/test_quickrun.py --model=AutoInt --dataset=yelp --feature_selection_method=Lasso >log/AutoInt-yelp-Lasso.log 2>&1
python test/test_quickrun.py --model=FM --dataset=yelp --feature_selection_method=Lasso >log/FM-yelp-Lasso.log 2>&1
python test/test_quickrun.py --model=DeepFM --dataset=yelp --feature_selection_method=Lasso >log/DeepFM-yelp-Lasso.log 2>&1
python test/test_quickrun.py --model=WideDeep --dataset=yelp --feature_selection_method=Lasso >log/WideDeep-yelp-Lasso.log 2>&1
python test/test_quickrun.py --model=xDeepFM --dataset=yelp --feature_selection_method=Lasso >log/xDeepFM-yelp-Lasso.log 2>&1

# feature_selecion_method = LPFS
python test/test_quickrun.py --model=DCN --dataset=yelp --feature_selection_method=LPFS >log/DCN-yelp-LPFS.log 2>&1
python test/test_quickrun.py --model=PNN --dataset=yelp --feature_selection_method=LPFS >log/PNN-yelp-LPFS.log 2>&1
python test/test_quickrun.py --model=AutoInt --dataset=yelp --feature_selection_method=LPFS >log/AutoInt-yelp-LPFS.log 2>&1
python test/test_quickrun.py --model=FM --dataset=yelp --feature_selection_method=LPFS >log/FM-yelp-LPFS.log 2>&1
python test/test_quickrun.py --model=DeepFM --dataset=yelp --feature_selection_method=LPFS >log/DeepFM-yelp-LPFS.log 2>&1
python test/test_quickrun.py --model=WideDeep --dataset=yelp --feature_selection_method=LPFS >log/WideDeep-yelp-LPFS.log 2>&1
python test/test_quickrun.py --model=xDeepFM --dataset=yelp --feature_selection_method=LPFS >log/xDeepFM-yelp-LPFS.log 2>&1

# feature_selecion_method = GBDT
python test/test_quickrun.py --model=DCN --dataset=yelp --feature_selection_method=GBDT >log/DCN-yelp-GBDT.log 2>&1
python test/test_quickrun.py --model=PNN --dataset=yelp --feature_selection_method=GBDT >log/PNN-yelp-GBDT.log 2>&1
python test/test_quickrun.py --model=AutoInt --dataset=yelp --feature_selection_method=GBDT >log/AutoInt-yelp-GBDT.log 2>&1
python test/test_quickrun.py --model=FM --dataset=yelp --feature_selection_method=GBDT >log/FM-yelp-GBDT.log 2>&1
python test/test_quickrun.py --model=DeepFM --dataset=yelp --feature_selection_method=GBDT >log/DeepFM-yelp-GBDT.log 2>&1
python test/test_quickrun.py --model=WideDeep --dataset=yelp --feature_selection_method=GBDT >log/WideDeep-yelp-GBDT.log 2>&1
python test/test_quickrun.py --model=xDeepFM --dataset=yelp --feature_selection_method=GBDT >log/xDeepFM-yelp-GBDT.log 2>&1
