import os, datetime, torch
import pandas as pd
from typing import *
from recstudio.utils import *
from recstudio.utils.utils import get_feature_selection

def run(model: str, dataset: str, model_config: Dict=None, data_config: Dict=None, model_config_path: str=None, data_config_path: str=None, verbose=True, 
        feature_selection_method='optFS', run_mode='light', **kwargs):
    model_class, model_conf = get_model(model)

    if model_config_path is not None:
        if isinstance(model_config_path, str):
            model_conf = deep_update(model_conf, parser_yaml(model_config_path))
        else:
            raise TypeError(f"expecting `model_config_path` to be str, while get {type(model_config_path)} instead.")

    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf = deep_update(model_conf, model_config)
        else:
            raise TypeError(f"expecting `model_config` to be Dict, while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf = deep_update(model_conf, kwargs)
    
    fs_class, fs_conf = get_feature_selection(feature_selection_method)
    model_conf.update(fs_conf)              # update feature selection config
    model_conf['fs']['class'] = fs_class

    log_path = f"{model}/{dataset}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
    logger = get_logger(log_path)
    torch.set_num_threads(model_conf['train']['num_threads'])

    if not verbose:
        import logging
        logger.setLevel(logging.ERROR)

    logger.info("Log saved in {}.".format(os.path.abspath(log_path)))
    if run_mode == 'tune':
        model_conf = update_config_with_nni(model_conf)
    model = model_class(model_conf)
    dataset_class = model_class._get_dataset_class()

    data_conf = {}
    if data_config_path is not None:
        if isinstance(data_config_path, str):
            # load dataset config from file
            conf = parser_yaml(data_config)
            data_conf.update(conf)
        else:
            raise TypeError(f"expecting `data_config_path` to be str, while get {type(data_config_path)} instead.")

    if data_config is not None:
        if isinstance(data_config, dict):
            # update config with given dict
            data_conf.update(data_config)
        else:
            raise TypeError(f"expecting `data_config` to be Dict, while get {type(data_config)} instead.")

    data_conf.update(model_conf['data'])    # update model-specified config

    datasets = dataset_class(name=dataset, config=data_conf).build(**model_conf['data'])
    logger.info(f"{datasets[0]}")
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    '''
    some machine learning algorithms to select features
    '''
    use_fields = None
    if feature_selection_method in ['Lasso', 'GBDT']:
        # 算上float和rating以及选择的token特征，一共k个
        k = 9
        use_fields = machine_learning_selection(datasets, feature_selection_method, k)


    val_result = model.fit(*datasets[:2], run_mode=run_mode, use_fields=use_fields) # 在fit函数里面才会调用model._init_model()函数
    test_result = model.evaluate(datasets[-1])
    return (model, datasets), (val_result, test_result)


def machine_learning_selection(datasets, feature_selection_method, k):
    all_data = datasets[0].inter_feat.data
    uid, iid = all_data[datasets[0].fuid], all_data[datasets[0].fiid]
    all_data.update(datasets[0].user_feat[uid])
    all_data.update(datasets[0].item_feat[iid])
    data = None
    for field in all_data:
        if data is None:
            data = all_data[field].reshape(-1,1)
        else:
            data = torch.cat([data, all_data[field].reshape(-1,1)], dim=1)
    print(data.shape)
    fields = list(all_data.keys())
    df = pd.DataFrame(data.numpy(), columns=fields)
    if feature_selection_method == 'Lasso':
        from sklearn.linear_model import Lasso
        lasso = Lasso()
        lasso.fit(df[[field for field in fields if field != datasets[0].frating]], df[datasets[0].frating])
        fields_lis = [field for field in fields if field != datasets[0].frating]
        field_importance = abs(lasso.coef_)
        # 取出importance最大的k个field
        field_importance = field_importance.argsort()[::-1]
        # use_fields = [datasets[0].frating, datasets[0].fuid, datasets[0].fiid]
        use_fields = [datasets[0].frating]
        tmp_num = len(use_fields)
        for i in field_importance:
            # if fields_lis[i] in [datasets[0].fuid, datasets[0].fiid]:
            #     continue
            # elif tmp_num < k:
            if tmp_num < k:
                use_fields.append(fields_lis[i])
                tmp_num += 1
            else:
                break
        print('************************')
        print('use_fields:', use_fields)
        print('************************')
        return use_fields
    elif feature_selection_method == 'GBDT':
        from sklearn.ensemble import GradientBoostingRegressor
        gbdt = GradientBoostingRegressor()
        gbdt.fit(df[[field for field in fields if field != datasets[0].frating]], df[datasets[0].frating])
        fields_lis = [field for field in fields if field != datasets[0].frating]
        field_importance = abs(gbdt.feature_importances_)
        # 取出importance最大的k个field
        field_importance = field_importance.argsort()[::-1]
        # use_fields = [datasets[0].frating, datasets[0].fuid, datasets[0].fiid]
        use_fields = [datasets[0].frating]
        tmp_num = len(use_fields)
        for i in field_importance:
            # if fields_lis[i] in [datasets[0].fuid, datasets[0].fiid]:
            #     continue
            # elif tmp_num < k:
            if tmp_num < k:
                use_fields.append(fields_lis[i])
                tmp_num += 1
            else:
                break
        print('************************')
        print('use_fields:', use_fields)
        print('************************')
        return use_fields
