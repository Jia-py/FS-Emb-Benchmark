import torch
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule
from ..feature_selection.optfs import optFS


class DCN(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, use_field):
        super()._init_model(train_data, use_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data) # embedding的顺序是按照self.fields的顺序
        num_features = self.embedding.num_features
        model_config = self.config['model']
        mlp_layer = model_config['mlp_layer_0'] + model_config['mlp_layer_1'] + model_config['mlp_layer_2']
        self.cross_net = ctr.CrossNetwork(num_features * self.embed_dim, model_config['num_layers'])
        self.mlp = MLPModule(
                    [num_features * self.embed_dim] + mlp_layer,
                    model_config['activation'],
                    model_config['dropout'],
                    batch_norm=model_config['batch_norm'])
        self.fc = torch.nn.Linear(num_features*self.embed_dim + mlp_layer[-1], 1)

    def score(self, batch):
        emb = self.embedding(batch)
        emb = self.feature_selection_layer(emb, self.nepoch, self.fields, batch)
        emb = emb.view(*emb.shape[:-2], -1)
        cross_out = self.cross_net(emb)
        deep_out = self.mlp(emb)
        score = self.fc(torch.cat([deep_out, cross_out], -1)).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
