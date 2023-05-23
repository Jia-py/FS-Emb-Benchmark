from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule


class DeepFM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, use_field, drop_unused_field=False):
        super()._init_model(train_data, use_field, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.fm = ctr.FMLayer(reduction='sum')
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        self.mlp = MLPModule([self.embedding.num_features*self.embed_dim]+model_config['mlp_layer']+[1],
                             model_config['activation'], model_config['dropout'],
                             last_activation=False, last_bn=False)

    def score(self, batch):
        batch = {field: batch[field] for field in self.embedding.field2types}
        lr_score = self.linear(batch)
        emb = self.embedding(batch)
        emb = self.feature_selection_layer(emb, self.nepoch, self.fields, batch)
        fm_score = self.fm(emb)
        mlp_score = self.mlp(emb.flatten(1)).squeeze(-1)
        return {'score' : lr_score + fm_score + mlp_score}

    def _get_loss_func(self):
        return BCEWithLogitLoss()
