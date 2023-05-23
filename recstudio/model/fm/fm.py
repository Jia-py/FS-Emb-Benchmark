import torch
from collections import OrderedDict
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr


class FM(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, use_field, drop_unused_field=False):
        super()._init_model(train_data, use_field, drop_unused_field)
        # self.fm = torch.nn.Sequential(OrderedDict([
        #     ("embeddings", ctr.Embeddings(
        #         fields=self.fields,
        #         embed_dim=self.embed_dim,
        #         data=train_data)),
        #     ("fm_layer", ctr.FMLayer(reduction='sum')),
        # ]))
        self.linear = ctr.LinearLayer(self.fields, train_data)

        self.embedding = ctr.Embeddings(
                fields=self.fields,
                embed_dim=self.embed_dim,
                data=train_data)
        self.fm_layer = ctr.FMLayer(reduction='sum')

    def score(self, batch):
        batch = {field: batch[field] for field in self.embedding.field2types}
        emb = self.embedding(batch)
        emb = self.feature_selection_layer(emb, self.nepoch, self.fields, batch)
        fm_score = self.fm_layer(emb)
        lr_score = self.linear(batch)
        return {'score' : fm_score + lr_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
