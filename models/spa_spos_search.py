from models.base_model import BaseModel
from models.e_spa_spos_search import ESPASPOSSearch


class SPASPOSSearch(BaseModel):
    def __init__(self, args, data, device, s2o, node_wise_feature, batch_pair_num):
        super(SPASPOSSearch, self).__init__(args, data, device)
        self.ent_encoder = ESPASPOSSearch(self.args, self.args.embed_size, self.args.hidden_size,
                                           self.args.embed_size,
                                           self.num_rels, s2o, node_wise_feature, batch_pair_num)
