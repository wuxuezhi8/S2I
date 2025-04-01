from models.base_model import BaseModel
from models.e_spa_darts import ESPADartsSearch


class SPADartsSearch(BaseModel):
    def __init__(self, args, data, device, s2o, node_wise_feature, batch_pair_num):
        super(SPADartsSearch, self).__init__(args, data, device)
        self.ent_encoder = ESPADartsSearch(self.args, self.args.embed_size, self.args.hidden_size,
                                           self.args.embed_size,
                                           self.num_rels, s2o, node_wise_feature, batch_pair_num)
    # def new(self):
    #     model_new = SPADartsSearch(self.args, self.data, self.device).cuda()
    #     for x, y in zip(model_new.ent_encoder.arch_parameters(), self.ent_encoder.arch_parameters()):
    #         x.data.copy_(y.data)
    #     return model_new