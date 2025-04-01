from models.base_model import BaseModel
from models.e_spa_tune import ESPATune


class SPATune(BaseModel):
    def __init__(self, args, dataset_info_dict, device, genotype, s2o, node_wise_feature, batch_pair_num):
        super(SPATune, self).__init__(args, dataset_info_dict, device)
        self.ent_encoder = ESPATune(genotype, self.args, self.args.embed_size, self.args.hidden_size,
                                              self.args.embed_size,
                                              self.num_rels, s2o, node_wise_feature,batch_pair_num)