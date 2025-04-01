from knowledge_graph import load_data
from process_data import process, TrainDataset, TestDataset
from collections import OrderedDict
from json import dump, load
from torch.utils.data import DataLoader
from utils import initialize_seed
from trainer import Trainer
from time import strftime
from os import mkdir, makedirs
from os.path import exists
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Search to Integrate Multi-level Heuristics with Graph Neural Networks for Multi-relational Link Prediction")
    parser.add_argument("--dataset", type=str, default="WN18RR")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of processes to construct batches')
    parser.add_argument("--train_mode", type=str, default="search",
                        choices=["search", "tune", "train", "debug"])
    parser.add_argument("--search_mode", type=str, default="random",
                        choices=["random", "spos", "spos_search","darts_valid_loss", "darts_train_loss"])
    parser.add_argument("--encoder", type=str, default="SPATune")
    parser.add_argument("--score_function", type=str, default="complex")
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--embed_size", type=int, default=200)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--random_seed", type=int, default=22)
    parser.add_argument("--gnn_layer_num", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gcn_drop", type=float, default=0.3)
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',
                        type=float, default=0.1, help='Label Smoothing')
    # base vector config
    parser.add_argument("--base_num", type=int, default=0)
    # RGAT config
    parser.add_argument("--head_num", type=int, default=4)
    # CompGCN config
    parser.add_argument("--comp_op", type=str, default="corr")
    parser.add_argument("--sampled_dataset", type=bool, default=False)
    # Optimizer config
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    # parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # parser.add_argument("--weight_decay", type=float, default=5e-4)
    # search config
    parser.add_argument("--baseline_sample_num", type=int, default=30)
    parser.add_argument("--search_run_num", type=int, default=1)
    parser.add_argument("--search_max_epoch", type=int, default=800)
    parser.add_argument("--min_learning_rate", type=float, default=0.001)
    parser.add_argument("--unrolled", action='store_true', default=False)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument("--arch_learning_rate", type=float, default=0.01)
    parser.add_argument("--min_arch_learning_rate", type=float, default=0.0005)
    parser.add_argument("--arch_weight_decay", type=float, default=1e-3)
    # spos config
    parser.add_argument("--arch_sample_num", type=int, default=1000)
    parser.add_argument("--stand_alone_path", type=str, default='')
    # fine-tune config
    parser.add_argument("--tune_sample_num", type=int, default=20) 
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--negative_sampling_num", type=int, default=500)
    parser.add_argument("--isolated_change", type=bool, default=False)
    parser.add_argument("--positive_fact_num", type=int, default=3000)
    parser.add_argument("--dataset_dir", type=str, default="datasets/")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--tensorboard_dir", type=str, default="tensorboard/")
    parser.add_argument("--saved_model_dir", type=str, default="saved_models/")
    parser.add_argument("--weight_path", type=str, default='')
    parser.add_argument("--fixed_ops", type=str, default='')
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--search_res_dir", type=str, default="searched_res/")
    parser.add_argument("--tune_res_dir", type=str, default="tune_res/")
    parser.add_argument("--search_res_file", type=str, default="")
    parser.add_argument("--arch", type=str, default="")
    args = parser.parse_args()
    initialize_seed(args.random_seed)
    data = load_data(args.dataset)
    num_ent, train_data, valid_data, test_data, num_rels = data.num_nodes, data.train, data.valid, data.test, data.num_rels
    triplets, s2o = process({'train': train_data, 'valid': valid_data, 'test': test_data}, num_rels)  # s2o获得每个节点的邻居

    device = torch.device('cuda:0')
    torch.autograd.set_detect_anomaly(True)
    # train_loader = DataLoader(TrainDataset(triplets['train'], num_rels, args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # evaluate_loader = DataLoader(TestDataset(triplets['valid'], num_rels, args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # test_loader = DataLoader(TestDataset(triplets['test'], num_rels, args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    train_loader = DataLoader(TrainDataset(triplets['train'], num_rels, args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=TrainDataset.collate_fn)
    evaluate_loader = DataLoader(TestDataset(triplets['valid'], num_rels, args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=TestDataset.collate_fn)
    test_loader = DataLoader(TestDataset(triplets['test'], num_rels, args), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=TestDataset.collate_fn)
    
    # 加载节点位置特征(度中心性 特征向量中心性 pgrank 介数 katz)
    position_feature = torch.load("./ptFile/" + args.dataset.split('-')[0] + "/" + args.dataset.split('-')[0] + "-" +"position_feature.pt").to(device)
    
    trainer = Trainer(args, data, train_loader, evaluate_loader, test_loader, device, s2o, position_feature)
    if args.train_mode == "train":
        trainer.train()
    elif args.train_mode == "tune":
        arch_set = set()
        with open(args.search_res_file, 'r') as f:
            search_res_list = load(f)
            for search_res in search_res_list:
                Trainer.cnt_tune = 0
                args.dataset = search_res["dataset"]
                args.search_mode = search_res["search_mode"]
                trainer.fine_tuning(search_res["genotype"])
    elif args.train_mode == "search":
        start_running_time = strftime("%Y%m%d_%H%M%S")
        args.time_log_dir = f'{start_running_time}'
        search_res = []
        for idx in range(args.search_run_num):
            if args.search_mode == "random":
                genotype = trainer.random_bayesian_search()
            elif args.search_mode == "spos":
                genotype = trainer.spos_train_supernet()
            elif args.search_mode == "spos_search":
                genotype = trainer.spos_arch_search()
            elif args.search_mode == "darts_valid_loss":
                genotype, search_time = trainer.darts_train_search("darts_valid_loss")
            elif args.search_mode == "darts_train_loss":
                genotype, search_time = trainer.darts_train_search("darts_train_loss")
            else:
                genotype = None
            if genotype:
                res_dict = OrderedDict()
                res_dict["seed"] = args.random_seed
                res_dict["dataset"] = args.dataset
                res_dict["search_mode"] = args.search_mode
                res_dict["genotype"] = genotype
                search_res.append(res_dict)
        res_dir = args.search_res_dir + args.dataset + '/' + args.search_mode
        if not exists(res_dir):
            mkdir(res_dir)
        with open(res_dir + f'/{start_running_time}.json', 'w') as f:
            dump(search_res, f)
    elif args.train_mode == "debug":
        trainer.cnt_tune = 0
        start_running_time = strftime("%Y%m%d_%H%M%S")
        args.time_log_dir = f'{start_running_time}'
        # spa_icews14
        # trainer.debug("rgcn||sa||lc_concat||rgat_vanilla||identity||lc_concat||compgcn_rotate||identity||lf_mean")
        # spa_icews05_15
        # trainer.debug("rgcn||sa||lc_concat||rgcn||identity||lc_concat||compgcn_rotate||gru||lf_mean")
        # # spa_gdelt
        # trainer.debug("compgcn_rotate||gru||lc_concat||rgcn||gru||lc_skip||compgcn_rotate||gru||lf_mean")

if __name__ == '__main__':
    main()
