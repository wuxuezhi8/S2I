from model_list import MODEL
from utils import get_metrics, get_logger, get_name, count_parameters_in_MB
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch
import time
from time import strftime
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, rand, space_eval
from os import mkdir, makedirs
from os.path import exists
from models.genotypes_new import NA_PRIMITIVES, LC_PRIMITIVES, LF_PRIMITIVES, NODE_INFO_PRIMITIVES, PAIR_INFO_PRIMITIVES, NF_PRIMITIVES, PF_PRIMITIVES
# DEBUG
from hyperopt.pyll.stochastic import sample
from pprint import pprint
from itertools import product
from sortedcontainers import SortedDict

EPOCH_TEST = {"WN18RR": 5,
              "FB15k-237": 10,
              "UMLS": 15,
              "Kinship": 20}


class Trainer(object):
    cnt_tune = 0

    def __init__(self, args, data, train_loader, evaluate_loader, test_loader, device, s2o, node_wise_feature):
        self.args = args
        self.device = device
        self.data = data      
        self.train_loader = train_loader
        self.evaluate_loader = evaluate_loader
        self.test_loader = test_loader
        self.optimizer = None
        # self.scheduler = None
        self.search_space = None
        self.logger = None
        self.s2o = s2o
        self.node_wise_feature = node_wise_feature

    def train(self):
        name = get_name(self.args)
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/'
        if not exists(log_dir):
            mkdir(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        writer = SummaryWriter(self.args.tensorboard_dir + self.args.dataset + name)
        model = MODEL[self.args.encoder](self.args, self.data, self.device, self.s2o, self.node_wise_feature, self.args.batch_size)
        model = model.cuda()
        self.logger.info("Parameter size = %fMB", count_parameters_in_MB(model))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=0.0001)

        best_val_mrr, best_test_mrr = 0.0, 0.0
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            training_loss = self.train_epoch(epoch, model, architect=None, lr=None, mode="train")
            valid_mrr = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_val_mrr:
                early_stop_cnt = 0
                best_val_mrr = valid_mrr
                test_mrr = self.evaluate_epoch(epoch, model, split="test")
                if test_mrr > best_test_mrr:
                    best_test_mrr = test_mrr
                    self.logger.info("Success")
                    # torch.save(model.state_dict(), f'{args.saved_model_dir}{name}.pth')
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 10:
                self.logger.info("Early stop!")
                self.logger.info(best_test_mrr)
                break
            writer.add_scalar('Loss/train', training_loss, epoch)
            writer.add_scalar('MRR/test', best_test_mrr, epoch)

    def random_bayesian_search(self):
        genotype_space = []

        for i in range(self.args.gnn_layer_num):
            genotype_space.append(hp.choice("G" + str(i), NA_PRIMITIVES))
            if i != self.args.gnn_layer_num - 1:
                genotype_space.append(hp.choice("LC" + str(i), LC_PRIMITIVES))
            else:
                genotype_space.append(hp.choice("LA" + str(i), LF_PRIMITIVES))
        genotype_space.append(hp.choice("NI", NODE_INFO_PRIMITIVES))
        genotype_space.append(hp.choice("NF", NF_PRIMITIVES))
        genotype_space.append(hp.choice("PI", PAIR_INFO_PRIMITIVES))
        trials = Trials()
        search_time = 0.0
        t_start = time.time()
        if self.args.search_mode == "random":
            best = fmin(self.train_parameter, genotype_space, algo=rand.suggest,
                        max_evals=self.args.baseline_sample_num,
                        trials=trials)
        elif self.args.search_mode == "bayesian":
            best = fmin(self.train_parameter, genotype_space,
                        algo=partial(tpe.suggest, n_startup_jobs=int(self.args.baseline_sample_num) / 5),
                        max_evals=self.args.baseline_sample_num,
                        trials=trials)
        else:
            raise NotImplementedError
        best_genotype = space_eval(genotype_space, best)
        t_end = time.time()
        search_time += (t_end - t_start)
        return "||".join(best_genotype)

    def evaluate_epoch(self, current_epoch, model, split="valid", mode=None):
        rank_list = []
        loss_list = []
        model.eval()
        with torch.no_grad():
            if split == 'test':
                evaluate = self.test_loader
            else:
                evaluate = self.evaluate_loader
                
            for batch_idx, (triplets, labels) in enumerate(evaluate):
                if len(triplets) < self.args.batch_size:
                  remaining_samples = self.args.batch_size - len(triplets)
                  random_indices = torch.randint(0, len(triplets), (remaining_samples,))
                  additional_samples = triplets[random_indices]
                  additional_labels = labels[random_indices]
                  # 将additional_samples和additional_labels添加到batch_data和batch_labels中
                  triplets = torch.cat((triplets, additional_samples), dim=0)
                  labels = torch.cat((labels, additional_labels), dim=0)

                if mode == "spos_train":
                    model.ent_encoder.ops = model.ent_encoder.generate_single_path()
                rank, loss = model.evaluate(triplets, labels, split)
                rank_list.append(rank)
                if split == 'valid' or split == 'train':
                    loss_list.append(loss.item())
                else:
                    loss_list.append(loss.item())
            if split == "train":
                self.logger.info(
                    '[Epoch:{} | {}]: Loss:{:.4}'.format(
                        current_epoch, split.capitalize(), np.mean(loss_list)))
                return np.mean(loss_list)
            else:
                all_ranks = torch.cat(rank_list)
                mrr, hit_1, hit_3, hit_10 = get_metrics(all_ranks)
                metrics_dict = {'mrr': mrr, 'hit_10': hit_10, 'hit_3': hit_3, 'hit_1': hit_1}
                metrics_result = {k: v.item() for k, v in metrics_dict.items()}
                # self.logger.info(
                #     '[Epoch:{} | {}]: {} Loss:{:.4}'.format(current_epoch, split.capitalize(), split.capitalize(), np.mean(loss_list)))
                self.logger.info('[Epoch:{} | {}]: Loss:{:.4}, MRR:{:.3}, Hits@10:{:.3}, Hits@3:{:.3}, Hits@1:{:.3}'.format(
                    current_epoch, split.capitalize(), np.mean(loss_list),
                    metrics_result['mrr'], metrics_result['hit_10'],
                    metrics_result['hit_3'],
                    metrics_result['hit_1']))
                return metrics_result, metrics_result['mrr'], np.mean(loss_list)

    def train_epoch(self, current_epoch, model, lr=None, mode='NONE'):
        train_loss_list = []
        for batch_idx, (triplets, labels) in enumerate(self.train_loader):
            if len(triplets) < self.args.batch_size:
              remaining_samples = self.args.batch_size - len(triplets)
              random_indices = torch.randint(0, len(triplets), (remaining_samples,))
              additional_samples = triplets[random_indices]
              additional_labels = labels[random_indices]
              # 将additional_samples和additional_labels添加到batch_data和batch_labels中
              triplets = torch.cat((triplets, additional_samples), dim=0)
              labels = torch.cat((labels, additional_labels), dim=0)
            # print("batch,{}".format(batch_idx))
            labels = labels.to(self.device)
            if mode == "spos_search":
                score = model(triplets)
                train_loss = model.calc_loss(score, labels)
                train_loss_list.append(train_loss.item())
            else:
                model.train()
                if mode == "spos_train":
                    model.ent_encoder.ops = model.ent_encoder.generate_single_path()
                self.optimizer.zero_grad()
                score = model(triplets)
                train_loss = model.calc_loss(score, labels)
                # print(train_loss)
                train_loss_list.append(train_loss.item())
                train_loss.backward()
                self.optimizer.step()
                if mode == "darts_valid_loss":
                    # self.optimizer.zero_grad()
                    self.arch_optimizer.zero_grad()
                    _, valid_loss = model.evaluate(triplets, labels, split="valid")  # validation loss
                    valid_loss.backward()
                    self.arch_optimizer.step()
                    # architect.step(train_timestamps, valid_timestamps, lr, self.optimizer, self.args.unrolled)
                elif mode == "darts_train_loss":
                    self.arch_optimizer.step()
        self.logger.info('[Epoch:{} | {}]: Train Loss:{:.4}'.format(current_epoch, self.args.train_mode.capitalize(),
                                                                    np.mean(train_loss_list)))
        return np.mean(train_loss_list)

    def fine_tuning(self, genotype):
        hyper_space = {
            # 'weight_decay': hp.uniform("wr", -8, -6),
            # 'weight_decay': hp.choice("wr", [-5]),
            'learning_rate': hp.uniform("lr", 0.0001, 0.001),
            # 'learning_rate': hp.choice("lr", [0.001,0.0001,0.0005]),
            "gcn_drop":  hp.uniform("gcn_drop", 0.1, 0.4),
            # "gcn_drop":  hp.choice("gcn_drop",[0.3]),
            # 'optimizer': hp.choice('optimizer', ['adam', 'adagrad'])
            # 'batch_size': hp.choice('batch_size', [256, 512, 1024]),
        }

        
        self.args.genotype = genotype
        trials = Trials()
        best = fmin(self.train_parameter, hyper_space,
                    algo=partial(tpe.suggest, n_startup_jobs=int(self.args.tune_sample_num) / 5),
                    max_evals=self.args.tune_sample_num,
                    trials=trials)
        space = space_eval(hyper_space, best)
        for k, v in space.items():
            setattr(self.args, k, v)
        best_test_mrr = 0.0
        for d in trials.results:
            if d["test_mrr"] > best_test_mrr:
                # best_val_mrr = -d["loss"]
                best_test_mrr = d["test_mrr"]
                best_test_result = d["test_result"]
        tune_dir = self.args.tune_res_dir + self.args.dataset +'/' + self.args.genotype
        if not exists(tune_dir):
            makedirs(tune_dir)
        with open(tune_dir+ '/' + strftime("%Y%m%d_%H%M%S") + '.txt', "w") as f1:
            f1.write(str(vars(self.args)) + "\n")
            f1.write(str(best_test_result))

    def train_parameter(self, parameter):
        Trainer.cnt_tune += 1
        self.args.index = Trainer.cnt_tune
        if self.args.train_mode == "search":
            self.args.genotype = "||".join(parameter)
        else:
            # parameter['weight_decay'] = 10 ** parameter['weight_decay']
            for k, v in parameter.items():
                setattr(self.args, k, v)
        name = get_name(self.args)
        search_res_dir = self.args.search_res_file.split('/')[-1].split('.')[0]
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/' \
                  f'{search_res_dir}/'
        if not exists(log_dir):
            makedirs(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        model = MODEL[self.args.encoder](self.args, self.data, self.device, self.args.genotype, self.s2o, self.node_wise_feature, self.args.batch_size)
        model = model.cuda()
        self.logger.info("Parameter size = %fMB", count_parameters_in_MB(model))
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)
        best_valid_mrr, best_test_mrr = 0.0, 0.0
        best_valid_result, best_test_result = {}, {}
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            loss = self.train_epoch(epoch, model, mode="tune")
            valid_result, valid_mrr, _ = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_valid_mrr:
                early_stop_cnt = 0
                best_valid_mrr = valid_mrr
                best_valid_result = valid_result
                if self.args.train_mode == "tune" and epoch > EPOCH_TEST[self.args.dataset]:
                    test_result, test_mrr, _ = self.evaluate_epoch(epoch, model, split="test")
                    if test_mrr > best_test_mrr:
                        best_test_mrr = test_mrr
                        best_test_result = test_result
                        self.logger.info("Success")
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 45 or epoch == self.args.max_epoch:
                self.logger.info("Early stop!")
                self.logger.info(f'{best_valid_mrr} {self.args.genotype}')
                break
            # self.scheduler.step(best_valid_mrr)
        return {'loss': -best_valid_mrr, 'status': STATUS_OK} if self.args.train_mode == "search" else {'loss': -best_valid_mrr, 'test_result':best_test_result, 'test_mrr': best_test_mrr, 'status': STATUS_OK}

    def debug(self, genotype):
        Trainer.cnt_tune += 1
        self.args.genotype = genotype
        name = get_name(self.args)
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/{self.args.encoder}/' \
                  f'{self.args.time_log_dir}_{self.args.random_seed}/'
        if not exists(log_dir):
            makedirs(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        writer = SummaryWriter(
            f'{self.args.tensorboard_dir}{self.args.dataset}{self.args.train_mode}/{self.args.encoder}/{self.args.time_log_dir}_{self.args.random_seed}/')
        model = MODEL[self.args.encoder](self.args, self.data, self.device, self.args.genotype, self.s2o, self.node_wise_feature, self.args.batch_size)
        model = model.cuda()
        self.logger.info("Parameter size = %fMB", count_parameters_in_MB(model))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)

        best_valid_mrr, best_test_mrr = 0.0, 0.0
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            train_loss = self.train_epoch(epoch, model, mode="tune")
            valid_mrr, valid_loss = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_valid_mrr:
                early_stop_cnt = 0
                best_valid_mrr = valid_mrr
                test_mrr, _ = self.evaluate_epoch(epoch, model, split="test")
                if test_mrr > best_test_mrr:
                    best_test_mrr = test_mrr
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 25 or epoch == self.args.max_epoch:
                self.logger.info("Early stop!")
                self.logger.info(f'{best_valid_mrr} {self.args.genotype}')
                break
            # self.scheduler.step(best_valid_mrr)
            # writer.add_scalar('Loss/train', train_loss, epoch)
            # writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('MRR/test', best_test_mrr, epoch)

    def spos_train_supernet(self):
        name = get_name(self.args)
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/' \
                  f'{self.args.time_log_dir}_{self.args.random_seed}/'
        if not exists(log_dir):
            makedirs(log_dir)
        weights_dir = f'weights/{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/{self.args.time_log_dir}_{self.args.random_seed}/'
        if not exists(weights_dir):
            makedirs(weights_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        self.logger.info(f'Log file is saved in {log_dir}')
        self.logger.info(f'Weight file is saved in {weights_dir}')
        writer = SummaryWriter(
            f'{self.args.tensorboard_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/{self.args.time_log_dir}/{self.args.random_seed}')
        model = MODEL[self.args.encoder](self.args, self.data, self.device, self.s2o, self.node_wise_feature, self.args.batch_size)
        model = model.cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        search_time = 0.0
        best_val_mrr, best_test_mrr = 0.0, 0.0
        for epoch in range(1, self.args.search_max_epoch + 1):
            t_start = time.time()
            train_loss = self.train_epoch(epoch, model, mode="spos_train")
            valid_result, valid_mrr, valid_loss = self.evaluate_epoch(epoch, model, split="valid",
                                                        mode="spos_train")
            t_end = time.time()
            search_time += (t_end - t_start)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('MRR/valid', valid_mrr, epoch)
            if epoch % 100 == 0:
                torch.save(model.state_dict(), f'{weights_dir}epoch_{epoch}.pt')
        # torch.save(model.state_dict(), f'{weights_dir}epoch_{self.args.search_max_epoch}.pt')
        search_time = search_time / 3600
        self.logger.info(f'The search process costs {search_time:.2f}h.')

        return None

    def spos_arch_search(self):
        name = '_search_'+str(self.args.random_seed)
        log_dir = self.args.weight_path.replace('weights', 'logs', 1).split('.')[0]
        if not exists(log_dir):
            makedirs(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(f'Log file is saved in {log_dir}')
        model = MODEL[self.args.encoder](self.args, self.data, self.device, self.s2o, self.node_wise_feature, self.args.batch_size)
        model = model.cuda()
        model.load_state_dict(torch.load(self.args.weight_path))
        self.logger.info(f'Finish loading checkpoint from {self.args.weight_path}')
        search_time = 0.0
        valid_mrr_searched_arch_res = SortedDict()
        for epoch in range(1, self.args.arch_sample_num + 1):
            model.ent_encoder.ops = model.ent_encoder.generate_single_path()
            arch = "||".join(model.ent_encoder.ops)
            t_start = time.time()
            valid_result, valid_mrr, valid_loss = self.evaluate_epoch(epoch, model, split="valid")
            valid_mrr_searched_arch_res.setdefault(valid_mrr, arch)
            self.logger.info('[Epoch:{} | {}]: Path:{}'.format(epoch, self.args.arch_sample_num, arch))
            t_end = time.time()
            search_time += (t_end - t_start)
        search_time = search_time / 3600
        self.logger.info(f'The search process costs {search_time:.2f}h.')
        import csv

        with open(log_dir+'_search_valid_mrr_'+str(self.args.random_seed)+'_res.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['valid mrr', 'arch'])
            res = valid_mrr_searched_arch_res.items()
            for i in range(500):
                writer.writerow([res[-1-i][0], res[-1-i][1]])

        return res[-1][1]

    def darts_train_search(self, mode):
            name = get_name(self.args)
            log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/' \
                      f'{self.args.time_log_dir}_{self.args.random_seed}/'
            if not exists(log_dir):
                makedirs(log_dir)
            self.logger = get_logger(name, log_dir)
            self.logger.info(self.args)
            writer = SummaryWriter(
                f'{self.args.tensorboard_dir}{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/{self.args.time_log_dir}_{self.args.random_seed}')
            
            model = MODEL[self.args.encoder](self.args, self.data, self.device, self.s2o, self.node_wise_feature, self.args.batch_size)
            model = model.cuda()
            self.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=self.args.learning_rate,
                                              weight_decay=self.args.weight_decay)
            self.arch_optimizer = torch.optim.Adam(model.ent_encoder.arch_parameters(),
                                              lr=self.args.arch_learning_rate,
                                              weight_decay=self.args.arch_weight_decay)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.search_max_epoch,
            #                                                             eta_min=self.args.min_learning_rate)
            # architect = Architect(model, self.args)
            search_time = 0.0
            best_val_mrr, best_test_mrr = 0.0, 0.0
            test_ws_mrr = 0.0
            best_genotype = ''
            for epoch in range(1, self.args.search_max_epoch + 1):
                t_start = time.time()
                # lr = self.scheduler.get_lr()[0]
                lr = self.args.learning_rate
                # genotype, softmax_dict = model.ent_encoder.genotype(show_softmax=True)
                # self.logger.info(f'[Epoch:{epoch}]: Learning Rate: {lr:.5f}, Genotype: {genotype}')
                # if best_genotype != genotype:
                #     best_genotype = genotype
                train_loss = self.train_epoch(epoch, model, lr, mode=mode)
                valid_result, valid_mrr, valid_loss = self.evaluate_epoch(epoch, model, split="valid")
                # if epoch % 10 == 0:
                test_result, test_mrr, test_loss = self.evaluate_epoch(epoch, model, split="test")
                # if valid_mrr > best_val_mrr:
                #     best_val_mrr = valid_mrr
                # if epoch > EPOCH_TEST[self.args.dataset]:
                #     test_mrr, _ = self.evaluate_epoch(epoch, model, split="test")
                #     if test_mrr > best_test_mrr:
                #         best_test_mrr = test_mrr
                #         self.logger.info("Success")

                # self.scheduler.step()
                t_end = time.time()
                search_time += (t_end - t_start)
                weights_dir = f'weights/{self.args.dataset}{self.args.train_mode}/{self.args.search_mode}/{self.args.encoder}/{self.args.time_log_dir}_{self.args.random_seed}'
                if not exists(weights_dir):
                    makedirs(weights_dir)
                torch.save(model.state_dict(), f'{weights_dir}/weights_{self.args.random_seed}.pt')
                genotype = model.ent_encoder.genotype()
                if best_genotype != genotype:
                    best_genotype = genotype
                self.logger.info(f'[Epoch:{epoch}]: Genotype: {genotype}')
                # model = MODEL['DynamicGRUSDT'](self.args, self.dataset_info_dict, self.device, genotype)
                # writer.add_scalar('Loss/train', train_loss, epoch)
                # writer.add_scalar('Loss/valid', valid_loss, epoch)
                # writer.add_scalar('Loss/valid_ws', valid_ws_loss, epoch)
                # writer.add_scalar('MRR/valid', valid_mrr, epoch)
                # writer.add_scalar('MRR/valid_ws', valid_ws_mrr, epoch)
                # writer.add_scalar('MRR/test_ws', test_ws_mrr, epoch)
                # writer.add_scalars('SOFTMAX/G', softmax_dict['G'], epoch)
                # writer.add_scalars('SOFTMAX/T', softmax_dict['T'], epoch)
                # writer.add_scalars('SOFTMAX/LC', softmax_dict['LC'], epoch)
                # writer.add_scalars('SOFTMAX/LA', softmax_dict['LA'], epoch)
                # writer.add_scalars('LC_SOFTMAX', softmax_dict, epoch)
                # writer.add_scalars('LA_SOFTMAX', softmax_dict, epoch)
                # writer.add_scalars('T_SOFTMAX', softmax_dict, epoch)
            search_time = search_time / 3600
            self.logger.info(f'The search process costs {search_time:.2f}h, arch is {best_genotype}')
            return best_genotype, search_time
   