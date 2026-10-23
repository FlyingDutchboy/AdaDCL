import numpy as np
import torch
import torch.nn as nn
import time
import os
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from Args import args
from diff import *
import Log
from utility import *
from collections import defaultdict
import matplotlib.pyplot as plt


# class AdaptiveKLScheduler:
#     def __init__(self, initial_beta=0.001, beta_min=1e-6, beta_max=0.01, target_kl=2.5, kp=0.001):
#         """
#         Args:
#             initial_beta: 初始 beta 值
#             beta_min: beta 的下限 (防止变为0导致后验坍塌)
#             beta_max: beta 的上限
#             target_kl: 期望达到的 KL 散度值 (Target)
#             kp: 比例增益 (Proportional Gain)，即 beta 调整的速率/步长
#         """
#         self.current_beta = initial_beta
#         self.beta_min = beta_min
#         self.beta_max = beta_max
#         self.target_kl = target_kl
#         self.kp = kp
#
#     def update(self, current_kl_value):
#         """
#         Logic: error = current - target
#                beta_new = beta_old + kp * error
#         """
#         error = current_kl_value - self.target_kl
#         self.current_beta += self.kp * error
#
#         self.current_beta = max(self.beta_min, min(self.current_beta, self.beta_max))
#         return self.current_beta
#
#     def get_beta(self):
#         return self.current_beta


class Coach:
    def __init__(self, handler):
        self.handler = handler
        self.n_user = handler.n_user
        self.n_item = handler.n_item
        self.dim = args.hidden_factor

        # 尝试从 args 获取参数，如果没有则使用默认推荐值
        # target_kl = getattr(args, 'target_kl', 0.5)  # 建议设为 0.1 ~ 1.0 之间
        # beta_lr = getattr(args, 'beta_lr', 0.0001)  # beta 的调整速率

        # self.kl_beta_scheduler = AdaptiveKLScheduler(
        #     initial_beta=0.0001,  # 从较小值开始
        #     beta_min=1e-6,
        #     beta_max=args.beta,  # 最高不超过 args.beta
        #     target_kl=target_kl,
        #     kp=beta_lr
        # )

        log_print("USER", self.n_user, "ITEM", self.n_item)
        log_print("NUM OF INTERACTIONS", self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ["Loss", "preLoss", "prediction", "Recall", "NDCG"]
        for met in mets:
            self.metrics["Train" + met] = list()
            self.metrics["Test" + met] = list()
        setup_seed(args.random_seed)
        self.history = {
            "BPR_Loss": [],
            'kl_loss': [],
            'reg_loss': [],
            'total_loss': [],
            'NDCG': [],
            'Recall': [],
            'lr': [],
            # 'beta': []
        }

    def makePrint(self, name, ep, reses, save):
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += "%s = %.4f, " % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + "  "
        return ret

    def run(self):
        self.prepareModel()
        log_print("Model Prepared")
        log_print("All Parameters:")
        for arg, value in vars(args).items():
            log_print(f"{arg}: {value}")

        best_epoch = 0
        best_recall_20 = 0
        best_results = ()
        trnLoader = self.handler.trnLoader
        log_print("Model Initialized")
        et = 0

        try:
            for ep in range(0, args.epoch):
                if et > 1000:
                    print(f"Early stopping at epoch {ep}")
                    break
                start_time = time.time()
                tstFlag = ep % args.tstEpoch == 0

                # 训练一个 Epoch
                loss_dict = self.trainEpoch(trnLoader=trnLoader)

                # 记录数据
                self.history['BPR_Loss'].append(loss_dict['BPR_loss'])
                self.history['kl_loss'].append(loss_dict['kl_loss'])
                self.history['reg_loss'].append(loss_dict['reg_loss'])
                self.history['total_loss'].append(loss_dict['total_loss'])
                self.history['lr'].append(loss_dict['lr'])
                # self.history['beta'].append(loss_dict['beta'])  # 记录动态 beta

                if args.report_epoch:
                    if ep % 1 == 0:
                        log_str = "Epoch {:03d}; ".format(ep)
                        for k, v in loss_dict.items():
                            if k == 'beta':  # 单独打印 beta，方便观察
                                log_str += "{}: {:.6f}; ".format(k, v)
                            else:
                                log_str += "{}: {:.4f}; ".format(k, v)
                        log_str += "Time cost: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                        log_print(log_str)

                if tstFlag:
                    eval_start = time.time()
                    reses = self.testEpoch()
                    self.history['Recall'].append(reses["Recall"][1])
                    self.history['NDCG'].append(reses["NDCG"][1])

                    if reses["Recall"][1] > best_recall_20:
                        best_epoch = ep
                        best_recall_20 = reses["Recall"][1]
                        best_results = reses
                        if not os.path.exists("./result"):
                            os.makedirs("./result")
                        state = {
                            'model_1': self.model_1.state_dict(),
                            'model_2': self.model_2.state_dict(),
                            'uEmb': self.uEmb.state_dict(),
                            'iEmb': self.iEmb.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                        }
                        torch.save(state, os.path.join("./result", f"{args.data}_best_model.pt"))
                        log_print("Saved best model")
                    else:
                        et = et + 1
                    log_print("Evalution cost: " + time.strftime("%H: %M: %S", time.gmtime(time.time() - eval_start)))
                    print_results(None, test_result=reses)
                    log_print("----------------------------------------------------------------")

        except KeyboardInterrupt:
            log_print("\nTraining interrupted manually (KeyboardInterrupt). Preparing to plot curves...")

        finally:
            if len(self.history['BPR_Loss']) > 0:
                log_print("Plotting curves...")
                self.plot_curves()

            if best_results:
                log_print(f"Best Results achieved at epoch {best_epoch}:")
                print_results(test_result=best_results)
            else:
                log_print("No evaluation results available.")

    def prepareModel(self):
        self.topN = [10, 20, 50, 100]
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.model_1 = Tenc(
            args.hidden_factor, args.text_dim, self.n_item, args.statesize,
            args.dropout_rate, args.diffuser_type, self.device,
        )
        self.model_2 = Tenc(
            args.hidden_factor, args.image_dim, self.n_item, args.statesize,
            args.dropout_rate, args.diffuser_type, self.device,
        )
        self.diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
        self.iEmb = nn.Embedding(self.n_item, args.hidden_factor).to(self.device)
        self.uEmb = nn.Embedding(self.n_user, args.hidden_factor).to(self.device)
        self.interactions = self.load_interactions()
        self.iEmb_pt = torch.from_numpy(np.load(f"{args.data_dir}/{args.data}/iEmbeds_pt.npy")).float().to(self.device)
        self.uEmb_pt = torch.from_numpy(np.load(f"{args.data_dir}/{args.data}/uEmbeds_pt.npy")).float().to(self.device)
        self.load_add_iEmb()
        nn.init.normal_(self.iEmb.weight, std=0.1)
        nn.init.normal_(self.uEmb.weight, std=0.1)
        params_to_optimize = (
                list(self.model_1.parameters()) + list(self.model_2.parameters()) +
                list(self.iEmb.parameters()) + list(self.uEmb.parameters())
        )

        if args.lr > 0.01:
            print(f"WARNING: LR ({args.lr}) is dangerously high! Resetting to 0.001")
            args.lr = 0.001

        if args.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
        elif args.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
        elif args.optimizer.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
        elif args.optimizer.lower() == "rmsprop":
            self.optimizer = torch.optim.RMSprop(params_to_optimize, lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

        warmup_epochs = args.warmup_epochs
        cosine_epochs = max(1, args.epoch - warmup_epochs)

        scheduler_warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        scheduler_cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs, eta_min=1e-6)

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs])

        self.model_1.to(self.device)
        self.model_2.to(self.device)

        checkpoint_path = os.path.join('./result', f'{args.data}_best_model.pt')
        if getattr(args, 'load_model', False) and os.path.exists(checkpoint_path):
            log_print(f"Loading best model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model_1.load_state_dict(checkpoint['model_1'])
            self.model_2.load_state_dict(checkpoint['model_2'])
            self.uEmb.load_state_dict(checkpoint['uEmb'])
            self.iEmb.load_state_dict(checkpoint['iEmb'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            log_print('Model loaded successfully')
        else:
            log_print("No best model found")

        self.graph = self.handler.graph.to(self.device)

    def trainEpoch(self, trnLoader=None):
        epoch_loss = 0.0
        epoch_bpr = 0.0
        epoch_kl = 0.0
        epoch_reg = 0.0
        epoch_ssl = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']

        num_batches = len(trnLoader)
        for j, batch in enumerate(trnLoader):
            user, item = batch
            user = user.long().to(self.device)
            pos = item.long().to(self.device)
            negs = trnLoader.dataset.negSampling(pos, user)
            neg = torch.tensor(negs).long().to(self.device)
            self.optimizer.zero_grad()
            x_start = self.iEmb(pos)
            n = torch.randint(0, args.timesteps, (x_start.shape[0],), device=self.device).long()

            add_info_1 = self.iKge[pos]
            reconloss_1, predicted_x_1, mu_1, log_var_1 = self.diff.p_losses(
                self.model_1, x_start, add_info_1, n, loss_type="l2", flag=args.flag
            )
            add_info_2 = self.iSoc[pos]
            reconloss_2, predicted_x_2, mu_2, log_var_2 = self.diff.p_losses(
                self.model_2, x_start, add_info_2, n, loss_type="l2", flag=args.flag
            )

            kl_loss = 0
            if mu_1 is not None and log_var_1 is not None:
                kl_1 = -0.5 * torch.mean(1 + log_var_1 - mu_1.pow(2) - log_var_1.exp())
                kl_loss += kl_1
            if mu_2 is not None and log_var_2 is not None:
                kl_2 = -0.5 * torch.mean(1 + log_var_2 - mu_2.pow(2) - log_var_2.exp())
                kl_loss += kl_2

            user_emb0 = self.uEmb(user)
            pos_emb0 = self.iEmb(pos)
            neg_emb0 = self.iEmb(neg)

            pos_scores = torch.sum(torch.mul(user_emb0, pos_emb0), dim=1)
            neg_scores = torch.sum(torch.mul(user_emb0, neg_emb0), dim=1)
            bprloss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

            sslloss_1 = calc_ssl_loss_1(pos, predicted_x_1, self.iEmb.weight, tau=0.5)
            sslloss_2 = calc_ssl_loss_1(pos, predicted_x_2, self.iEmb.weight, tau=0.5)

            regloss = (1 / 2) * (user_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2)) / len(user)
            predicted_x_loss = predicted_x_1.norm(2).pow(2) / len(item) + predicted_x_2.norm(2).pow(2) / len(item)

            # ============================================================
            # 4. 使用 cur_beta 计算总 Loss
            # ============================================================
            loss = (
                    reconloss_1 * args.recon_weight
                    + reconloss_2 * args.recon_weight
                    + bprloss * args.bpr_alpha
                    + regloss * args.reg_alpha
                    + predicted_x_loss * args.prex_weight
                    + (sslloss_1 + sslloss_2) * args.ssl_weight
                    + sslloss_1 * args.ssl_weight
                    + kl_loss * args.beta
            )

            loss.backward()

            # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(self.model_1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model_2.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.iEmb.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.uEmb.parameters(), max_norm=1.0)

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_bpr += bprloss.item()
            epoch_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            epoch_reg += regloss.item()
            epoch_ssl += sslloss_1.item()

        self.scheduler.step()

        # ============================================================
        # 5. Epoch 结束，根据本轮平均 KL 值更新 Beta
        # ============================================================
        # avg_kl_loss = epoch_kl / num_batches
        # self.kl_beta_scheduler.update(avg_kl_loss)

        return {
            "total_loss": epoch_loss / num_batches,
            "kl_loss": epoch_kl / num_batches,
            "BPR_loss": epoch_bpr / num_batches,
            "reg_loss": epoch_reg / num_batches,
            'ssl_loss': epoch_ssl / num_batches,
            'predicted_x_loss_1': reconloss_1,
            'predicted_x_loss_2': reconloss_2,
            'lr': current_lr,
            # 'beta': cur_beta  # 返回本轮使用的 beta
        }

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg, epPrecision, epMRR = None, None, None, None
        num = math.ceil(len(tstLoader.dataset) / args.tstBat)
        for usr, trnMask in tstLoader:
            usr = usr.long().to(self.device)
            trnMask_tensor = trnMask.clone().detach().to(self.device)
            tst_users_tensor = usr.clone().detach().to(self.device)
            prediction = compute_prediction(
                self.uEmb.weight,
                self.iEmb.weight,
                args.n_layers,
                self.graph,
                self.n_user,
                self.n_item,
                tst_users_tensor,
                self.model_1,
            )
            prediction = prediction * (1 - trnMask_tensor) - trnMask_tensor * 1e8
            _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
            topK = topK.cpu().detach().numpy().tolist()
            predict_items = []
            predict_items.extend(topK)
            tstLocs_np = np.array(
                self.handler.tstLoader.dataset.tstLocs, dtype=object
            )
            target_items = tstLocs_np[usr.cpu()]
            precision, recall, NDCG, MRR = computeTopNAccuracy(
                target_items, predict_items, self.topN
            )

            def accumulate(epMetric, metric):
                if epMetric is None:
                    return metric
                else:
                    return [epMetric[j] + metric[j] for j in range(len(metric))]

            epRecall = accumulate(epRecall, recall)
            epNdcg = accumulate(epNdcg, NDCG)
            epPrecision = accumulate(epPrecision, precision)
            epMRR = accumulate(epMRR, MRR)
        ret = dict()
        ret["Recall"] = [x / num for x in epRecall]
        ret["NDCG"] = [x / num for x in epNdcg]
        ret["Precision"] = [x / num for x in epPrecision]
        ret["MRR"] = [x / num for x in epMRR]
        return ret

    def gcn(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb])
        embs = [all_emb]
        for layer in range(args.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users_final, items_final = torch.split(light_out, [self.n_user, self.n_item])
        return users_final, items_final

    def load_interactions(self):
        npy_path = os.path.join(args.data_dir, args.data, 'train_list.npy')
        txt_path = os.path.join(args.data_dir, args.data, 'train.txt')

        if os.path.exists(npy_path):
            try:
                data = np.load(npy_path, allow_pickle=True)
                return data
            except Exception as e:
                print(f"Failed to load npy file: {e}")

        if os.path.exists(txt_path):
            data = []
            with open(txt_path, 'r') as f:
                for line in f:
                    try:
                        user, item = map(int, line.strip().split())
                        data.append((user, item))
                    except ValueError as ve:
                        print(f"Skipping invalid line: {line.strip()} — {ve}")
            return data

        print("No valid data file found.")
        return []

    def get_social_guided_item_embeddings(self):
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        for u, i in self.interactions:
            self.user_items[u].add(i)
            self.item_users[i].add(u)

        self.user_sim = torch.zeros((self.n_user, self.n_user), device=self.device)
        for u1 in tqdm(range(self.n_user), desc="compute user similarity"):
            for u2 in range(u1 + 1, self.n_user):
                inter1 = self.user_items[u1]
                inter2 = self.user_items[u2]
                if not inter1 or not inter2:
                    continue
                sim = len(inter1 & inter2) / (len(inter1 | inter2) + 1e-8)
                if sim > 0:
                    self.user_sim[u1, u2] = self.user_sim[u2, u1] = sim

        u_social_emb = self.uEmb_pt.clone()
        for u in tqdm(range(self.n_user), desc="generate user social embeddings"):
            neighbors = self.user_sim[u]
            sim_sum = neighbors.sum()
            if sim_sum > 0:
                weighted_sum = torch.matmul(neighbors, self.uEmb_pt)
                u_social_emb[u] = (self.uEmb_pt[u] + weighted_sum / sim_sum) / 2

        item_social_emb = torch.zeros((self.n_item, self.dim), device=self.device)
        for i in tqdm(range(self.n_item)):
            users = list(self.item_users[i])
            if users:
                item_social_emb[i] = u_social_emb[users].mean(dim=0)
            else:
                item_social_emb[i] = self.iEmb_pt[i]

        self.iSoc = item_social_emb
        np.save(args.data_dir + f"/{args.data}/item_social_emb.npy", item_social_emb.cpu().numpy())
        print("✅ item_social_emb.npy saved!")

    def get_item_knowledge_graph_embeddings(self, topk=20):
        if not hasattr(self, "item_users"):
            self.item_users = defaultdict(set)
            for u, i in self.interactions:
                self.item_users[i].add(u)
        self.item_sim = torch.zeros((self.n_item, self.n_item), device=self.device)
        for i in tqdm(range(self.n_item), desc="compute item-item Jaccard similarity"):
            for j in range(i + 1, self.n_item):
                users_i = self.item_users[i]
                users_j = self.item_users[j]
                if not users_i or not users_j:
                    continue
                sim = len(users_i & users_j) / (len(users_i | users_j) + 1e-8)
                if sim > 0:
                    self.item_sim[i, j] = self.item_sim[j, i] = sim

        item_kg_emb = torch.zeros((self.n_item, self.dim), device=self.device)
        for i in tqdm(range(self.n_item), desc="aggregate item neighbor embeddings"):
            sim_row = self.item_sim[i]
            topk_indices = torch.topk(sim_row, k=topk).indices
            valid = sim_row[topk_indices] > 0
            neighbors = topk_indices[valid]
            if len(neighbors) > 0:
                item_kg_emb[i] = self.iEmb_pt[neighbors].mean(dim=0)
            else:
                item_kg_emb[i] = self.iEmb_pt[i]

        self.iKge = item_kg_emb
        np.save(args.data_dir + f"/{args.data}/item_knowledge_emb.npy", item_kg_emb.cpu().numpy())
        print("✅ item_knowledge_emb.npy saved!")

    def load_add_iEmb(self):
        social_emb_path = os.path.join(args.data_dir, args.data, "item_social_emb.npy")
        if os.path.exists(social_emb_path):
            self.iSoc = torch.from_numpy(np.load(social_emb_path)).float().to(self.device)
        else:
            self.get_social_guided_item_embeddings()

        knowledge_emb_path = os.path.join(args.data_dir, args.data, "item_knowledge_emb.npy")
        if os.path.exists(knowledge_emb_path):
            self.iKge = torch.from_numpy(np.load(knowledge_emb_path)).float().to(self.device)
        else:
            self.get_item_knowledge_graph_embeddings()

    def plot_curves(self):
        plt.style.use("seaborn-v0_8")
        # 修改为 2x4 布局，新增 Beta 曲线图
        fig, axs = plt.subplots(2, 4, figsize=(24, 10), dpi=300)
        fig.suptitle(f'Training Metrics - {args.name}', fontsize=20)
        metrics_map = [
            ("BPR_Loss", 'BPR_Loss', axs[0, 0], 'tab:blue'),
            ('kl_loss', 'KL_Loss', axs[0, 1], 'tab:orange'),
            ('reg_loss', 'Reg_Loss', axs[0, 2], 'tab:green'),
            # ('beta', 'Beta (KL Weight)', axs[0, 3], 'tab:pink'),  # 新增
            ('total_loss', 'Total_Loss', axs[1, 0], 'tab:red'),
            ('NDCG', 'NDCG@20', axs[1, 1], 'tab:purple'),
            ('Recall', 'Recall@20', axs[1, 2], 'tab:brown'),
            ('lr', 'Learning Rate', axs[1, 3], 'gray')
        ]

        epochs = range(1, len(self.history['BPR_Loss']) + 1)
        eval_epochs = [i * args.tstEpoch for i in range(len(self.history['Recall']))]

        for key, title, ax, color in metrics_map:
            data = self.history.get(key, [])
            if len(data) > 0:
                x_axis = epochs if key not in ['Recall', 'NDCG'] else eval_epochs
                x_axis = x_axis[:len(data)]
                ax.plot(x_axis, data, label=key, color=color, linewidth=2)
                ax.set_title(title, fontsize=18)
                ax.set_ylabel('Value', fontsize=15)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(Log.log_save_path, "training_curves.png")
        plt.savefig(save_path)
        plt.close()
        log_print(f"Training curves saved to: {save_path}")
