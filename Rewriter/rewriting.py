import numpy as np
import torch
from tensorboardX import SummaryWriter
import os
from .data_obj import DataProcessor, Community
from .rewriter_core import RewritingAgent
from .symbol import EXPAND, EXCLUDE, VIRTUAL_EXCLUDE_NODE
from utils import generate_outer_boundary, eval_scores


class CommRewriting:
    def __init__(self, args, graph, feat_mat, train_comms, valid_comms, pred_comms, cost_choice):
        self.args = args
        self.data_processor = DataProcessor(args, args.dataset, feat_mat, graph, train_comms, valid_comms)
        self.graph = graph
        self.cost_choice = cost_choice
        self.feat_mat = feat_mat
        self.agent = RewritingAgent(args)
        self.valid_comms, self.pred_comms = valid_comms, pred_comms
        self.writer = SummaryWriter(args.writer_dir)
        self.max_step = args.max_step
        self.best_epoch = None

    def load_net(self, filename=None):
        file = self.args.writer_dir + "/commr.pt" if not filename else filename
        print(f"Load net from {file} at Epoch{self.best_epoch}")
        data = torch.load(file)
        self.agent.expand_net.load_state_dict(data[EXPAND])
        self.agent.exclude_net.load_state_dict(data[EXCLUDE])

    def save_net(self, file_name=None):
        data = {
            EXPAND: self.agent.expand_net.state_dict(),
            EXCLUDE: self.agent.exclude_net.state_dict()
        }
        f_name = self.args.writer_dir + "/commr.pt" if not file_name else file_name
        torch.save(data, f_name)

    def train(self):
        self.agent.exclude_net.train()
        self.agent.expand_net.train()

        gamma = self.args.gamma

        n_episode = self.args.n_episode
        n_epoch = self.args.n_epoch
        n_sample = 0

        prg_bar = range(n_epoch)
        # Validation set for getting best model
        val_data = self.data_processor.generate_data(batch_size=n_episode, valid=True)

        steps, expand_steps, exclude_steps = [], [], []
        total_exclude_rewards, total_expand_rewards = [], []

        best_f, best_j, best_nmi = 0, 0, 0

        for epoch in prg_bar:
            n_sample += 1
            exclude_log_probs, exclude_rewards = [], []
            expand_log_probs, expand_rewards = [], []

            batch_data = self.data_processor.generate_data(batch_size=n_episode)

            for i in range(len(batch_data)):
                obj = batch_data[i]
                episode_exclude_rewards, episode_expand_rewards = [], []
                total_exclude_reward, total_expand_reward = 0, 0
                step, expand_step, exclude_step = 0, 0, 0

                expand, exclude = True, True

                while True:
                    if exclude:
                        exclude_action = self.agent.choose_action(obj, EXCLUDE)
                        if exclude_action is not None:
                            exclude_log_probs.append(exclude_action["log_prob"])
                            # Apply EXCLUDE
                            exclude_reward = obj.apply_exclude(exclude_action["node"], self.cost_choice)
                            total_exclude_reward += exclude_reward
                            episode_exclude_rewards.append(exclude_reward)
                            exclude_step += 1
                        else:
                            exclude = False

                    if expand:
                        expand_action = self.agent.choose_action(obj, EXPAND)
                        if expand_action is not None:
                            expand_log_probs.append(expand_action["log_prob"])
                            # Apply EXPAND
                            expand_reward = obj.apply_expand(expand_action["node"], self.cost_choice)
                            total_expand_reward += expand_reward
                            episode_expand_rewards.append(expand_reward)
                            expand_step += 1
                        else:
                            expand = False
                    next_state = obj.step(self.agent.gcn)

                    if (not exclude and not expand) or step >= self.max_step:
                        if len(episode_exclude_rewards) > 0:
                            r = [np.sum(episode_exclude_rewards[i] * (gamma**np.array(
                                range(0, len(episode_exclude_rewards)-i)))) for i in range(len(episode_exclude_rewards))]
                            exclude_rewards.append(np.array(r))
                            total_exclude_rewards.append(total_exclude_reward)
                        if len(episode_expand_rewards) > 0:
                            r = [np.sum(episode_expand_rewards[i] * (gamma**np.array(
                                range(0, len(episode_expand_rewards)-i)))) for i in range(len(episode_expand_rewards))]
                            expand_rewards.append(np.array(r))
                            total_expand_rewards.append(total_expand_reward)
                        steps.append(step)
                        exclude_steps.append(exclude_step)
                        expand_steps.append(expand_step)
                        break

                    obj.feat_mat = next_state
                    step += 1

            if len(total_exclude_rewards) > 0:
                avg_total_exclude_reward = sum(total_exclude_rewards) / len(total_exclude_rewards)
                self.writer.add_scalar(f"{self.cost_choice}-Reward/AvgExcludeReward", avg_total_exclude_reward, n_sample)
            if len(total_expand_rewards) > 0:
                avg_total_expand_reward = sum(total_expand_rewards) / len(total_expand_rewards)
                self.writer.add_scalar(f"{self.cost_choice}-Reward/AvgExpandReward", avg_total_expand_reward, n_sample)

            self.writer.add_scalar(f"{self.cost_choice}-Steps/AvgSteps", sum(steps)/len(steps), epoch)
            self.writer.add_scalar(f"{self.cost_choice}-Steps/AvgExcludeSteps", sum(exclude_steps)/len(exclude_steps), epoch)
            self.writer.add_scalar(f"{self.cost_choice}-Steps/AvgExpandSteps", sum(expand_steps)/len(expand_steps), epoch)

            if len(exclude_rewards):
                exclude_rewards = np.concatenate(exclude_rewards, axis=0)
                exclude_rewards = (exclude_rewards - np.mean(exclude_rewards)) / (np.std(exclude_rewards) + 1e-9)
                self.agent.learn(torch.stack(exclude_log_probs), torch.from_numpy(exclude_rewards), cal_type=EXCLUDE)
            if len(expand_rewards):
                expand_rewards = np.concatenate(expand_rewards, axis=0)
                expand_rewards = (expand_rewards - np.mean(expand_rewards)) / (np.std(expand_rewards)+1e-9)
                self.agent.learn(torch.stack(expand_log_probs), torch.from_numpy(expand_rewards), cal_type=EXPAND)

            # TODO: Validation on val-set and save the best model
            if (epoch + 1) % 20 == 0:
                f, j, nmi = eval_scores(val_data, self.valid_comms)
                rewrite_val = self.rewrite_community(valid=True, val_pred=val_data)
                new_f, new_j, new_nmi = eval_scores(rewrite_val, self.valid_comms, tmp_print=False)
                if new_f - f > 0 or new_j - j > 0 or new_nmi - nmi > 0:
                    print(f"[Eval-Epoch{epoch+1}] Improve f1 {new_f - f :.04f}, "
                          f"improve jaccard {new_j -j:.04f}, improve new_nmi {new_nmi-nmi:.04f}")
                    # self.save_net(self.args.writer_dir + f"/commr/epoch{epoch+1}.pt")
                    if new_f - f >= best_f and epoch >= 400:
                        best_f = new_f - f
                        self.best_epoch = epoch
                        self.save_net(self.args.writer_dir + f"/commr_eval_best.pt")
        # TODO: Save model
        self.save_net()

    def get_rewrite(self, filename=None):
        if filename:
            self.load_net(filename)
        elif os.path.exists(self.args.writer_dir + f"/commr_eval_best.pt"):
            self.load_net(self.args.writer_dir + "/commr_eval_best.pt")
        else:
            self.load_net()
        rewrite_comms = self.rewrite_community(valid=False, val_pred=False)
        lengths = np.array([len(pred_com) for pred_com in rewrite_comms])
        print(f"[Rewrite] Pred size {len(rewrite_comms)}, Avg Length {np.mean(lengths):.04f}")
        return rewrite_comms

    def rewrite_community(self, valid=False, val_pred=None):
        new_preds = []
        pred_comms = self.pred_comms if not valid else val_pred

        for i in range(len(pred_comms)):
            pred = pred_comms[i]
            pred = sorted(pred)

            outer_bound = generate_outer_boundary(self.graph, pred, max_size=20)
            nodes = sorted(pred + outer_bound)

            expand, exclude = True, True
            mapping = {idx: node for idx, node in enumerate(nodes)}

            com_obj = Community(self.feat_mat[nodes, :], pred, None, nodes, self.graph.subgraph(nodes), mapping,
                                expand=expand)

            step = 0
            while True:
                step += 1

                if exclude:
                    exclude_action = self.agent.choose_action(com_obj, "exclude")
                    if exclude_action is not None:
                        node = exclude_action["node"]
                        if node in com_obj.pred_com:
                            com_obj.pred_com.remove(node)
                    else:
                        exclude = False
                if expand:
                    expand_action = self.agent.choose_action(com_obj, "expand")
                    if expand_action is not None:
                        node = expand_action["node"]
                        if node not in com_obj.pred_com:
                            com_obj.pred_com.append(node)
                    else:
                        expand = False
                if (not exclude and not expand) or step >= self.args.max_step:
                    break
                next_state = com_obj.step(self.agent.gcn)
                com_obj.feat_mat = next_state

            if VIRTUAL_EXCLUDE_NODE in com_obj.pred_com:
                com_obj.pred_com.remove(VIRTUAL_EXCLUDE_NODE)
            if len(com_obj.pred_com) > 0:
                new_preds.append(com_obj.pred_com)

        return new_preds
