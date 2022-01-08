from math import log
from typing import List, Union, Set
import numpy as np


def compare_comm(pred_comm: Union[List, Set],
                 true_comm: Union[List, Set]) -> (float, float, float, float):
    """
    Compute the Precision, Recall, F1 and Jaccard similarity
    as the second argument is the ground truth community.
    """
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return p, r, f, j


def eval_scores(pred_comms, true_comms, tmp_print=False):
    # 4 columns for precision, recall, f1, jaccard
    pred_scores = np.zeros((len(pred_comms), 4))
    truth_scores = np.zeros((len(true_comms), 4))

    for i, pred_comm in enumerate(pred_comms):
        np.max([compare_comm(pred_comm, true_comms[j])
                for j in range(len(true_comms))], 0, out=pred_scores[i])

    for j, true_comm in enumerate(true_comms):
        np.max([compare_comm(pred_comms[i], true_comm)
                for i in range(len(pred_comms))], 0, out=truth_scores[j])
    truth_scores[:, :2] = truth_scores[:, [1, 0]]

    if tmp_print:
        print("P, R, F, J AvgAxis0: ", pred_scores.mean(0))
        print("P, R, F, J AvgAxis1: ", truth_scores.mean(0))

    # Avg F1 / Jaccard
    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.

    # detect percent
    comm_nodes = {node for com in true_comms for node in com}
    pred_nodes = {node for com in pred_comms for node in com}
    percent = len(list(comm_nodes & pred_nodes)) / len(comm_nodes)

    # NMI
    nmi_score = get_nmi_score(pred_comms, true_comms)

    if tmp_print:
        print(f"AvgF1: {mean_score_all[2]:.4f} AvgJaccard: {mean_score_all[3]:.4f} NMI: {nmi_score:.4f} "
              f"Detect percent: {percent:.4f}")
    return round(mean_score_all[2], 4), round(mean_score_all[3], 4), round(nmi_score, 4)


def get_intersection(a, b, choice=None):
    return len(list(set(a) & set(b))) if not choice else list(set(a) & set(b))


def get_difference(a, b):
    intersection = get_intersection(a, b, choice="List")
    nodes = {x for x in a if x not in intersection}
    return len(list(nodes))


def get_nmi_score(pred, gt):
    def get_overlapping(pred_comms, ground_truth):
        """All nodes number"""
        nodes = {node for com in pred_comms + ground_truth for node in com}
        return len(nodes)

    def h(x):
        return -1 * x * (log(x) / log(2)) if x > 0 else 0

    def H_func(comm):
        p1 = len(comm) / overlapping_nodes
        p0 = 1 - p1
        return h(p0) + h(p1)

    def h_xi_joint_yj(xi, yj):
        p11 = get_intersection(xi, yj) / overlapping_nodes
        p10 = get_difference(xi, yj) / overlapping_nodes
        p01 = get_difference(yj, xi) / overlapping_nodes
        p00 = 1 - p11 - p10 - p01

        if h(p11) + h(p00) >= h(p01) + h(p10):
            return h(p11) + h(p10) + h(p01) + h(p00)
        return H_func(xi) + H_func(yj)

    def h_xi_given_yj(xi, yj):
        return h_xi_joint_yj(xi, yj) - H_func(yj)

    def H_XI_GIVEN_Y(xi, Y):
        res = h_xi_given_yj(xi, Y[0])
        for y in Y:
            res = min(res, h_xi_given_yj(xi, y))
        return res / H_func(xi)

    def H_X_GIVEN_Y(X, Y):
        res = 0
        # for idx in tqdm(range(len(X)), desc="ComputeNMI"):
        for idx in range(len(X)):
            res += H_XI_GIVEN_Y(X[idx], Y)
        return res / len(X)

    if len(pred) == 0 or len(gt) == 0:
        return 0

    overlapping_nodes = get_overlapping(pred, gt)
    return 1 - 0.5 * (H_X_GIVEN_Y(pred, gt) + H_X_GIVEN_Y(gt, pred))
