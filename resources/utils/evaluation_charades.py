# Part of the code is from
# `https://github.com/mwray/Joint-Part-of-Speech-Embeddings/tree/main/src/evaluation/mAP.py`

import numpy as np

def softmax(x, dim=1):
    x -= np.max(x, axis= dim, keepdims=True)
    
    f_x = np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)

    return f_x

def compute_map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        #print(prec)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def get_charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    gt_array = gt_array.T
    #gt_array[gt_array != 1] = 0
    fix = submission_array.copy().T
    fix = softmax(fix, dim = 1)
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    m_ap, w_ap, m_aps = compute_map(fix, gt_array)
    return m_ap, m_ap, m_ap