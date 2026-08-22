from typing import Literal

import numpy as np
import numpy.typing as npt

from .utils import label_cvms as lcvms
from .utils.validation import validate_labels, validate_pair

CVMType = Literal["dsc", "ch_btw"]


def measure(
    orig: npt.NDArray,
    emb: npt.NDArray,
    label: npt.NDArray,
    cvm: CVMType | str = "dsc",
) -> dict:
    """
    Compute label-trustworthiness and label-continuity of the embedding
    INPUT:
            ndarray: emb: embedded data
            ndarray: label: label of the original data
            str: measure: clustering validation measure (CVM) to compute (Optional)
                    Currently supports "dsc" (Distance Consistency)
    OUTPUT:
            dict: label-trustworthiness and label-continuity
    """

    orig, emb = validate_pair(orig, emb)
    label = validate_labels(label, orig.shape[0], min_classes=2)

    # change label into 0, 1, 2,....
    unique_labels = np.unique(label)
    label_dict = {}
    for i, label_single in enumerate(unique_labels):
        label_dict[label_single] = i

    int_labels = np.zeros(label.shape[0])
    for i in range(label.shape[0]):
        int_labels[i] = label_dict[label[i]]
    label_num = len(unique_labels)

    cvm_name = cvm.lower()
    cvms = {"dsc": lcvms.dsc_normalize, "ch_btw": lcvms.btw_ch}
    if cvm_name not in cvms:
        allowed = ", ".join(sorted(cvms.keys()))
        raise ValueError(f"Invalid cvm '{cvm}'. Allowed values: {allowed}")
    cvm_fn = cvms[cvm_name]

    # compute the label-pairwise cvm of the original data
    raw_cvm_mat = np.zeros((label_num, label_num))
    emb_cvm_mat = np.zeros((label_num, label_num))

    for label_i in range(label_num):
        for label_j in range(label_i + 1, label_num):
            # raw data of a pair of labels
            filter_label = np.logical_or(int_labels == label_i, int_labels == label_j)
            raw_pair = orig[filter_label]
            emb_pair = emb[filter_label]
            # label of the raw data of a pair of labels
            raw_pair_label = int_labels[filter_label]
            emb_pair_label = int_labels[filter_label]

            # change the label to 0 and 1
            raw_pair_label[raw_pair_label == label_i] = 0
            raw_pair_label[raw_pair_label == label_j] = 1
            emb_pair_label[emb_pair_label == label_i] = 0
            emb_pair_label[emb_pair_label == label_j] = 1

            # compute cvm
            raw_cvm_mat[label_i, label_j] = cvm_fn(raw_pair, raw_pair_label)
            emb_cvm_mat[label_i, label_j] = cvm_fn(emb_pair, emb_pair_label)

    # compute the label-trustworthiness and label-continuity score
    lt_mat = raw_cvm_mat - emb_cvm_mat
    lt_mat[lt_mat < 0] = 0
    lt = 1 - np.sum(lt_mat) / (label_num * (label_num - 1) / 2)

    lc_mat = emb_cvm_mat - raw_cvm_mat
    lc_mat[lc_mat < 0] = 0
    lc = 1 - np.sum(lc_mat) / (label_num * (label_num - 1) / 2)

    # set the dictionary to return
    return {"label_trustworthiness": float(lt), "label_continuity": float(lc)}
