
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def calc_hit_rank(prediction, reference, type=1):
    for i, p in enumerate(prediction):
        if reference[p] == type:
            return i+1
    print(prediction)
    print(reference)
    raise ValueError('No reference!')


def recall(predictions, references, k=1 , type=1):
    assert len(predictions) == len(references)
    total = len(references)
    hits = 0
    for p, c in zip(predictions, references):
        hits += int(calc_hit_rank(p, c, type) <= k)
    rate = hits * 100.0 / total
    interval = proportion_confint(count=hits, nobs=total, alpha=0.05)
    return rate, interval


def mean_rank(predictions, references, type=1):
    assert len(predictions) == len(references)
    ranks = []
    for p, c in zip(predictions, references):
        rank = calc_hit_rank(p, c, type)
        ranks.append(rank)
    m, h = mean_confidence_interval(ranks)

    return m, h


def mean_reciprocal_rank(predictions, references, type=1):
    assert len(predictions) == len(references)
    ranks = []
    for p, c in zip(predictions, references):
        rank = calc_hit_rank(p, c, type)
        ranks.append(1.0 / rank)
    m, h = mean_confidence_interval(ranks)

    return m, h

def evaluate_recall(test_set, prediction_ids):
    predictions, references = [], []
    for i, data in enumerate(test_set):
        comments = list(data['candidate'].keys())
        candidates = []
        ids = prediction_ids[i]
        for id in ids:
            candidates.append(comments[id])
        predictions.append(candidates)
        references.append(data['candidate'])
    recall_1, inter_1 = recall(predictions, references, 1)
    recall_5, inter_5 = recall(predictions, references, 5)
    recall_10, inter_10 = recall(predictions, references, 10)
    mr, mrh = mean_rank(predictions, references)
    mrr, mrrh = mean_reciprocal_rank(predictions, references)
    print("recall@1:{}+-{}\nrecall@5:{}+-{}\nrecall@10:{}+-{}\nmean rank:{}+-{}\nmean reci rank:{}+-{}".format(recall_1, inter_1, recall_5,inter_5, recall_10,inter_10, mr, mrh, mrr, mrrh))



