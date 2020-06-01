
import json
import time
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer

def load_from_json(fin):
    datas = []
    for line in fin:
        data = json.loads(line)
        datas.append(data)
    return datas

def dump_to_json(datas, fout):
    for data in datas:
        fout.write(json.dumps(data, sort_keys=True, separators=(',', ': '), ensure_ascii=False))
        fout.write('\n')
    fout.close()

def get_comment_set(comment_list):
    comment_set = {}
    for comment in comment_list:
        if comment in comment_set:
            comment_set[comment] += 1
        else:
            comment_set[comment] = 1
    comment_set = sorted(comment_set.items(), key=lambda kv: -kv[1])
    comment_set = list(zip(*comment_set))[0]
    return comment_set

def get_correct_set(news, candidate_set):
    for comment in news['target']:
        if comment not in candidate_set:
            candidate_set[comment] = 1
    return candidate_set

def get_popular_set(comment_set, candidate_set, k):
    for comment in comment_set[:k]:
        if comment not in candidate_set:
            candidate_set[comment] = 2
    return candidate_set

def get_random_set(comment_set, candidate_set, k):
    while len(candidate_set) < k:
        rand = random.randint(0, len(comment_set) - 1)
        if comment_set[rand] not in candidate_set:
            candidate_set[comment_set[rand]] = 3
    return candidate_set

def get_plausible_set(comment_set, candidate_set, k, query_tfidf, comment_tfidf):
    matrix = (query_tfidf * comment_tfidf.transpose()).todense()
    ids = np.array(np.argsort(-matrix, axis=1))[0]
    for id in ids[:k]:
        if comment_set[id] not in candidate_set:
            candidate_set[comment_set[id]] = 3
    return candidate_set

def get_candidate_set(fin, fout, comment_list, tvec, comment_tfidf, inv_map):
    datas = load_from_json(fin)
    newdatas = []
    for data in datas:
        candidate_set = {}
        vid_title = inv_map[int(data['vid'])]
        candidate_set = get_correct_set(data, candidate_set)
        candidate_set = get_popular_set(comment_list, candidate_set, 20)
        candidate_set = get_plausible_set(comment_list, candidate_set, 20, tvec.transform([vid_title]), comment_tfidf)
        # candidate_set = get_plausible_set(comment_list, candidate_set, 20, tvec.transform([data['context']]), comment_tfidf)

        candidate_set = get_random_set(comment_list, candidate_set, 100)
        newdatas.append({'vid': data['vid'], 'time': data['time'],
                         'context': data['context'], 'target': data['target'],
                         'candidate': candidate_set})
    dump_to_json(newdatas, fout)

def prepare_candidate(test_context_file, out):
    comments = []
    with open("../livebot_data/train.json", 'r') as f:
        train_data = json.load(f)
    vids = train_data.keys()
    for vid in vids:
        vid_data = train_data[vid]
        for danmu in vid_data:
            comments.append(danmu['danmu'])

    comment_set = get_comment_set(comments)
    tvec = TfidfVectorizer()
    tvec = tvec.fit(comment_set)
    comment_tfidf = tvec.transform(comment_set)
    vid_map_f = '../livebot_data/video_map.json'
    vid_map = json.load(open(vid_map_f, 'r'))
    inv_map = {v: k for k, v in vid_map.items()}
    get_candidate_set(open(test_context_file, 'r', encoding='utf8'),
                      open(out, 'w', encoding='utf8'),
                      comment_set, tvec, comment_tfidf, inv_map)
# if __name__ == '__main__':
#     comments = []
#     with open("train_unique2.json",'r') as f:
#         train_data = json.load(f)
#     vids = train_data.keys()
#     for vid in vids:
#         vid_data = train_data[vid]
#         for danmu in vid_data:
#             comments.append(danmu['danmu'])
#
#     comment_set = get_comment_set(comments)
#     tvec = TfidfVectorizer()
#     tvec = tvec.fit(comment_set)
#     comment_tfidf = tvec.transform(comment_set)
#     vid_map_f = 'video_map.json'
#     vid_map = json.load(open(vid_map_f, 'r'))
#     inv_map = {v: k for k, v in vid_map.items()}
#     get_candidate_set(open('test_context_unique2_oa.json', 'r', encoding='utf8'), open('test-candidate_oa.json', 'w', encoding='utf8'),
#                       comment_set, tvec, comment_tfidf, inv_map)
#
#     # get_candidate_set(open('dev_context.json', 'r', encoding='utf8'), open('dev-candidate.json', 'w', encoding='utf8'),
#     #                   comment_set, tvec, comment_tfidf, inv_map)
