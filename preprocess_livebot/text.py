import os, jieba, json
MIN_DANMU_LEN = 2
MAX_DANMU_LEN = 50

def format_time(str):

    [prev, _] = str.split(".")
    [hour, min, sec] = [int(x) for x in prev.split(":")]
    total_sec = 3600 * hour + 60 * min + sec
    return total_sec


def fenci(danmu):
    """
    :param danmu_list: list of sent to be processed
    :return: fenci_list: list of sent with spaces.
    """

    fenci_line = list(jieba.cut(danmu.replace(' ', '')))
    fenci_line = " ".join(fenci_line)
    return fenci_line.strip()

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def danmu_filter_2rounds(danmu_data, top_n=50, dist_gate=0.6):
    """

    :param danmu_list: danmu_data with danmu, time pair
    :return: new list which get rid of duplicate danmu. using jaccard similarity for second round filter.
    """
    danmu_list = [x[0] for x in danmu_data]
    new_danmu_list = []
    danmu_dict = {}
    result = []
    for danmu in danmu_list:
        if danmu in danmu_dict.keys():
            danmu_dict[danmu] += 1
        else:
            danmu_dict[danmu] = 0

    danmu_dict = sorted(danmu_dict.items(), key=lambda n: n[1], reverse=True)

    top_n_list = []
    normal = []

    for popular_danmu in danmu_dict[:top_n]:
        top_n_list.append(popular_danmu[0])
    new_danmu_list.extend(top_n_list)

    for normal_danmu in danmu_dict[top_n:]:
        normal.append(normal_danmu[0])
    c_n = 0
    for danmu in normal:
        canAppend = True
        for popular_danmu in top_n_list:

            dist = 1 - jaccard_similarity(popular_danmu, danmu)

            if dist < dist_gate:
                # means 2 sentences are about the same stuff.
                canAppend = False
                c_n += 1
                # print("{} vs {}".format(popular_danmu, danmu))
            # if 0.4 < dist < 0.5:
            #     print("{} vs {}".format(popular_danmu, danmu))
        if canAppend:
            new_danmu_list.append(danmu)
    d_map = {}
    for danmu, time in danmu_data:
        if (danmu not in new_danmu_list) or (danmu in d_map.keys()):
            continue
        result.append((danmu, time))
        d_map[danmu] = 0

    return result

class Dict(object):
    def __init__(self):
        self.word2id = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<&&&>': 4}
        self.id2word = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>', 4: '<&&&>'}
        self.frequency = {}

    def add(self, s):
        ids = []
        for w in s:
            if w in self.word2id:
                id = self.word2id[w]
                self.frequency[w] += 1
            else:
                id = len(self.word2id)
                self.word2id[w] = id
                self.id2word[id] = w
                self.frequency[w] = 1
            ids.append(id)
        return ids

    def transform(self, s):
        ids = []
        for w in s:
            if w in self.word2id:
                id = self.word2id[w]
            else:
                id = self.word2id['<UNK>']
            ids.append(id)
        return ids

    def prune(self, k):
        sorted_by_value = sorted(self.frequency.items(), key=lambda kv: -kv[1])
        newDict = Dict()
        newDict.add(list(zip(*sorted_by_value))[0][:k])
        return newDict


    def save(self, fout):
        return json.dump({'word2id': self.word2id, 'id2word': self.id2word}, fout, ensure_ascii=False)

    def load(self, fin):
        datas = json.load(fin)
        self.word2id = datas['word2id']
        self.id2word = datas['id2word']

    def __len__(self):
        return len(self.word2id)

def get_danmu_list_with_time(file):
    """
    :param file: danmu avid file
    :return: danmu data : dict
    """
    danmu_data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.startswith("Dialogue"):
                danmu = line.split("}")[-1]
                time = line.split(",")[1]
                time = format_time(time)
                danmu = fenci(danmu)
                if len(danmu) < MIN_DANMU_LEN or len(danmu) > MAX_DANMU_LEN:
                    continue
                danmu_data.append((danmu,time))
               # danmu_data.append({'danmu': danmu, 'time': time})

    return danmu_data


if __name__ == "__main__":
    file = '../data/video/airplane/4分钟看《航班蛇患》，飞机上爬出400条毒蛇，很多人童年噩梦！.ass'
    get_danmu_list_with_time(file)