import torch,time,json

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

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, vocabs, rev_vocabs, img_path, is_train=True, imgs=None):
        print("starting load...")
        start_time = time.time()
        self.datas = load_from_json(open(data_path, 'r', encoding='utf8'))
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = torch.load(open(img_path, 'rb'))
        print("loading time:", time.time() - start_time)

        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.max_len = 20
        self.n_img = 5
        self.n_com = 5

        self.vocab_size = len(self.vocabs)

        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        if 'vid' in data.keys():
            video_id, video_time = int(data['vid']), int(data['time'])
        else:
            video_id, video_time = int(data['video']), int(data['time'])
        X = self.load_imgs(video_id, video_time)
        T = self.load_comments(data['context'])

        if not self.is_train:
            if "target" in data.keys():
                comment = data['target'][0]
            else:
                comment = data['comment'][0]
        else:
            if "target" in data.keys():
                comment = data['target']
            else:
                comment = data['comment']
        Y = DataSet.padding(comment, self.max_len, self.vocabs)

        return X, Y, T

    def _get_img_and_candidate(self, index):
        data = self.datas[index]
        video_id, video_time = int(data['vid']), int(data['time'])

        X = self.load_imgs(video_id, video_time)
        T = self.load_comments(data['context'])

        Y = [DataSet.padding(c, self.max_len, self.vocabs) for c in data['candidate']]
        return X, torch.stack(Y), T, data

    def get_img_and_candidate(self, index):
        data = self.datas[index]
        if 'vid' in data.keys():
            video_id, video_time = int(data['vid']), int(data['time'])
        else:
            video_id, video_time = int(data['video']), int(data['time'])
        if "target" in data.keys():
            comment = data['target'][0]
        else:
            comment = data['comment'][0]
        X = self.load_imgs(video_id, video_time)
        T = self.load_comments(data['context'])
        Y = DataSet.padding(comment, self.max_len, self.vocabs)
        candidate = [DataSet.padding(c, self.max_len, self.vocabs) for c in data['candidate']]


        return X, Y, T, data, torch.stack(candidate)


    def load_imgs(self, video_id, video_time):
        if self.n_img == 0:
            return torch.stack([self.imgs[0][0].fill_(0.0) for _ in range(5)])

        surroundings = [0, -1, 1, -2, 2, -3, 3, -4, 4]
        X = []
        for t in surroundings:
            if video_time + t >= 0 and video_time + t < len(self.imgs[video_id]):
                X.append(self.imgs[video_id][video_time + t])
                if len(X) == self.n_img:
                    break
        if len(X) < 5:
            n_pad = 5-len(X)
            pad = list(self.imgs[video_id].values())[-n_pad:]
            X.extend(pad)
            #print("video id {}, video time {}".format(video_id, video_time))

        # if not X:
        #     X = list(self.imgs[video_id].values())[-5:]
        #     print("video id {}, video time {}".format(video_id, video_time))
        return torch.stack(X)

    def load_comments(self, context):

        if self.n_com == 0:
            return torch.LongTensor([1]+[0]*self.max_len*5+[2])
        return DataSet.padding(context, self.max_len*self.n_com, self.vocabs)

    @staticmethod
    def padding(data, max_len, vocabs):
        data = data.split()
        if len(data) > max_len-2:
            data = data[:max_len-2]

        Y = list(map(lambda t: vocabs.get(t, 3), data))
        Y = [1] + Y + [2]
        length = len(Y)
        Y = torch.cat([torch.LongTensor(Y), torch.zeros(max_len - length).long()])
        return Y

    @staticmethod
    def transform_to_words(ids, rev_vocabs):
        words = []
        for id in ids:
            if id == 2:
                break
            words.append(rev_vocabs[str(id.item())])
        return "".join(words)


def get_dataset(data_path, vocabs, rev_vocabs, img_path, is_train=True, imgs=None):
    return DataSet(data_path, vocabs, rev_vocabs, img_path=img_path, is_train=is_train, imgs=imgs)

def get_dataloader(dataset, batch_size, is_train=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)
