import os, json, torch, jieba, string, random
from add_candidate import prepare_candidate
from text import get_danmu_list_with_time
from visual import build_resnet, get_resnet_feature
import numpy as np

MIN_DANMU_NUMBER = 0
np.random.seed(4321)


def save_pretrain_imgs(img_dir, output_file):
    imgs = {}
    model = build_resnet()
    model.cuda()
    video_id = 0
    for i in range(len(os.listdir(img_dir))):
        imgs[i] = {}
        dir = os.path.join(img_dir, str(i))
        for j in range(len(os.listdir(dir))):
            img_file = os.path.join(dir, "%d.bmp" % (j + 1))
            imgs[i][j] = get_resnet_feature(img_file, model)
            video_id += 1
        print("%d/%d" % (i + 1, len(os.listdir(img_dir))))

    print(video_id)
    torch.save(imgs, open(output_file, 'wb'))


def randomString(stringLength=8):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))



def extract_frame(filename, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cmd_str = 'ffmpeg -i "%s" -r 1/1 -s 224x224 -f image2 %s/' % (filename, outdir) + '%d.bmp'
    print(cmd_str)
    os.system(cmd_str)


def run(data_root, out_filename, filter=False):
    print("-Processing raw danmu data, min danmu number is {}, filter switch is {}".format(MIN_DANMU_NUMBER, filter))
    video_id = 0
    duplicate_num = 0
    # video id starts from 0
    danmu_data = {}
    vid_title_map = {}

    for dir, _, filenames in os.walk(data_root):
        for filename in filenames:
            if filename.endswith('flv'):
                title = filename[:-4]
                danmu_file = os.path.join(dir, title + ".ass")
                if os.path.exists(danmu_file):
                    if title in vid_title_map.keys():
                        video_id += 1
                        duplicate_num += 1
                        continue
                    else:
                        vid_title_map[title] = video_id
                    data = get_danmu_list_with_time(danmu_file)
                    data = [{'danmu': x[0], 'time': int(x[1])} for x in data]
                    danmu_data[video_id] = data
                    extract_frame(os.path.join(dir, filename), 'img/%d' % video_id)
                    video_id += 1

    print("-Finish processing, originally {} videos, {} videos are duplicate, end up with {} videos".format(video_id,
                                                                                                            duplicate_num,
                                                                                                            video_id - duplicate_num))

    with open("video_map.json",'w', encoding='utf-8') as vmap:
        print("-Writing video title-id map")
        json.dump(vid_title_map, vmap)
    with open(out_filename,'w', encoding='utf-8') as output:
        json.dump(danmu_data, output)
    return danmu_data


def ids_to_json(ids, data, out_filename):
    selecetion = {x: data[x] for x in ids}
    with open(out_filename, 'w', encoding='utf-8') as output:
        json.dump(selecetion, output)


def load_from_json(fin):
    datas = []
    for line in fin:
        data = json.loads(line)
        datas.append(data)
    return datas


def split(data_file):
    print("-Splitting danmu data into train / dev (100 videos) / test (100 videos) set")
    with open(data_file) as f:
        data = json.load(f)
    valid_ids = list(data.keys())
    np.random.shuffle(valid_ids)
    train, dev, test = valid_ids[:-200], valid_ids[-200:-100], valid_ids[-100:]
    ids_to_json(train, data, 'train.json')
    ids_to_json(dev, data, 'dev.json')
    ids_to_json(test, data, 'test.json')


def dump_to_json(datas, fout):
    for data in datas:
        fout.write(json.dumps(data, sort_keys=True, separators=(',', ': '), ensure_ascii=False))
        fout.write('\n')
    fout.close()


def cal_len(data_file):
    with open(data_file) as f:
        data = json.load(f)
    num_danmu = 0
    num_word = 0

    for vid, danmu_data in data.items():
        for i, danmu in enumerate(danmu_data):
            danmu = danmu['danmu']
            num_danmu += 1
            num_word += len(danmu.split())
    return num_danmu, num_word, num_word / float(num_danmu)


def form_training_context(data_file, out_file, num_context=5):
    with open(data_file) as f:
        data = json.load(f)
    new_data = []
    for vid, danmu_data in data.items():
        for i, danmu in enumerate(danmu_data):
            max_len = len(danmu_data) - 1
            surrounding = [-5, -4, -3, -2, -1, 1, 2]
            target_danmu = danmu['danmu']
            current_time = danmu['time']
            context = []

            for j in surrounding:
                # building context
                context_index = i + j
                if 0 <= context_index <= max_len:

                    if len(context) == num_context:
                        # max context size is 5
                        break
                    context.append(danmu_data[context_index]['danmu'])

            context = " <&&&> ".join(context)
            new_data.append({"target": target_danmu, "context": context, "vid": vid, "time": current_time})


    dump_to_json(new_data, open(out_file, 'w', encoding='utf8'))


def form_test_set(data_file, outfile, num_context=5, samples=5000):
    with open(data_file) as f:
        data = json.load(f)
    test_set = []
    surrounding = [0, -1, 1, -2, 2, -3, 3]
    for vid, danmu_data in data.items():
        time_comment_map = {}
        for danmu in danmu_data:
            target_danmu = danmu['danmu']
            time = danmu['time']
            if time in time_comment_map.keys():
                time_comment_map[time].append(target_danmu)
            else:
                time_comment_map[time] = [target_danmu]
        keys = list(time_comment_map.keys())
        for t in keys:
            if len(time_comment_map[t]) >= num_context:
                comments = time_comment_map[t][:num_context]
                # target comments, 5 gt
                context_pool = []

                for s in surrounding:
                    t_candidate = t + s
                    if t_candidate in keys:
                        context_pool.extend([x for x in time_comment_map[t_candidate] if x not in comments])
                if len(context_pool) < num_context:
                    continue
                np.random.shuffle(context_pool)
                context = context_pool[:num_context]

                context = " <&&&> ".join(context)
                test_set.append({'vid': vid, 'time': t,
                                 'context': context, 'target': comments})
    if len(test_set) > samples:
        np.random.shuffle(test_set)
        test_set = test_set[:samples]

    print("{} test sample built".format(len(test_set)))
    dump_to_json(test_set, open(outfile, 'w', encoding='utf8'))


def dump_text(data, file):
    data = [x + '\n' for x in data]
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines(data)


def prepare_onmt_train(train_file, dev_file, out_dir):
    train_data = load_from_json(open(train_file, 'r', encoding='utf-8'))
    dev_data = load_from_json(open(dev_file, 'r', encoding='utf-8'))
    train_src = [x['context'] for x in train_data]
    train_tgt = [x['target'] + " time{}:{}:{}".format(str(x['vid']), str(x['time']), randomString()) for x in
                 train_data]
    valid_src = [x['context'] for x in dev_data]
    valid_tgt = [x['target'][0] + " time{}:{}:{}".format(str(x['vid']), str(x['time']), randomString()) for x in
                 dev_data]
    dump_text(train_src, os.path.join(out_dir, 'train_src.txt'))
    dump_text(train_tgt, os.path.join(out_dir, 'train_tgt.txt'))
    dump_text(valid_src, os.path.join(out_dir, 'valid_src.txt'))
    dump_text(valid_tgt, os.path.join(out_dir, 'valid_tgt.txt'))


def prepare_onmt_test(test_file, out_dir):
    test_data = load_from_json(open(test_file, 'r', encoding='utf-8'))
    test_src, test_tgt = [], []
    for data in test_data:
        for c in data['candidate']:
            test_src.append(data['context'])
            test_tgt.append(c + " time{}:{}:{}".format(str(data['vid']), str(data['time']), randomString()))
    dump_text(test_src, os.path.join(out_dir,'test_src.txt'))
    dump_text(test_tgt, os.path.join(out_dir,'test_tgt.txt'))




if __name__ == "__main__":

    onmt_data_dir = '../onmt_data'
    onmt_test_dir = '../onmt_data/test/'
    if not os.path.exists(onmt_data_dir):
        os.mkdir(onmt_data_dir)
    if not os.path.exists(onmt_test_dir):
        os.mkdir(onmt_test_dir)
    train_source = '../livebot_data/train.json'
    dev_source = '../livebot_data/dev.json'
    test_source = '../livebot_data/test.json'
    form_training_context(train_source, os.path.join(onmt_data_dir, 'train_context.json'), num_context=5)
    form_test_set(dev_source, os.path.join(onmt_data_dir,'dev_context.json'), num_context=5)
    form_test_set(test_source, os.path.join(onmt_data_dir,'test_context.json'), num_context=5)

    prepare_onmt_train(os.path.join(onmt_data_dir, 'train_context.json'), os.path.join(onmt_data_dir,'dev_context.json'), onmt_data_dir)
    prepare_candidate(os.path.join(onmt_data_dir,'test_context.json'), os.path.join(onmt_test_dir, 'test-candidate.json'))
    prepare_onmt_test(os.path.join(onmt_test_dir, 'test-candidate.json'), onmt_test_dir)

