from random import shuffle
import cv2
import os
from tqdm import tqdm
import fjn_util
import argparse


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--data', type=str, default='nwpu', choices=['nwpu', 'aid', 'rsscn7'])
    parser.add_argument('--train_num', type=int, default=620)
    parser.add_argument('--test_num', type=int, default=50)

    return parser.parse_args()


def make_datasets(data_path, train_label_path, test_label_path, train_num, test_num, train_no, test_no):
    for root, dirs, files in os.walk(data_path, topdown=False):
        shuffle(files)
        i = 1
        pbar = tqdm(total=train_num + test_num)
        for item in files:
            img_label = cv2.imread(os.path.join(root, item), cv2.IMREAD_UNCHANGED)
            if img_label.shape[0] == 256 and img_label.shape[1] == 256:
                if i <= train_num:
                    cv2.imwrite(os.path.join(train_label_path, str(train_no).zfill(5) + ".png"), img_label)  # 保存训练标签
                    train_no += 1
                elif i <= train_num + test_num:
                    cv2.imwrite(os.path.join(test_label_path, str(test_no).zfill(5) + ".png"), img_label)  # 保存训练标签
                    test_no += 1
                i += 1
            pbar.update(1)
        pbar.close()

    return train_no, test_no


if __name__ == '__main__':
    paras = get_parameters()

    data_select_list = ['forest', 'golf_course', 'baseball_diamond', 'rectangular_farmland', 'circular_farmland',
                        'freeway', 'overpass', 'sparse_residential', 'dense_residential', 'medium_residential',
                        'mobile_home_park', 'roundabout', 'railway', 'runway', 'stadium', 'tennis_court']

    train_label_path = os.path.join(paras.save_path, r'train/label/')
    test_label_path = os.path.join(paras.save_path, r'test/label/')

    fjn_util.make_folder(train_label_path, test_label_path)

    train_no, test_no = 1, 1

    for item in data_select_list:
        assert os.path.exists(os.path.join(paras.root_path, item)), 'please check if the path: "{}" exists.'.format(
            os.path.join(paras.root_path, item))
        train_no, test_no = make_datasets(os.path.join(paras.root_path, item), train_label_path, test_label_path,
                                          paras.train_num, paras.test_num, train_no, test_no)

# python prepare.py --root_path /home/eric/PycharmProjects/NWPU-RESISC45/ --save_path ./data/
