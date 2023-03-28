import os.path
import numpy as np
import matplotlib.pyplot as plt

from data.datasets.kitti_utils import read_label


def load_kitti_labels():
    root = "/home/afoxxvi/data/kitti_object/training/"
    label_path = os.path.join(root, 'label_2')
    calib_path = os.path.join(root, 'calib')
    img_sets = os.path.join(root, 'ImageSets', 'train.txt')
    label_files, calib_files = [], []
    for line in open(img_sets, 'r'):
        label_files.append(os.path.join(label_path, line[:6] + ".txt"))
        calib_files.append(os.path.join(calib_path, line[:6] + ".txt"))
    obj_counts = []
    new_obj_counts = []
    for label_file in label_files:
        objs = read_label(label_file)
        obj_count = 0
        new_obj_count = 0
        for obj in objs:
            if obj.type == 'Car':
                new_objs = 4
            elif obj.type == 'Pedestrian' or obj.type == 'Cyclist':
                new_objs = 0
            else:
                continue
            if obj.t[-1] <= 0 or obj.t[-1] > 60:
                continue
            obj_count += 1
            new_obj_count += new_objs + 1
        obj_counts.append(obj_count)
        new_obj_counts.append(new_obj_count)
    np1 = np.array(obj_counts)
    np2 = np.array(new_obj_counts)
    print(np1.max(initial=0))
    print(np2.max(initial=0))
    print(np1.mean())
    print(np2.mean())
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(np1)
    ax[1].hist(np2)
    plt.show()


if __name__ == "__main__":
    load_kitti_labels()
    pass
