import os.path
import numpy as np
import matplotlib.pyplot as plt

from data.datasets.kitti_utils import read_label, Calibration


def test_pseudo_labels():
    root = "/home/afoxxvi/data/kitti_object/training/"
    label_path = os.path.join(root, 'label_2')
    calib_path = os.path.join(root, 'calib')
    img_sets = os.path.join(root, 'ImageSets', 'train.txt')
    inst = '000018.txt'
    label_file = os.path.join(label_path, inst)
    calib_file = os.path.join(calib_path, inst)
    calib = Calibration(calib_file)
    objs = read_label(label_file)
    obj = objs[0]
    print(obj.t)
    print(calib.project_rect_to_image(obj.t.reshape(1, -1)))
    ctx, cty, ctz = obj.t
    xz, yz = ctx / ctz, cty / ctz
    ctz *= 20
    ctx = ctz * xz
    cty = ctz * yz
    nt = np.array((float(ctx), float(cty), float(ctz)))
    print(nt)
    print(calib.project_rect_to_image(nt.reshape(1, -1)))


def calc_obj_counts():
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
    loc_x = []
    loc_y = []
    loc_z = []
    for label_file in label_files:
        objs = read_label(label_file)
        obj_count = 0
        new_obj_count = 0
        for obj in objs:
            if obj.type == 'Car' or obj.type == 'Pedestrian' or obj.type == 'Cyclist':
                pass
            else:
                continue
            if obj.t[-1] <= 0 or obj.t[-1] > 60:
                continue
            obj_count += 1
            new_obj_count += 5
            loc_x.append(obj.t[0])
            loc_y.append(obj.t[1])
            loc_z.append(obj.t[2])
        obj_counts.append(obj_count)
        new_obj_counts.append(new_obj_count)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(np.array(loc_x))
    ax[0, 1].hist(np.array(loc_y))
    ax[1, 0].hist(np.array(loc_z))
    plt.show()


if __name__ == "__main__":
    test_pseudo_labels()
    pass
