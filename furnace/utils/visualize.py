import numpy as np
import cv2
import scipy.io as sio

from numpy.linalg import norm

def normalize(a,b):
    return norm(a-b)/(max(norm(a),norm(b))+0.001)

def set_img_color(colors, background, img, gt, show255=False):
    for i in range(1, len(colors)):
        if i != background:
            img[np.where(gt == i)] = colors[i]
    if show255:
        img[np.where(gt == 255)] = 255
    return img


def show_prediction(colors, background, img, pred):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred)
    final = np.array(im)
    return final


def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    set_img_color(colors, background, im1, clean)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt,True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final


def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1, 3)) * 255).tolist()[0])

    return colors


def get_ade_colors():
    colors = sio.loadmat('./color150.mat')['colors']
    colors = colors[:, ::-1, ]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0, [0, 0, 0])

    return colors


def print_iou(iu, mean_pixel_acc,area,area_norm,no_fires,no_fires_norm,no_fires_arr,deviation,deviation_norm, class_names=None, show_no_back=False,no_print=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append(
            '%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f' % (
                'mean_IU', mean_IU * 100, 'mean_IU_no_back',
                mean_IU_no_back * 100,
                'mean_pixel_ACC', mean_pixel_acc * 100,'area',area,'no_fires',no_fires,'deviation',deviation,'no_fires_actual',no_fires_arr[0],'no_fires_pred',no_fires_arr[1],
                'area_norm',area_norm,'no_fires_norm',no_fires_norm,'deviation_norm',deviation_norm
            ))
    else:
        print(mean_pixel_acc)
        lines.append(
            '%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f\t%-8s\t%.3f' % (
                'mean_IU', mean_IU * 100, 'mean_pixel_ACC',
                mean_pixel_acc * 100,'area',area,'no_fires',no_fires,'deviation',deviation,'no_fires_actual',no_fires_arr[0],'no_fires_pred',no_fires_arr[1],
                'area_norm',area_norm,'no_fires_norm',no_fires_norm,'deviation_norm',deviation_norm))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line
