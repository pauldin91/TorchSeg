#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from datasets.fire import Fire as IDataset
from network import BiSeNet

logger = get_logger()

from numpy.linalg import norm

from scipy.spatial import distance

import math

def find_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def similarity(x, y):
    v = np.dot(x, y) / (norm(x) * norm(y))
    return v


def fire_angle(b):
    return np.arctan((b[0] + b[3]) * 1.0 / (b[1] + b[2] + 1e-8)) * 180 / np.pi


def maximum_filter(n, img):
    # Creates the shape of the kernel
    size = (n, n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the maximum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)
    # kernel = cv2.getStructuringElement(shape, size)

    imgResult = cv2.erode(imgResult, kernel)
    return imgResult


def distances(centers):
    distance = []

    for i in range(len(centers)):
        for j in range(i+1,len(centers)):
            if j!=i:
                distance.append(find_distance(centers[i], centers[j]))

    return distance

import cv2

def cosine_similarity(mask,pred):
    mask = np.sort(mask,axis=0)
    pred = np.sort(pred,axis=0)
    if mask.shape[1] > 1:
        mm = np.concatenate(mask).ravel()
    else:
        mask.ravel()
    if pred.shape[1] > 1:
        pp = np.concatenate(pred).ravel()
    else:
        pp=pred.ravel()
    mlength = max(len(mm),len(pp))
    mm = np.pad(mm,((0, mlength - mm.shape[0])), mode='constant')
    pp = np.pad(pp,((0, mlength - pp.shape[0])), mode='constant')
    v = np.dot(mm,pp)/(norm(mm)*norm(pp))
    return v

def extract_metrics(mat):
    mat = np.asarray(mat,dtype='uint8')

    contours, hierarchy = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 2, True)
        boundRect[i] = np.asarray(cv2.boundingRect(contours_poly[i]))
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    blur = maximum_filter(17, mat)

    number_of_fires, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    deviation = np.std(distances(centers))
    if len(centers)<=1:
        deviation = 0.0

    areas = np.zeros(len(number_of_fires))
    for j, c in enumerate(number_of_fires):
        area = cv2.contourArea(c)
        areas[j] = area

    if len(number_of_fires)==0:
        areas = np.asarray([0.0])
    return boundRect,len(number_of_fires),deviation,areas


def normalize(a,b):
    return np.abs(norm(a)-norm(b))

def normalize_percent(a,b):
    if(max(norm(a),norm(b)))!=0.0:
        return min(norm(a),norm(b))*1.0/max(norm(a),norm(b))
    return 0.0

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        pred = self.sliding_eval(img, config.eval_crop_size,
                                 config.eval_stride_rate, device=device)

        p, q, r, l = extract_metrics(pred)
        m, n, o, k = extract_metrics(label)
        area_l_mean = l.mean()

        area_k_mean = k.mean()

        area = normalize(area_l_mean,area_k_mean)
        deviation = normalize(r,o)
        no_fires = np.abs(q-n)

        area_norm = normalize_percent(area_l_mean, area_k_mean)
        deviation_norm = normalize_percent(r, o)
        no_fires_norm = normalize_percent(q, n)
        no_fires_arr = [n,q]

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes,
                                                       pred,
                                                       label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp,'area':area,'no_fires':no_fires,'deviation': deviation,'no_fires_arr' : no_fires_arr,
                        'area_norm':area_norm,'deviation_norm' : deviation_norm, 'no_fires_norm': no_fires_norm }

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict


    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        area = 0.0
        no_fires = 0.0
        no_fires_norm =0.0
        deviation_norm = 0.0
        area_norm = 0.0
        no_fires_arr = [0.0,0.0]
        deviation = 0.0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            area += d['area']
            no_fires += d['no_fires']
            no_fires_arr[0] += d['no_fires_arr'][0]
            no_fires_arr[1] += d['no_fires_arr'][1]
            deviation += d['deviation']
            area_norm += d['area_norm']
            deviation_norm += d['deviation_norm']
            no_fires_norm += d['no_fires_norm']

            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,labeled)
        area = area*1.0/count
        deviation = deviation*1.0/count
        area_norm = area_norm*1.0/count
        deviation_norm = deviation_norm*1.0/count
        no_fires = no_fires*1.0/count
        no_fires_arr = [no_fires_arr[0]*1.0/count,no_fires_arr[1]*1.0/count]
        no_fires_norm = no_fires_norm*1.0/count
        result_line = print_iou(iu, mean_pixel_acc,area,area_norm,no_fires,no_fires_norm,
                                no_fires_arr,deviation,deviation_norm, dataset.get_class_names(), True)
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = BiSeNet(config.num_classes, is_training=False,
                      criterion=None)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    dataset = IDataset(data_setting, 'val', None)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
