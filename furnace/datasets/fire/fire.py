import numpy as np

from datasets.BaseDataset import BaseDataset


class Fire(BaseDataset):
    trans_labels = [0, 1]

    @classmethod
    def get_class_colors(*args):
        return [[0,0,0],[255,0,0]]

    @classmethod
    def get_class_names(*args):
        return ['background','smoke']

    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        print('Trans', name, 'to', new_name, '    ',
              np.unique(np.array(pred, np.uint8)), ' ---------> ',
              np.unique(np.array(label, np.uint8)))
        return label, new_name