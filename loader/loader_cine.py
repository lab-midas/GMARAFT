import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
import cv2
from random import shuffle
import h5py


class CineDatasetPairwise(data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.data_root_dir = config['data_dir']
        self.sampling = config['sampling']
        self.csv = config['csv']
        self.R_list = config['R_list']
        self.data_amount = 500000 if mode == 'train' else 28
        # initialize lists
        self.list_info = []
        self.img_list = []
        self.img_fully_list = []
        # fill data lists
        self.create_list_data()
        print(len(self.list_info), 'is the list of the files length')

    def create_list_data(self):
        data_info = pd.read_csv(self.csv)
        files_all = data_info.filename.tolist()

        if self.mode == 'train':
            shuffle(files_all)
            num_per_r = len(files_all) // len(self.R_list)

        else:
            num_per_r = len(files_all)

        r = len(files_all) % len(self.R_list)
        list_R_files = []
        offset = 0
        num_used = 0
        for ind, R in enumerate(self.R_list):
            begin = ind * num_per_r + offset
            end = (ind + 1) * num_per_r + offset
            if r:
                end += 1
                r -= 1
                offset += 1
            num_used += len(files_all[begin:end])
            list_R_files.append(files_all[begin:end])

        print('given files', len(files_all), 'used files', num_used)
        print('number of subjects per acceleration', num_per_r)
        print('dict_R_files', list_R_files)
        n = 0
        index_us = 0
        index_fully = 0
        for ind, files in enumerate(list_R_files):
            R = self.R_list[ind]
            print(f'reading R{R} files {files}')
            for ind, filename in enumerate(files):
                print(filename)
                if n > self.data_amount - 1:
                    break
                f_name = filename.split('.')[0]

                with h5py.File(f"{self.data_root_dir}/h5/{f_name}.h5", 'r') as f:
                    data_fully = np.abs(f['dImgC'][:])

                if R == 1:
                    data = np.abs(data_fully)
                else:
                    data = np.abs(np.load(f"{self.data_root_dir}/{self.sampling}/R{R}/{f_name}_img.npy"))  # todo:changed

                data = np.abs(data)
                data_fully = np.abs(data_fully)
                n_slices, n_frames = data_fully.shape[:2]
                self.img_fully_list.append(data_fully)
                self.img_list.append(data)
                n = self.fill_list_train(n, n_slices, n_frames, index_us, index_fully)
                index_us += 1
                index_fully += 1

    def fill_list_train(self, n, n_slices, n_frames, index_us, index_fully):
        for z in range(n_slices):
            for t1 in range(n_frames):
                for t2 in range(n_frames):
                    if n > self.data_amount - 1:
                        break
                    dict = {}
                    dict['img_us_idx'] = index_us
                    dict['img_idx'] = index_fully
                    dict['z'] = z
                    dict['t1'] = t1
                    dict['t2'] = t2
                    self.list_info.append(dict)
                    n += 1
        return n


    def __getitem__(self, index):
        subject_dict = self.list_info[index]
        data = self.img_list[subject_dict['img_us_idx']]
        data_fully = self.img_fully_list[subject_dict['img_idx']]

        z, t1, t2 = subject_dict['z'], subject_dict['t1'], subject_dict['t2']

        idx1, idx2 = self.get_neighboring_frames(data.shape[1], t2)
        ref_fully, mov_fully = self.img_preprocessing(data_fully[z, t1]), self.img_preprocessing(data_fully[z, t2])

        ref_fully = torch.from_numpy(ref_fully[None]).float()
        mov_fully = torch.from_numpy(mov_fully[None]).float()
        context_full = np.stack((self.img_preprocessing(data_fully[z, idx1]),
                                 self.img_preprocessing(data_fully[z, t2]),
                                 self.img_preprocessing(data_fully[z, idx2])), axis=0)
        context_full = torch.from_numpy(context_full).float()

        ref, mov = self.img_preprocessing(data[z, t1]), self.img_preprocessing(data[z, t2])

        context = np.stack((self.img_preprocessing(data[z, idx1]),
                            self.img_preprocessing(data[z, t2]),
                            self.img_preprocessing(data[z, idx2])), axis=0)

        ref = torch.from_numpy(ref[None]).float()
        mov = torch.from_numpy(mov[None]).float()
        context = torch.from_numpy(context).float()

        return (ref, mov, context), (ref_fully, mov_fully, context_full)

    def img_preprocessing(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def get_neighboring_frames(self, n_frames, t):
        idx1, idx2 = t - 1, t + 1
        if t == 0:
            idx1 = 24
        if t == (n_frames - 1):
            idx2 = 0
        return idx1, idx2

    def __rmul__(self, v):
        self.list_info = v * self.list_info
        return self

    def __len__(self):
        return len(self.list_info)



