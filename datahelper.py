import csv
import numpy as np
from PIL import Image


class DataHelper:

    def __init__(self, batch_size, test_idx=None):
        self.test_idx = test_idx
        self.train_imgs, self.test_imgs = self._get_valid_image_files(test_idx)
        self.train_lbls, self.test_lbls = self._get_labels_no_header(test_idx)
        self.current_idx = 0
        self.batch_size = batch_size

    def get_next_batch(self):
        end_idx = self.current_idx + self.batch_size

        if end_idx > self.test_idx:
            end_idx = self.test_idx

        data = self._get_image_data(self.train_imgs[self.current_idx:end_idx])
        labels = self.train_lbls[self.current_idx:end_idx]

        self.current_idx += self.batch_size

        return (data, labels)

    def get_test_data(self, count=None):
        if count is None:
            data = self._get_image_data(self.test_imgs)
            labels = self.test_lbls
            return (data, labels)
        else:
            data = self._get_image_data(self.test_imgs[:count])
            labels = self.test_lbls[:count]
            return (data, labels)

    def _get_valid_image_files(self, test_idx=None):
        imgs = []
        with open('./data/valid_images.csv', 'rb') as f:
            reader = csv.reader(f)
            for r in reader:
                for i in r:
                    imgs.append(i)

        # TODO make this an argument
        img_dir = './data/jpeg_redundant_f160/'
        if test_idx is None:
            return [img_dir + i for i in imgs]
        else:
            train_imgs = [img_dir + i for i in imgs[:self.test_idx]]
            test_imgs = [img_dir + i for i in imgs[self.test_idx:]]
            return (train_imgs, test_imgs)

    def _get_labels_no_header(self, test_idx=None):
        labels = []
        with open('./data/valid_image_labels_no_names.csv', 'rb') as f:
            reader = csv.reader(f)
            reader.next()  # Skip header
            for r in reader:
                labels.append(r)

        if test_idx is None:
            return np.array(labels, dtype=np.float32)
        else:
            return (np.array(labels[:test_idx], dtype=np.float32),
                    np.array(labels[test_idx:], dtype=np.float32))

    def _get_image_data(self, image_files):
        tmp = []

        for f in image_files:
            img = Image.open(f)
            img.thumbnail((45, 45))
            arr = np.asarray(img, dtype=np.float32)
            tmp.append(arr / 255.0)

        return np.array(tmp)
