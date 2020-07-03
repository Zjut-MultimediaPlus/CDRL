# process the dataset
import numpy as np
import math
import os

# set dataset path
DATA_DIR = '/home/sola/Dataset/SBIR/Flickr15K'
LIST_IMAGE = os.path.join(DATA_DIR,'list_images.txt')
LIST_SKETCH = os.path.join(DATA_DIR,'list_sketches.txt')

class DataStore:

    def __init__(self,batch_size,classes):
        self.image_origin, self.image_origin_index, self.image_num_per = self._read_list(LIST_IMAGE,'images')
        self.sketch_origin, self.sketch_origin_index, self.sketch_num_per = self._read_list(LIST_SKETCH,'sketches')
        self.batch_size = batch_size
        self.batch_index = 0
        self.classes = classes
        self.epoch_is_over = False

    # def create_test_sketch(self, test_per):
    #     sketch_train = []
    #     sketch_test = []
    #     for l in range(self.classes):
    #         sketch_test.append(self.sketch_origin[range(self.sketch_origin_index[l],
    #                                                     self.sketch_origin_index[l]+test_per)])
    #         sketch_train.append(self.sketch_origin[range(self.sketch_origin_index[l]+test_per,
    #                                                      self.sketch_origin_index[l]+self.sketch_num_per[l])])
    #     sketch_test = np.array(sketch_test).reshape([-1, 3])
    #     sketch_train = np.array(sketch_train).reshape([-1, 3])
    #     self.sketch_test = sketch_test
    #     self.sketch_train = sketch_train
    #     return self.sketch_train, self.sketch_test

    # def create_pair_list(self):
    #     pair_image_list = []
    #     pair_sketch_list = []
    #     indice = np.random.permutation(self.image_origin.shape[0])
    #     image_shuffle = self.image_origin[indice]
    #     index_list = self.sketch_origin_index.copy()
    #
    #     for i in range(image_shuffle.shape[0]):
    #         pair_image_list.append(image_shuffle[i])
    #         label = int(image_shuffle[i][2])
    #         pair_sketch_list.append(self.sketch_origin[index_list[label]])
    #         if (index_list[label] == (self.sketch_origin_index[label] + self.sketch_num_per[label] - 1)):
    #             index_list[label] = self.sketch_origin_index[label]
    #         else:
    #             index_list[label] = index_list[label] + 1
    #
    #     assert pair_image_list.__len__() == pair_sketch_list.__len__(), 'error in generating image-sketch pair'
    #
    #     self.len_train = pair_image_list.__len__()
    #     self.pair_image_list = np.array(pair_image_list).reshape([-1,3])
    #     self.pair_sketch_list = np.array(pair_sketch_list).reshape([-1,3])
    #
    #     return self.pair_image_list, self.pair_sketch_list

    # get one train batch
    def get_train_batch(self, unique_batch_size = None):

        if unique_batch_size:
            self.batch_size = unique_batch_size

        assert ((self.pair_sketch_list is not None) and (self.pair_image_list is not None)), 'need create pair first'

        if self.batch_index == (math.ceil(self.len_train / self.batch_size) - 1):
            image_batch = self.pair_image_list[range(self.batch_index * self.batch_size, self.len_train)]
            sketch_batch = self.pair_sketch_list[range(self.batch_index * self.batch_size, self.len_train)]
            batch_size = self.len_train % self.batch_size
            self.batch_index = 0
            self.epoch_is_over = True
            print('epoch over')
        else:
            image_batch = self.pair_image_list[range(self.batch_index * self.batch_size, (self.batch_index + 1)* self.batch_size)]
            sketch_batch = self.pair_sketch_list[range(self.batch_index * self.batch_size, (self.batch_index + 1)* self.batch_size)]
            batch_size = self.batch_size
            self.batch_index += 1

        return image_batch, sketch_batch, batch_size

    # disorder train set and renew the training flag
    def renew(self, re_create = False):
        if re_create:
            # self.create_pair_list()
            self.create_train_pair_list()
        self.epoch_is_over = False
        self.batch_index = 0

    # split data into train set and test set
    def split_data(self, p=0.5):
        sketch_train = []
        sketch_test = []
        image_train = []
        image_test = []
        for i in range(self.classes):
            sketch_train.extend(self.sketch_origin[range(self.sketch_origin_index[i],
                                                         self.sketch_origin_index[i]+int(self.sketch_num_per[i]*p))])
            sketch_test.extend(self.sketch_origin[range(self.sketch_origin_index[i]+int(self.sketch_num_per[i]*p),
                                                        self.sketch_origin_index[i]+self.sketch_num_per[i])])
            image_train.extend(self.image_origin[range(self.image_origin_index[i],
                                                       self.image_origin_index[i]+int(self.image_num_per[i]*p))])
            image_test.extend(self.image_origin[range(self.image_origin_index[i]+int(self.image_num_per[i]*p),
                                                      self.image_origin_index[i]+self.image_num_per[i])])
        self.sketch_train = np.array(sketch_train).reshape([-1,3])
        self.sketch_test = np.array(sketch_test).reshape([-1,3])
        self.image_train = np.array(image_train).reshape([-1,3])
        self.image_test = np.array(image_test).reshape([-1,3])

        def _get_index_num_per(label):
            label_index = [0]
            l = 0
            for i in range(label.shape[0]):
                if l != label[i]:
                    label_index.extend([i])
                    l += 1
            num_per = []
            for i in range(label_index.__len__()):
                num_per.extend([label_index[i + 1] - label_index[i]])
                if i == label_index.__len__() - 2:
                    num_per.extend([label.shape[0] - label_index[i + 1]])
                    break
            return label_index, num_per

        self.sketch_train_index, self.sketch_train_num_per = _get_index_num_per(self.sketch_train[:, 2].astype(int))
        self.image_train_index, self.image_train_num_per = _get_index_num_per(self.image_train[:, 2].astype(int))
        # print(self.sketch_train[:,2])

    # create train pairs
    def create_train_pair_list(self):
        assert self.image_train is not None, 'have to split data first!!'

        pair_image_list = []
        pair_sketch_list = []
        indice = np.random.permutation(self.image_train.shape[0])
        # print(self.image_train.shape)
        image_shuffle = self.image_train[indice]
        index_list = self.sketch_train_index.copy()
        # print(index_list)

        for i in range(image_shuffle.shape[0]):
            pair_image_list.append(image_shuffle[i])
            # print(image_shuffle[i][2])
            label = int(image_shuffle[i][2])
            # print(label)
            pair_sketch_list.append(self.sketch_train[index_list[label]])
            if (index_list[label] == (self.sketch_train_index[label] + self.sketch_train_num_per[label] - 1)):
                index_list[label] = self.sketch_train_index[label]
            else:
                index_list[label] = index_list[label] + 1

        assert pair_image_list.__len__() == pair_sketch_list.__len__(), 'error in generating image-sketch pair'

        self.len_train = pair_image_list.__len__()
        self.pair_image_list = np.array(pair_image_list).reshape([-1, 3])
        self.pair_sketch_list = np.array(pair_sketch_list).reshape([-1, 3])

        return self.pair_image_list, self.pair_sketch_list


    # read list
    def _read_list(self, path, name):
        data = np.loadtxt(path,dtype=str,delimiter=' ')
        label = data[:,0].astype(int) - 1 # Flickr15K start with 1, so...
        image = data[:,1]
        m = np.vectorize(lambda x: os.path.join(DATA_DIR,name,x))
        image = m(image)
        image_index = np.array([i+1 for i in range(np.shape(image)[0])])
        indices = np.argsort(label)
        label = label[indices].reshape([-1,1])
        image = image[indices].reshape([-1,1])
        image_index = image_index[indices].reshape([-1,1])

        label_index = [0]
        l = 0
        for i in range(label.shape[0]):
            if l != label[i]:
                label_index.extend([i])
                l += 1
        num_per = []
        for i in range(label_index.__len__()):
            num_per.extend([label_index[i + 1] - label_index[i]])
            if i == label_index.__len__() - 2:
                num_per.extend([image.shape[0] - label_index[i + 1]])
                break
        return np.hstack((image_index, image, label)), label_index, num_per



# test codes
# a = DataStore(8,33)
# a.split_data(p=0.5)
# a.create_train_pair_list()
# a.create_pair_list()
# image_batch, sketch_batch, batch_size = a.get_train_batch()
# a.create_test_sketch(10)
# query = a.sketch_test

# print(a.sketch_origin)
# print(a.sketch_origin_index)
# print(a.image_num_per)
# print(a.sketch_num_per)
# print(a.pair_sketch_list.shape)
# print(a.pair_image_list.shape)
# print(image_batch)
# print(sketch_batch)
# print(batch_size)
# print(query)