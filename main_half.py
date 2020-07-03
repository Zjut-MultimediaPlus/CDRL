# train and get features
import tensorflow as tf
import network as net
import numpy as np
import data_ut_half as du
import utilities as ut
from PIL import Image

import math
import threading
import queue

_BATCH_SIZE = 8
_EPOCHS = 25
_CLASSES = 33
_INPUT_SIZE = 256
_LOGITS_DIMS = 256

lock = threading.Lock()
data = du.DataStore(_BATCH_SIZE,_CLASSES)

# p = 0.4 means that 40% samples are used on training.
data.split_data(p = 0.4)
data.create_train_pair_list()
# data.create_pair_list()
# data.create_test_sketch(10)
# database = data.image_origin
database = data.image_test
query = data.sketch_test
q = queue.Queue(maxsize=10)
eq_threads = []

_TRAIN_NUM = data.len_train

def read_image(path):
    # return path
    # return sktrans.resize(sio.imread(path), (256,256))
    return np.array(Image.open(path).convert('RGB').resize((256,256)))

def read_image_batch(paths):
    images = []
    for path in paths:
        images.append(read_image(path))
    images = 2 * np.array(images).astype(np.float32) / 255 - 1 # [-1,1]
    return images

# use multiple threads for feeding train samples
class enqueue_thread(threading.Thread):
    def __init__(self, name = None):
        super(enqueue_thread, self).__init__()
        self.name = name

    def run(self):
        global lock, data, q
        while not data.epoch_is_over:
            lock.acquire()
            image_batch, sketch_batch, bs = data.get_train_batch()
            # print(data.batch_index)
            lock.release()
            label_batch = image_batch[:, 2].astype(int)
            image_batch = image_batch[:, 1]
            sketch_batch = sketch_batch[:, 1]
            image_batch = read_image_batch(image_batch)
            sketch_batch = read_image_batch(sketch_batch)
            q.put((image_batch, sketch_batch, label_batch, bs))
        else:
            print('threads',self.name,'over')

def start_threads(nums_threads):
    for i in range(nums_threads):
        eq_thread = enqueue_thread(name=str(i))
        eq_thread.start()
        eq_threads.append(eq_thread)

def stop_all_threads():
    for t in eq_threads:
        t.exitFlag = 1
    for t in eq_threads:
        t.join()

def train():

    inputs_image = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    inputs_sketch = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    label_indices = tf.placeholder(tf.int32,shape=(None,))
    oh_labels = tf.one_hot(label_indices, _CLASSES)
    model = net.CDRL(_BATCH_SIZE,_CLASSES,_LOGITS_DIMS)
    losses, HfI_F, HrI_F = model.get_losses(inputs_sketch,inputs_image,oh_labels,is_training=True)
    learning_rate = tf.placeholder(tf.float32)
    train_op = model.get_train_op(losses, learning_rate)

    tf.summary.scalar('glt',losses['g_loss_total'])
    tf.summary.scalar('d_S',losses['d_S_loss'])
    tf.summary.scalar('d_I',losses['d_I_loss'])
    tf.summary.scalar('H_loss',losses['H_loss'])

    merge_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter('tf-board', tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=5)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.90
    # tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        sess.run(tf.global_variables_initializer())
        # use pretrained inception_v3
        H_ckpt_path = 'model/pretrained/inception_v3.ckpt'
        model.restore_H(sess, H_ckpt_path)

        lr_init = 0.0002
        lr = lr_init
        decay = [15,20]
        decay_weight = [0.1,0.01]
        step = 0
        for e in range(_EPOCHS):
            print('epoch %d' % e)
            if e != 0:
                data.renew(re_create=True)
            if (e + 1) in decay:
                lr = lr_init * decay_weight[decay.index((e + 1))]
                print('learning_rate change to: %f' % lr)
            start_threads(2)
            for i in range(math.ceil(data.len_train / _BATCH_SIZE)):
                image_batch, sketch_batch, label_batch, bs = q.get()
                glt,dsl,dil,hl,HfI_fea,HrI_fea,train_summary,_ = sess.run(
                    [losses['g_loss_total'],losses['d_S_loss'],losses['d_I_loss'],losses['H_loss'],
                     HfI_F,HrI_F,merge_summary,train_op],
                    feed_dict={
                        inputs_image:image_batch,
                        inputs_sketch:sketch_batch,
                        label_indices:label_batch,
                        learning_rate:lr
                    }
                )
                if (i + 1) % 50 == 0:
                    mean_dis = ut.mean_cosine_distance(HfI_fea, HrI_fea)
                    print('epoch:%d-%d, g_loss:%.5f, d_S:%.5f, d_I:%.5f, H_loss:%.5f, mean_dis:%.5f' %
                          (e,i,glt,dsl,dil,hl,mean_dis))

                writer.add_summary(train_summary, step)
                step += 1
            stop_all_threads()
            if not q.empty():
                print('queue is not empty after one epoch!')
                while not q.empty():
                    _ = q.get()
            if (e+1)%25==0:
            	saver.save(sess, 'model/CDRL/model.ckpt', global_step=e)

def process_image_tf(image_path, mode):
    ima_raw = tf.read_file(image_path)
    if mode == 'png':
        img_data = tf.image.decode_png(ima_raw, channels=3)
    elif mode == 'jpg':
        img_data = tf.image.decode_jpeg(ima_raw, channels=3)
    else:
        img_data = None
    assert img_data is not None, 'error in processing image!'
    image = tf.image.convert_image_dtype(img_data, dtype=tf.float32)  # [0~1]
    image = tf.image.resize_images(image, [_INPUT_SIZE, _INPUT_SIZE])
    image = 2 * image - 1 # [-1,1]
    return image

def save_images(inputs,path,row=4, col=4):
    def _normalization(a):
        mi = np.min(a)
        ma = np.max(a)
        return (a - mi) / (ma - mi)
    images = []
    rows = []
    for i in range(inputs.shape[0]):
        im = _normalization(inputs[i])*255
        im = im.astype(np.uint8)
        images.append(im)
    for i in range(row):
        rows.append(np.concatenate(images[col*i:col*i+col],axis=1))
    imgs = np.concatenate(rows,axis=0)
    imgs = Image.fromarray(imgs)
    imgs.save(path)

def sample_images(nums = 16):
    tf.reset_default_graph()
    image_dataset = tf.data.Dataset.from_tensor_slices(database[range(nums), 1])
    np.savetxt('im/itos_i.csv', database[range(nums)], fmt='%s')
    image_dataset = image_dataset.map(lambda img_paths: process_image_tf(img_paths, 'jpg'))
    image_dataset = image_dataset.batch(nums)
    sketch_dataset = tf.data.Dataset.from_tensor_slices(query[range(nums), 1])
    np.savetxt('im/stoi_s.csv', query[range(nums)], fmt='%s')
    sketch_dataset = sketch_dataset.map(lambda img_paths: process_image_tf(img_paths, 'png'))
    sketch_dataset = sketch_dataset.batch(nums)
    iterator = tf.data.Iterator.from_structure(image_dataset.output_types, image_dataset.output_shapes)

    imgs = iterator.get_next()
    image_init = iterator.make_initializer(image_dataset)
    sketch_init = iterator.make_initializer(sketch_dataset)

    model = net.CDRL(nums, _CLASSES, _LOGITS_DIMS)
    itos = model.itos(imgs)
    stoi = model.stoi(imgs)

    saver = tf.train.Saver()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.90
    # tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        model_path = tf.train.latest_checkpoint('model/CDRL')
        saver.restore(sess, model_path)
        sess.run(image_init)
        sfromi = sess.run(itos)
        sess.run(sketch_init)
        ifroms = sess.run(stoi)
        save_images(sfromi,'im/sfromi.jpg')
        save_images(ifroms,'im/ifroms.jpg')


def save_logits():
    tf.reset_default_graph()
    image_dataset = tf.data.Dataset.from_tensor_slices(database[:,1])
    image_dataset = image_dataset.map(lambda img_paths:process_image_tf(img_paths,'jpg'))
    image_dataset = image_dataset.prefetch(4*32)
    image_dataset = image_dataset.batch(2*32)
    sketch_dataset = tf.data.Dataset.from_tensor_slices(query[:,1])
    sketch_dataset = sketch_dataset.map(lambda img_paths:process_image_tf(img_paths,'png'))
    sketch_dataset = sketch_dataset.prefetch(4*32)
    sketch_dataset = sketch_dataset.batch(2*32)
    iterator = tf.data.Iterator.from_structure(image_dataset.output_types, image_dataset.output_shapes)

    imgs = iterator.get_next()
    image_init = iterator.make_initializer(image_dataset)
    sketch_init = iterator.make_initializer(sketch_dataset)

    # inputs_image = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    # inputs_sketch = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])

    model = net.CDRL(_BATCH_SIZE,_CLASSES,_LOGITS_DIMS)
    image_logits = model.inference_image(imgs)
    sketch_logits = model.inference_sketch(imgs)

    saver = tf.train.Saver()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.90
    # tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        model_path = tf.train.latest_checkpoint('model/CDRL')
        saver.restore(sess, model_path)
        logits_list = []
        sess.run(image_init)
        try:
            while True:
                logits_batch = sess.run(image_logits)
                logits_list.extend(logits_batch)
        except tf.errors.OutOfRangeError:
            pass
        logits_list = np.array(logits_list).reshape([-1, _LOGITS_DIMS])
        np.savetxt('csv/database_index.csv',
                   np.hstack((database[:,0].reshape([-1,1]).astype(int),database[:,2].reshape([-1,1]).astype(int))),
                   fmt="%d,%d",delimiter=',')
        np.savetxt('csv/database.csv',logits_list,fmt="%.8f")
        logits_list = []
        sess.run(sketch_init)
        try:
            while True:
                logits_batch = sess.run(sketch_logits)
                logits_list.extend(logits_batch)
        except tf.errors.OutOfRangeError:
            pass
        logits_list = np.array(logits_list).reshape([-1, _LOGITS_DIMS])
        np.savetxt('csv/query_index.csv',
                   np.hstack((query[:,0].reshape([-1,1]).astype(int),query[:,2].reshape([-1,1]).astype(int))),
                   fmt="%d,%d",delimiter=',')
        np.savetxt('csv/query.csv', logits_list,fmt="%.8f")

if __name__ == '__main__':
    train()
    save_logits()
    # sample_images()