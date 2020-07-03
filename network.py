# CDRL model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


_WEIGHT_DECAY = 4e-5
_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-5

LambdaA = 10
LambdaB = 10
LambdaC = 10


class Generator:
    def __init__(self, name, in_channel = 3, out_channel = 3, res_blocks = 4, filters = 32):
        self.name = name
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.depths = [filters, filters*2, filters*4]
        self.reuse = False

    def __call__(self, inputs, is_training):

        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                inputs = tf.layers.conv2d(inputs, self.depths[0], [8,8], (2,2), padding='SAME')
                inputs = self._batch_norm_relu(inputs, is_training)
            with tf.variable_scope('conv2'):
                inputs = tf.layers.conv2d(inputs, self.depths[1], [3,3], (2,2), padding='SAME')
                inputs = self._batch_norm_relu(inputs, is_training)
            with tf.variable_scope('conv3'):
                inputs = tf.layers.conv2d(inputs, self.depths[2], [3,3], (2,2), padding='SAME')
                inputs = self._batch_norm_relu(inputs, is_training)
            with tf.variable_scope('_res_block'):
                inputs = tf.layers.conv2d(inputs, self.depths[2], [3,3], (1,1), padding='SAME')
                for i in range(self.res_blocks):
                    inputs = self._res_block(inputs, is_training)
                inputs = tf.nn.relu(inputs)
            with tf.variable_scope('deconv1'):
                inputs = tf.layers.conv2d_transpose(inputs, self.depths[1], [3,3], (2,2), padding='SAME')
                inputs = self._batch_norm_relu(inputs, is_training)
            with tf.variable_scope('deconv2'):
                inputs = tf.layers.conv2d_transpose(inputs, self.depths[0], [3,3], (2,2), padding='SAME')
                inputs = self._batch_norm_relu(inputs, is_training)
            with tf.variable_scope('deconv3'):
                inputs = tf.layers.conv2d_transpose(inputs, self.out_channel, [8,8], (2,2), padding='SAME')
            inputs = tf.nn.tanh(inputs)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        return inputs

    def _batch_norm_relu(self, inputs, is_training):
        inputs = tf.layers.batch_normalization(inputs, momentum=_BATCH_NORM_DECAY,
                                               epsilon=_BATCH_NORM_EPSILON, training=is_training)
        inputs = tf.nn.relu(inputs)
        return inputs

    def _res_block(self, inputs, is_training):
        x = self._batch_norm_relu(inputs, is_training)
        x = tf.layers.conv2d(x, self.depths[2], [3, 3], (1, 1), padding='SAME')
        x = self._batch_norm_relu(x, is_training)
        x = tf.layers.conv2d(x, self.depths[2], [3, 3], (1, 1), padding='SAME')
        return inputs + x

class Discriminator:
    def __init__(self, name, in_channel = 3, out_channel = 64, filters = 64):
        self.name = name
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depths = [filters, filters*2, filters*4, filters*8]
        self.reuse = False

    def __call__(self, inputs, is_training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                inputs = tf.layers.conv2d(inputs, self.depths[0], [4,4], (2,2), padding='SAME')
            with tf.variable_scope('conv2'):
                inputs = tf.layers.conv2d(inputs, self.depths[1], [4,4], (2,2), padding='SAME')
                inputs = self._batch_norm_leaky_relu(inputs, is_training)
            with tf.variable_scope('conv3'):
                inputs = tf.layers.conv2d(inputs, self.depths[2], [4,4], (2,2), padding='SAME')
                inputs = self._batch_norm_leaky_relu(inputs, is_training)
            with tf.variable_scope('conv4'):
                inputs = tf.layers.conv2d(inputs, self.depths[3], [4,4], (1,1), padding='SAME')
                inputs = self._batch_norm_leaky_relu(inputs, is_training)
            with tf.variable_scope('conv5'):
                inputs = tf.layers.conv2d(inputs, self.out_channel, [4,4], (1,1), padding='SAME')
                # shape=(3, 32, 32, 64)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        return inputs

    def _batch_norm_leaky_relu(self, inputs, is_training):
        inputs = tf.layers.batch_normalization(inputs, momentum=_BATCH_NORM_DECAY,
                                               epsilon=_BATCH_NORM_EPSILON, training=is_training)
        inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
        return inputs


class SLN:

    def __init__(self, name, dims, classes):
        self.name = name
        self.dims = dims
        self.classes = classes
        self.reuse = False

    def __call__(self, inputs, is_training):
        with tf.name_scope(self.name):
            with slim.arg_scope(nets.inception.inception_v3_arg_scope()):
                # inception_v3.default_image_size = 299， but 256 is also OK.
                features, _ = nets.inception.inception_v3(inputs, num_classes=self.dims, is_training=is_training, reuse=tf.AUTO_REUSE)
            with tf.variable_scope('InceptionV3/MyLogits', reuse=self.reuse):
                logits = tf.layers.dense(features, self.classes)
            self.reuse = True

        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV3')
        return features, logits

    def restore(self, sess, ckpt_path):
        # exclusions = ['InceptionV3/AuxLogits','InceptionV3/Logits']
        # variables_to_restore = []
        # for var in slim.get_variables_to_restore(include=['InceptionV3']):
        #     excluded = False
        #     for exclusion in exclusions:
        #         if var.op.name.startswith(exclusion):
        #             excluded = True
        #     if not excluded:
        #         variables_to_restore.append(var)
        # saver_restore = tf.train.Saver(var_list=variables_to_restore)
        # saver_restore.restore(sess, ckpt_path)

        variables_to_restore = slim.get_variables_to_restore(exclude=['G_I','D_I','G_S','D_S',
                                                                      'InceptionV3/AuxLogits',
                                                                      'InceptionV3/Logits',
                                                                      'Adam',
                                                                      'Adam_1'])
        init_fn = slim.assign_from_checkpoint_fn(ckpt_path, variables_to_restore, ignore_missing_vars=True)
        init_fn(sess)
        print('Inception_v3 Loaded.')

class CDRL:

    def __init__(self, batch_size, classes, logits_dims = 256):
        self.batch_size = batch_size
        self.logits_dims = logits_dims
        self.classes = classes
        self.G_S = Generator('G_S')
        self.G_I = Generator('G_I')
        self.D_S = Discriminator('D_S')
        self.D_I = Discriminator('D_I')
        self.H = SLN('H', logits_dims, classes)
        # noises
        self.z_S = tf.random_uniform([self.batch_size, 256, 256, 3], minval=-1.0, maxval=1.0)
        self.z_I = tf.random_uniform([self.batch_size, 256, 256, 3], minval=-1.0, maxval=1.0)

    def get_losses(self, real_S, real_I, onehot_labels, is_training):
        # generate real_S to fake_I
        fake_I = self.G_I(real_S, is_training)
        d_fake_I_logits = self.D_I(fake_I, is_training)
        # MSEloss 这里可以考虑换成DCGAN的loss之类的试试
        # g_I_loss = tf.reduce_mean(tf.square(d_fake_I_logits - tf.ones(d_fake_I_logits.get_shape().as_list(), dtype=tf.int64)))
        g_I_loss = tf.reduce_mean(tf.square(d_fake_I_logits - 1))

        # reconstruct fake_I to rec_S
        rec_S = self.G_S(fake_I, is_training)
        S_cycle_loss = tf.reduce_mean(tf.abs((real_S - rec_S))) * LambdaA # L1 Loss
        # S_cycle_loss = tf.nn.l2_loss((real_S - rec_S)) # L2 Loss

        # generator real_I to fake_S
        fake_S = self.G_S(real_I, is_training)
        d_fake_S_logits = self.D_S(fake_S, is_training)
        # g_S_loss = tf.reduce_mean(tf.square(d_fake_S_logits - tf.ones(d_fake_S_logits.get_shape().as_list(), dtype=tf.int64)))
        g_S_loss = tf.reduce_mean(tf.square(d_fake_S_logits - 1))

        # reconstruct fake_S to fake_I
        rec_I = self.G_I(fake_S, is_training)
        I_cycle_loss = tf.reduce_mean(tf.abs((real_I - rec_I))) * LambdaB

        g_loss_total = g_I_loss + S_cycle_loss + g_S_loss + I_cycle_loss

        d_real_S = self.D_S(real_S, is_training)
        # d_real_S_loss = tf.reduce_mean(tf.square(d_real_S - tf.ones(d_real_S.get_shape().as_list(), dtype=tf.int64)))
        d_real_S_loss = tf.reduce_mean(tf.square(d_real_S - 1))
        d_fake_S = self.D_S(self.z_S, is_training) # 这里应该可以应用cGAN进去来提升效果
        # d_fake_S_loss = tf.reduce_mean(tf.square(d_fake_S - tf.zeros(d_fake_S.get_shape().as_list(), dtype=tf.int64)))
        d_fake_S_loss = tf.reduce_mean(tf.square(d_fake_S - 0))
        d_S_loss = (d_real_S_loss + d_fake_S_loss) * 0.5

        d_real_I = self.D_I(real_I, is_training)
        # d_real_I_loss = tf.reduce_mean(tf.square(d_real_I - tf.ones(d_real_I.get_shape().as_list(), dtype=tf.int64)))
        d_real_I_loss = tf.reduce_mean(tf.square(d_real_I - 1))
        d_fake_I = self.D_I(self.z_I, is_training)
        # d_fake_I_loss = tf.reduce_mean(tf.square(d_fake_I - tf.zeros(d_fake_I.get_shape().as_list(), dtype=tf.int64)))
        d_fake_I_loss = tf.reduce_mean(tf.square(d_fake_I - 0))
        d_I_loss = (d_real_I_loss + d_fake_I_loss) * 0.5
        # d_loss_total = d_S_loss + d_I_loss

        # Hash net
        H_fake_I_F, H_fake_I_logits= self.H(fake_I, is_training)
        H_real_I_F, H_real_I_logits = self.H(real_I, is_training)
        cross_entropy_fake = tf.losses.softmax_cross_entropy(logits=H_fake_I_logits,onehot_labels=onehot_labels)
        cross_entropy_real = tf.losses.softmax_cross_entropy(logits=H_real_I_logits,onehot_labels=onehot_labels)
        # compare_l2 = 0.001 * tf.nn.l2_loss((H_fake_I_F - H_real_I_F))
        compare_l2 = 0.5 * tf.nn.l2_loss((H_fake_I_F - H_real_I_F)) / self.logits_dims
        

        tf.summary.scalar('compare_l2',compare_l2)
        tf.summary.scalar('cross_entropy_fake',cross_entropy_fake)
        tf.summary.scalar('cross_entropy_real',cross_entropy_real)

        H_loss = compare_l2 + cross_entropy_fake + cross_entropy_real + _WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in self.H.variables])

        return {
            'g_loss_total':g_loss_total,
            # 'd_loss_total':d_loss_total,
            'd_S_loss':d_S_loss,
            'd_I_loss':d_I_loss,
            'H_loss':H_loss
        }, H_fake_I_F, H_real_I_F

    def inference_pair(self,real_S, real_I, is_training = False):
        fakeI = self.G_I(real_S, is_training)
        fakeI_logits,_ = self.H(fakeI, is_training)
        realI_logits,_ = self.H(real_I, is_training)
        return fakeI_logits, realI_logits

    def inference_sketch(self,real_S, is_training = False):
        fakeI = self.G_I(real_S, is_training)
        fakeI_logits, _ = self.H(fakeI, is_training)
        return fakeI_logits

    def inference_image(self,real_I, is_training = False):
        realI_logits, _ = self.H(real_I, is_training)
        return realI_logits

    def get_train_op(self, losses, learning_rate=0.0002, beta1=0.5, beta2=0.999):

        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        d_S_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        d_I_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        H_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_opt_op = g_opt.minimize(losses['g_loss_total'], var_list=self.G_I.variables + self.G_S.variables)
            d_S_opt_op = d_S_opt.minimize(losses['d_S_loss'], var_list=self.D_S.variables)
            d_I_opt_op = d_I_opt.minimize(losses['d_I_loss'], var_list=self.D_I.variables)
            H_opt_op = H_opt.minimize(losses['H_loss'], var_list=self.H.variables)

        with tf.control_dependencies([g_opt_op, d_S_opt_op, d_I_opt_op, H_opt_op]):
            return tf.no_op(name='train')

    def restore_H(self, sess, ckpt_path):
        self.H.restore(sess, ckpt_path)

    def itos(self, image, is_training=False):
        return self.G_S(image,is_training)

    def stoi(self, sketch, is_training=False):
        return self.G_I(sketch, is_training)






























