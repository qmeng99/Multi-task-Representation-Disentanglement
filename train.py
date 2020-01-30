import os
import tensorflow as tf
import numpy as np
import residual_def
import pdb
import random


height = 224
width = 288
batch_size = 50
lr = 1e-5
model_dir = './model'
logs_path = './model'
max_iter_step = 10010
rot_cls = 25 #degree
rot_func = 10 #degree
bg_num = 2
shape_num = 2
seed = 42


def read_decode(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={"img": tf.FixedLenFeature([],tf.string),
                  "bglbl": tf.FixedLenFeature([], tf.int64),
                  "shapelbl": tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features["img"], tf.float32)
    BG_label = tf.cast(features["bglbl"], tf.int64)
    Shape_label = tf.cast(features["shapelbl"], tf.int64)
    image = tf.reshape(image, [height, width, 1])
    images, BG_labels, Shape_labels = tf.train.batch([image, BG_label, Shape_label], batch_size=batch_size, capacity=1000, num_threads=8)
    # images and labels are tensor object
    return images, BG_labels, Shape_labels

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def load():
    filename = '/home/train/train_syth.tfrecords'
    filename_queue = tf.train.string_input_producer([filename])
    image, bg_lbl, shape_lbl = read_decode(filename_queue)

    return image, bg_lbl, shape_lbl

def build_gpu():

    with tf.device("/gpu:0"):

        image_orig, bg_lbl, shape_lbl = load()

        # data augmentation: adding noise
        # image_noise = gaussian_noise_layer(image_orig, 0.1)
        image_noise = image_orig
        # data augmentation: random flip
        image_squz = tf.transpose(tf.squeeze(image_noise, axis=3), [1,2,0])
        image_flip = tf.image.random_flip_left_right(image_squz)
        image_flip = tf.expand_dims(tf.transpose(image_flip, [2,0,1]), axis=3)

        # # data augmentation: random rotate
        # rot_rad_cls = tf.random_uniform(shape=(tf.shape(image_orig)[0],), minval=-np.pi / 180 * rot_cls, maxval=np.pi / 180 * rot_cls)
        # rot_rad_func = tf.random_uniform(shape=(tf.shape(image_orig)[0],), minval=-np.pi / 180 * rot_func, maxval=np.pi / 180 * rot_func)
        #
        # image_rot_cls = tf.contrib.image.rotate(image_flip, rot_rad_cls, interpolation='BILINEAR')
        # image_rot_func = tf.contrib.image.rotate(image_flip, rot_rad_func, interpolation='BILINEAR')

        image = tf.expand_dims(image_flip, axis=1)

        w_adv = tf.Variable(1e-2, dtype=tf.float32, trainable=False)
        l_r = tf.Variable(lr, dtype=tf.float32, trainable=False)
        w_cls = tf.Variable(1, dtype=tf.float32, trainable=False)

        opt_cls = tf.train.AdamOptimizer(learning_rate=l_r, beta1=0., beta2=0.9, epsilon=1e-5)
        opt_adv = tf.train.MomentumOptimizer(learning_rate=l_r, momentum=0.9)

        # ----------------------Encoder-------------------------

        with tf.variable_scope('Encoder_bg'):
            bg_fea = residual_def.residual_encoder(
                inputs=image,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64),
                strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ----------------------Encoder-------------------------

        with tf.variable_scope('Encoder_shape'):
            shape_fea = residual_def.residual_encoder(
                inputs=image,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64),
                strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ----------------------Anat_classification----------------------

        with tf.variable_scope('bg_cls'):
            bg_logits = residual_def.classify_dense_bn_relu(
                bg_fea,
                units=[256],
                is_train=True,
                num_class=bg_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ----------------------Anat_adversarial----------------------

        with tf.variable_scope('bg_adv'):
            shape_adv_logits = residual_def.classify_dense_bn_relu(
                shape_fea,
                units=[256],
                is_train=True,
                num_class=bg_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))


        # ----------------------Func_classification----------------------

        with tf.variable_scope('shape_cls'):
            shape_logits = residual_def.classify_dense_bn_relu(
                shape_fea,
                units=[256],
                is_train=True,
                num_class=shape_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ----------------------Func_adversarial----------------------
        with tf.variable_scope('shape_adv'):
            bg_adv_logits = residual_def.classify_dense_bn_relu(
                bg_fea,
                units=[256],
                is_train=True,
                num_class=shape_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))


        # ----------------------Loss--------------------------
        labels_onehot_bg = tf.one_hot(bg_lbl, depth=bg_num)
        bg_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=bg_logits, labels=labels_onehot_bg))
        reg_bg = tf.losses.get_regularization_loss('bg_cls')
        labels_onehot_shape = tf.one_hot(shape_lbl, depth=shape_num)
        shape_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=shape_logits, labels=labels_onehot_shape))
        reg_shape = tf.losses.get_regularization_loss('shape_cls')

        bg_adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=shape_adv_logits, labels=labels_onehot_bg))
        reg_bg_adv = tf.losses.get_regularization_loss('bg_adv')
        shape_adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=bg_adv_logits, labels=labels_onehot_shape))
        reg_shape_adv = tf.losses.get_regularization_loss('shape_adv')


        bg_Closs = bg_cls_loss + reg_bg - w_adv * (bg_adv_loss + reg_bg_adv)
        shape_Closs = shape_cls_loss + reg_shape - w_adv * (shape_adv_loss + reg_shape_adv)


        loss = w_cls * bg_Closs + shape_Closs

        loss_adv = w_cls * (bg_adv_loss + reg_bg_adv) + shape_adv_loss + reg_shape_adv

        # ------------------optimization----------------------------
        encbg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder_bg')
        encshape_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder_shape')
        bg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'bg_cls')
        bg_adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'bg_adv')
        shape_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'shape_cls')
        shape_adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'shape_adv')


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt_cls.minimize(loss, var_list=[encbg_vars, encshape_vars, bg_vars, shape_vars])
            train_adv_op = opt_adv.minimize(loss_adv, var_list=[bg_adv_vars, shape_adv_vars])


    return train_op, train_adv_op, \
           bg_cls_loss, bg_adv_loss, bg_Closs, \
           shape_cls_loss, shape_adv_loss, shape_Closs, loss, loss_adv, \
           image, \
           w_adv, w_cls

def main():
    train_op, train_adv_op, \
    bg_cls_loss, bg_adv_loss, bg_Closs, \
    shape_cls_loss, shape_adv_loss, shape_Closs, loss, loss_adv, \
    image, \
    w_adv, w_cls = build_gpu()

    # ----------------validation---------------------------------
    image_val = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
    lblbg = tf.placeholder(dtype=tf.int64, shape=[None])
    lblshape = tf.placeholder(dtype=tf.int64, shape=[None])

    # ----------------------Encoder-------------------------
    image_new = tf.expand_dims(image_val, 1)
    with tf.variable_scope('Encoder_bg', reuse=True):
        bg_fea_val = residual_def.residual_encoder(
            inputs=image_new,
            num_res_units=1,
            mode=tf.estimator.ModeKeys.EVAL,
            filters=(8, 16, 32, 64),
            strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    with tf.variable_scope('Encoder_shape', reuse=True):
        shape_fea_val = residual_def.residual_encoder(
            inputs=image_new,
            num_res_units=1,
            mode=tf.estimator.ModeKeys.EVAL,
            filters=(8, 16, 32, 64),
            strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Anat_classification----------------------

    with tf.variable_scope('bg_cls', reuse=True):
        bg_logits_val = residual_def.classify_dense_bn_relu(
            bg_fea_val,
            units=[256],
            is_train=False,
            num_class=bg_num,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Anat_adversarial----------------------

    with tf.variable_scope('bg_adv', reuse=True):
        shape_adv_logits_val = residual_def.classify_dense_bn_relu(
            shape_fea_val,
            units=[256],
            is_train=False,
            num_class=bg_num,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Func_classification----------------------

    with tf.variable_scope('shape_cls', reuse=True):
        shape_logits_val = residual_def.classify_dense_bn_relu(
            shape_fea_val,
            units=[256],
            is_train=False,
            num_class=shape_num,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Func_adversarial----------------------
    with tf.variable_scope('shape_adv', reuse=True):
        bg_adv_logits_val = residual_def.classify_dense_bn_relu(
            bg_fea_val,
            units=[256],
            is_train=False,
            num_class=shape_num,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Loss--------------------------
    onehot_bg = tf.one_hot(lblbg, depth=bg_num)
    bg_cls_loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=bg_logits_val, labels=onehot_bg))
    onehot_shape = tf.one_hot(lblshape, depth=shape_num)
    shape_cls_loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=shape_logits_val, labels=onehot_shape))

    bg_adv_loss_val = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=shape_adv_logits_val, labels=onehot_bg))
    shape_adv_loss_val = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=bg_adv_logits_val, labels=onehot_shape))

    bg_Closs_val = bg_cls_loss_val - w_adv * bg_adv_loss_val
    shape_Closs_val = shape_cls_loss_val - w_adv * shape_adv_loss_val

    loss_val = w_cls * bg_Closs_val + shape_Closs_val

    loss_adv_val = w_cls * bg_adv_loss_val + shape_adv_loss_val

    val_bg_label = tf.argmax(tf.nn.softmax(bg_logits_val), axis=1)
    val_shape_label = tf.argmax(tf.nn.softmax(shape_logits_val), axis=1)

    acc_bgtfcom = tf.divide(tf.reduce_sum(tf.cast(tf.equal(val_bg_label, lblbg), tf.float32)), tf.constant(300.0))
    acc_shapetfcom = tf.divide(tf.reduce_sum(tf.cast(tf.equal(val_shape_label, lblshape), tf.float32)), tf.constant(300.0))

    val_bg_label_adv = tf.argmax(tf.nn.softmax(shape_adv_logits_val), axis=1)
    val_shape_label_adv = tf.argmax(tf.nn.softmax(bg_adv_logits_val), axis=1)

    acc_bgtfcom_adv = tf.divide(tf.reduce_sum(tf.cast(tf.equal(val_bg_label_adv, lblbg), tf.float32)), tf.constant(300.0))
    acc_shapetfcom_adv = tf.divide(tf.reduce_sum(tf.cast(tf.equal(val_shape_label_adv, lblshape), tf.float32)),
                               tf.constant(300.0))

    # -----------------------------------------------------------

    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    # ###################################
    # # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # ###################################

    tf.set_random_seed(seed)
    np.random.seed(seed)

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a summary to monitor cost tensor
        tf.summary.scalar("bg_cls_loss", bg_cls_loss)
        tf.summary.scalar("bg_advFunc", bg_adv_loss)
        tf.summary.scalar("bg_Closs", bg_Closs)
        tf.summary.scalar("shape_cls_loss", shape_cls_loss)
        tf.summary.scalar("shape_advAnat", shape_adv_loss)
        tf.summary.scalar("shape_Closs", shape_Closs)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_adv", loss_adv)

        tf.summary.scalar("bg_cls_loss_val", bg_cls_loss_val)
        tf.summary.scalar("bg_advFunc_val", bg_adv_loss_val)
        tf.summary.scalar("bg_Closs_val", bg_Closs_val)
        tf.summary.scalar("shape_cls_loss_val", shape_cls_loss_val)
        tf.summary.scalar("shape_advAnat_val", shape_adv_loss_val)
        tf.summary.scalar("shape_Closs_val", shape_Closs_val)
        tf.summary.scalar("loss_val", loss_val)
        tf.summary.scalar("loss_adv_val", loss_adv_val)

        tf.summary.image('image', image[3,:,:,:,:], tf.float32)

        tf.summary.image('image_val', image_new[3, :, :, :, :], tf.float32)

        tf.summary.scalar("val_acc_bg", acc_bgtfcom)
        tf.summary.scalar("val_acc_shape", acc_shapetfcom)

        tf.summary.scalar("val_acc_bg_adv", acc_bgtfcom_adv)
        tf.summary.scalar("val_acc_shape_adv", acc_shapetfcom_adv)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        sess.run(init_op)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        valpath = '/home/validation/val_syth.npz'
        valimg = np.load(valpath)['img']
        val_lbl_bg = np.load(valpath)['bglbl']
        val_lbl_shape = np.load(valpath)['shapelbl']

        for i in range(max_iter_step):

            t_data = np.reshape(valimg, (valimg.shape[0], height, width, 1))
            t_bglbl = np.reshape(val_lbl_bg, (valimg.shape[0]))
            t_shapelbl = np.reshape(val_lbl_shape, (valimg.shape[0]))
            feed_dict = {image_val: t_data, lblbg: t_bglbl, lblshape: t_shapelbl}


            _, summary = sess.run([train_op, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)
            for k in range(3):
                _, summary = sess.run([train_adv_op, merged_summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, i)

            bg_Cls_Loss, bgLoss, shape_Cls_Loss, shapeLoss, Loss, \
            bg_Cls_Loss_val, bgLoss_val, shape_Cls_Loss_val, shapeLoss_val, Loss_val, acc_bgtf, acc_shapetf, acc_bgtf_adv, acc_shapetf_adv= sess.run(
                [bg_cls_loss, bg_Closs, shape_cls_loss, shape_Closs,
                 loss, bg_cls_loss_val, bg_Closs_val, shape_cls_loss_val, shape_Closs_val, loss_val, acc_bgtfcom,
                 acc_shapetfcom, acc_bgtfcom_adv, acc_shapetfcom_adv], feed_dict=feed_dict)

            tf_acc_bg = acc_bgtf
            tf_acc_shape = acc_shapetf

            tf_acc_bg_adv = acc_bgtf_adv
            tf_acc_shape_adv = acc_shapetf_adv


            if i % 100 == 0:
                print("i = %d" % i)
                print ("BG Cls Loss = {}".format(bg_Cls_Loss))
                print ("BG Cls_val = {}".format(bg_Cls_Loss_val))
                print ("BG Loss = {}".format(bgLoss))
                print ("BG val = {}".format(bgLoss_val))
                print ("Shape Cls Loss = {}".format(shape_Cls_Loss))
                print ("Shape Cls_val = {}".format(shape_Cls_Loss_val))
                print ("Shape Loss = {}".format(shapeLoss))
                print ("Shape val = {}".format(shapeLoss_val))
                print ("Loss all = {}".format(Loss))
                print ("Loss Val all = {}".format(Loss_val))

                print ('tf_acc_bg = {}'.format(tf_acc_bg))
                print ('tf_acc_shape = {}'.format(tf_acc_shape))

                print ('tf_acc_bg_adv = {}'.format(tf_acc_bg_adv))
                print ('tf_acc_shape_adv = {}'.format(tf_acc_shape_adv))

                with open('val_acc_bg_train5.txt', 'a') as lossFile1:
                    lossFile1.write('%.4f\n' % tf_acc_bg)
                with open('val_acc_shape_train5.txt', 'a') as lossFile2:
                    lossFile2.write('%.4f\n' % tf_acc_shape)

            if i % 500 == 0:
                saver.save(sess, os.path.join(model_dir, "model.val"), global_step=i)
            lossFile1.close()
            lossFile2.close()

        coord.request_stop()
        coord.join(threads)

main()
















