import os
import tensorflow as tf
import numpy as np
import residual_def
import pdb
import random



height = 224
width = 288
ckpt_dir = './model'
batch_size = 50
bgnum = 2
shapenum = 2


def load():
    testpath = '/home/test/test_syth_unseen.npz'
    testimg = np.load(testpath)['img']
    test_lbl_bg = np.load(testpath)['bglbl']
    test_lbl_shape = np.load(testpath)['shapelbl']

    return testimg, test_lbl_bg, test_lbl_shape


def main():
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/gpu:0"):
            image_val = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
            lblbg = tf.placeholder(dtype=tf.int64, shape=[None])
            lblshape = tf.placeholder(dtype=tf.int64, shape=[None])

            # ----------------------Encoder-------------------------
            image_new = tf.expand_dims(image_val, 1)
            with tf.variable_scope('Encoder_bg'):
                bg_fea_val = residual_def.residual_encoder(
                    inputs=image_new,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32, 64),
                    strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            with tf.variable_scope('Encoder_shape'):
                shape_fea_val = residual_def.residual_encoder(
                    inputs=image_new,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32, 64),
                    strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            # ----------------------Anat_classification----------------------

            with tf.variable_scope('bg_cls'):
                bg_logits_val = residual_def.classify_dense_bn_relu(
                    bg_fea_val,
                    units=[256],
                    is_train=False,
                    num_class=bgnum,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            # ----------------------Anat_adversarial----------------------

            with tf.variable_scope('bg_adv'):
                shape_adv_logits_val = residual_def.classify_dense_bn_relu(
                    shape_fea_val,
                    units=[256],
                    is_train=False,
                    num_class=bgnum,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            # ----------------------Func_classification----------------------

            with tf.variable_scope('shape_cls'):
                shape_logits_val = residual_def.classify_dense_bn_relu(
                    shape_fea_val,
                    units=[256],
                    is_train=False,
                    num_class=shapenum,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            # ----------------------Func_adversarial----------------------
            with tf.variable_scope('shape_adv'):
                bg_adv_logits_val = residual_def.classify_dense_bn_relu(
                    bg_fea_val,
                    units=[256],
                    is_train=False,
                    num_class=shapenum,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

                # ----------------------Loss--------------------------

            test_bg_softmax = tf.nn.softmax(bg_logits_val)
            test_bg_label = tf.argmax(test_bg_softmax, axis=1)
            loss_bg_softce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=bg_logits_val, labels=tf.one_hot(lblbg, depth=bgnum)))

            test_shape_softmax = tf.nn.softmax(shape_logits_val)
            test_shape_label = tf.argmax(test_shape_softmax, axis=1)
            loss_shape_softce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=shape_logits_val, labels=tf.one_hot(lblshape, depth=shapenum)))


            test_bg_adv_softmax = tf.nn.softmax(shape_adv_logits_val)
            test_bg_adv_label = tf.argmax(test_bg_adv_softmax, axis=1)
            loss_bg_adv_softce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=shape_adv_logits_val, labels=tf.one_hot(lblbg, depth=bgnum)))

            test_shape_adv_softmax = tf.nn.softmax(bg_adv_logits_val)
            test_shape_adv_label = tf.argmax(test_shape_adv_softmax, axis=1)
            loss_shape_adv_softce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=bg_adv_logits_val, labels=tf.one_hot(lblshape, depth=shapenum)))


        # ---------------------------------------------------
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

            data, lbl_bg, lbl_shape = load()
            data_new = np.reshape(data, (data.shape[0], height, width, 1))

            right_num_bg = 0
            right_num_shape = 0
            r_n_adv_bg = 0
            r_n_adv_shape = 0

            count_1 = data_new.shape[0] // batch_size
            count_2 = data_new.shape[0] % batch_size

            pred_bg_lbl = np.zeros(shape=lbl_bg.shape)
            prob_bg_lbl = np.zeros(shape=(lbl_bg.shape[0], bgnum))

            pred_adv_bg_lbl = np.zeros(shape=lbl_bg.shape)
            prob_adv_bg_lbl = np.zeros(shape=(lbl_bg.shape[0], bgnum))

            pred_shape_lbl = np.zeros(shape=lbl_shape.shape)
            prob_shape_lbl = np.zeros(shape=(lbl_shape.shape[0], shapenum))

            pred_adv_shape_lbl = np.zeros(shape=lbl_shape.shape)
            prob_adv_shape_lbl = np.zeros(shape=(lbl_shape.shape[0], shapenum))


            for i in range(count_1):
                t_data = data_new[i * batch_size:(i + 1) * batch_size, :, :, :]
                t_bg_lbl = lbl_bg[i * batch_size:(i + 1) * batch_size]
                t_shape_lbl = lbl_shape[i * batch_size:(i + 1) * batch_size]
                feed_dict = {image_val: t_data, lblbg: t_bg_lbl, lblshape: t_shape_lbl}

                prob_bg, pred_bg, loss_soft_bg, prob_shape, pred_shape, loss_soft_shape = sess.run(
                    [test_bg_softmax, test_bg_label, loss_bg_softce,
                    test_shape_softmax, test_shape_label, loss_shape_softce],
                    feed_dict=feed_dict)

                prob_adv_bg, pred_adv_bg, loss_adv_bg, prob_adv_shape, pred_adv_shape, loss_adv_shape = sess.run(
                    [test_bg_adv_softmax, test_bg_adv_label, loss_bg_adv_softce,
                     test_shape_adv_softmax, test_shape_adv_label, loss_shape_adv_softce],
                    feed_dict=feed_dict)

                prob_bg_lbl[i * batch_size:(i + 1) * batch_size, :] = prob_bg
                prob_shape_lbl[i * batch_size:(i + 1) * batch_size, :] = prob_shape

                prob_adv_bg_lbl[i * batch_size:(i + 1) * batch_size, :] = prob_adv_bg
                prob_adv_shape_lbl[i * batch_size:(i + 1) * batch_size, :] = prob_adv_shape

                print ('lossce_bg: {}, lossce_shape: {}'.format(loss_soft_bg, loss_soft_shape))

                pred_bg_lbl[i * batch_size:(i + 1) * batch_size] = pred_bg
                pred_shape_lbl[i * batch_size:(i + 1) * batch_size] = pred_shape

                pred_adv_bg_lbl[i * batch_size:(i + 1) * batch_size] = pred_adv_bg
                pred_adv_shape_lbl[i * batch_size:(i + 1) * batch_size] = pred_adv_shape

                for ss in range(batch_size):
                    if ((t_bg_lbl[ss] - pred_bg[ss]) == 0):
                        right_num_bg = right_num_bg + 1
                    if ((t_shape_lbl[ss] - pred_shape[ss]) == 0):
                        right_num_shape = right_num_shape + 1

                    if ((t_bg_lbl[ss] - pred_adv_bg[ss]) == 0):
                        r_n_adv_bg = r_n_adv_bg + 1
                    if ((t_shape_lbl[ss] - pred_adv_shape[ss]) == 0):
                        r_n_adv_shape = r_n_adv_shape + 1

                    print ("Image {}: BG T_lbl= {}, BG P_lbl= {}, Shape T_lbl= {}, Shape P_lbl= {}".format((i * batch_size + ss),
                                                                                                             t_bg_lbl[ss],
                                                                                                             pred_bg[ss],
                                                                                                             t_shape_lbl[ss],
                                                                                                             pred_shape[ss]
                                                                                                             ))
            if count_2 != 0:

                t_data_2 = data_new[count_1 * batch_size: count_1 * batch_size + count_2, :, :, :]
                t_bg_lbl_2 = lbl_bg[count_1 * batch_size: count_1 * batch_size + count_2]
                t_shape_lbl_2 = lbl_shape[count_1 * batch_size: count_1 * batch_size + count_2]
                feed_dict = {image_val: t_data_2, lblbg: t_bg_lbl_2, lblshape: t_shape_lbl_2}

                prob_bg_2, pred_bg_2, loss_soft_bg_2, prob_shape_2, pred_shape_2, loss_soft_shape_2 = sess.run(
                    [test_bg_softmax, test_bg_label, loss_bg_softce,
                     test_shape_softmax, test_shape_label, loss_shape_softce],
                    feed_dict=feed_dict)

                prob_adv_bg_2, pred_adv_bg_2, loss_adv_bg_2, prob_adv_shape_2, pred_adv_shape_2, loss_adv_shape_2 = sess.run(
                    [test_bg_adv_softmax, test_bg_adv_label, loss_bg_adv_softce,
                     test_shape_adv_softmax, test_shape_adv_label, loss_shape_adv_softce],
                    feed_dict=feed_dict)


                print ('lossce_bg: {}, lossce_shape: {}'.format(loss_soft_bg_2, loss_soft_shape_2))

                prob_bg_lbl[count_1 * batch_size: count_1 * batch_size + count_2, :] = prob_bg_2
                prob_shape_lbl[count_1 * batch_size: count_1 * batch_size + count_2, :] = prob_shape_2

                prob_adv_bg_lbl[count_1 * batch_size: count_1 * batch_size + count_2, :] = prob_adv_bg_2
                prob_adv_shape_lbl[count_1 * batch_size: count_1 * batch_size + count_2, :] = prob_adv_shape_2

                pred_bg_lbl[count_1 * batch_size: count_1 * batch_size + count_2] = pred_bg_2
                pred_shape_lbl[count_1 * batch_size: count_1 * batch_size + count_2] = pred_shape_2

                pred_adv_bg_lbl[count_1 * batch_size: count_1 * batch_size + count_2] = pred_adv_bg_2
                pred_adv_shape_lbl[count_1 * batch_size: count_1 * batch_size + count_2] = pred_adv_shape_2


                for ss in range(count_2):
                    if ((t_bg_lbl_2[ss] - pred_bg_2[ss]) == 0):
                        right_num_bg = right_num_bg + 1
                    if ((t_shape_lbl_2[ss] - pred_shape_2[ss]) == 0):
                        right_num_shape = right_num_shape + 1

                    if ((t_bg_lbl_2[ss] - pred_adv_bg_2[ss]) == 0):
                        r_n_adv_bg = r_n_adv_bg + 1
                    if ((t_shape_lbl_2[ss] - pred_adv_shape_2[ss]) == 0):
                        r_n_adv_shape = r_n_adv_shape + 1

                    print (
                    "Image {}: BG T_lbl= {}, BG P_lbl= {}, Shape T_lbl= {}, Shape P_lbl= {}".format((count_1 * batch_size + ss),
                                                                                                      t_bg_lbl_2[ss],
                                                                                                      pred_bg_2[ss],
                                                                                                      t_shape_lbl_2[ss],
                                                                                                      pred_shape_2[ss]
                                                                                                      ))

            right_1_bg = 0
            right_2_bg = 0
            right_1_shape = 0
            right_2_shape = 0

            right_1_adv_bg = 0
            right_2_adv_bg = 0
            right_1_adv_shape = 0
            right_2_adv_shape = 0


            for ss in range(data_new.shape[0]):
                if (lbl_bg[ss] == 0) and (pred_bg_lbl[ss] == 0):
                    right_1_bg = right_1_bg + 1
                elif (lbl_bg[ss] == 1) and (pred_bg_lbl[ss] == 1):
                    right_2_bg = right_2_bg + 1

                if (lbl_shape[ss] == 0) and (pred_shape_lbl[ss] == 0):
                    right_1_shape = right_1_shape + 1
                elif (lbl_shape[ss] == 1) and (pred_shape_lbl[ss] == 1):
                    right_2_shape = right_2_shape + 1


                if (lbl_bg[ss] == 0) and (pred_adv_bg_lbl[ss] == 0):
                    right_1_adv_bg = right_1_adv_bg + 1
                elif (lbl_bg[ss] == 1) and (pred_adv_bg_lbl[ss] == 1):
                    right_2_adv_bg = right_2_adv_bg + 1

                if (lbl_shape[ss] == 0) and (pred_adv_shape_lbl[ss] == 0):
                    right_1_adv_shape = right_1_adv_shape + 1
                elif (lbl_shape[ss] == 1) and (pred_adv_shape_lbl[ss] == 1):
                    right_2_adv_shape = right_2_adv_shape + 1



            print ("Mean accuracy BG= {:.4f}".format((right_num_bg / data_new.shape[0])))

            print ("Mean accuracy Shape= {:.4f}".format((right_num_shape / data_new.shape[0])))


            # print ("Mean error BG= {:.4f}".format(1-(r_n_adv_bg / data_new.shape[0])))
            #
            #
            # print ("Mean error Shape= {:.4f}".format(1-(r_n_adv_shape / data_new.shape[0])))


            print (right_num_bg, right_num_shape, right_1_bg, right_2_bg, len(np.where(lbl_bg == 0)[0]),
                   len(np.where(lbl_bg == 1)[0]))
            print (right_1_shape, right_2_shape, len(np.where(lbl_shape == 0)[0]), len(np.where(lbl_shape == 1)[0]))

            print (r_n_adv_shape, right_1_adv_shape, right_2_adv_shape, len(np.where(lbl_shape == 0)[0]), len(np.where(lbl_shape == 1)[0]))




    return

main()
















