import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pdb
import tensorflow as tf
from scipy.misc import imsave

height = 224
width = 288

r_min = 10
r_max = 80
h_min = 20
h_max = 210
w_min = 20
w_max = 260
d_min = 10
d_max = 40

random_len = 1500



def gen_shape(flag_back, flag_shape, image_num):

    image = np.zeros(shape=(image_num, height, width))
    r_random = np.random.randint(r_min, r_max, size=random_len)
    d_random = np.random.randint(d_min, d_max, size=random_len)
    p_h = np.random.randint(h_min, h_max, size=random_len)
    p_w = np.random.randint(w_min, w_max, size=random_len)

    count = 0
    for k in range(random_len):
        if count < image_num:
            minv = np.array([p_h[k], p_w[k], height - p_h[k], width - p_w[k]]).min()
            if minv > r_random[k]:
                if flag_back == 1:
                    # generate circle with wight background
                    image_tmp = np.ones(shape=(height, width))
                elif flag_back == 0:
                    # generate circle with black background
                    image_tmp = np.zeros(shape=(height, width))

                for i in range(height):
                    for j in range(width):
                        if flag_shape == 0:
                            # generate circle
                            dis = math.sqrt((i - p_h[k])**2 + (j - p_w[k])**2)
                            if dis <= r_random[k]:
                                image_tmp[i, j] = 0.5
                        elif flag_shape == 1:
                            # generate square
                            dis_h = math.fabs(i - p_h[k])
                            dis_w = math.fabs(j - p_w[k])
                            if dis_h <=d_random[k] and dis_w <=(d_random[k]*4):
                                image_tmp[i, j] = 0.5

                img_mean = image_tmp.mean()
                img_std = image_tmp.std()
                image_tmp_norm = np.array(np.divide(image_tmp - img_mean, img_std), dtype=np.float32)


                image[count,:,:] = image_tmp_norm
                count = count + 1


                # plt.figure()
                # plt.imshow(image_tmp, cmap='gray', vmax=1, vmin=0)
                # plt.show()
                #
                # pdb.set_trace()


    print (image.shape)

    return image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecord(data, bglbl, shapelbl, filename):
    num,height,width = data.shape
    writer = tf.python_io.TFRecordWriter(filename)
    print("Writing...")
    for index in range(num):
        data_raw = data[index].astype(np.float32).tostring()
        bglbl_raw = bglbl[index].astype(np.int64)
        shapelbl_raw = shapelbl[index].astype(np.int64)
        example = tf.train.Example(features=tf.train.Features(feature={
            "img": _bytes_feature(data_raw),
            "bglbl": _int64_feature(bglbl_raw),
            "shapelbl": _int64_feature(shapelbl_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    return

# generate train data
cw_num = 600
image_circle_wight = gen_shape(flag_back=1, flag_shape=0, image_num=cw_num)
lbl_bg_circle_wight = np.ones(shape=cw_num)
lbl_shape_circle_wight = np.zeros(shape=cw_num)

rw_num = 100
image_square_wight = gen_shape(flag_back=1, flag_shape=1, image_num=rw_num)
lbl_bg_square_wight = np.ones(shape=rw_num)
lbl_shape_square_wight = np.ones(shape=rw_num)

rb_num = 500
image_square_black = gen_shape(flag_back=0, flag_shape=1, image_num=rb_num)
lbl_bg_square_black = np.zeros(shape=rb_num)
lbl_shape_square_black = np.ones(shape=rb_num)

# generate test and val data
cw_num = 100
image_circle_wight_test = gen_shape(flag_back=1, flag_shape=0, image_num=cw_num)
lbl_bg_circle_wight_test = np.ones(shape=cw_num)
lbl_shape_circle_wight_test = np.zeros(shape=cw_num)

rw_num = 100
image_square_wight_test = gen_shape(flag_back=1, flag_shape=1, image_num=rw_num)
lbl_bg_square_wight_test = np.ones(shape=rw_num)
lbl_shape_square_wight_test = np.ones(shape=rw_num)

rb_num = 100
image_square_black_test = gen_shape(flag_back=0, flag_shape=1, image_num=rb_num)
lbl_bg_square_black_test = np.zeros(shape=rb_num)
lbl_shape_square_black_test = np.ones(shape=rb_num)

cw_num = 100
image_circle_wight_val = gen_shape(flag_back=1, flag_shape=0, image_num=cw_num)
lbl_bg_circle_wight_val = np.ones(shape=cw_num)
lbl_shape_circle_wight_val = np.zeros(shape=cw_num)

rw_num = 100
image_square_wight_val = gen_shape(flag_back=1, flag_shape=1, image_num=rw_num)
lbl_bg_square_wight_val = np.ones(shape=rw_num)
lbl_shape_square_wight_val = np.ones(shape=rw_num)

rb_num = 100
image_square_black_val = gen_shape(flag_back=0, flag_shape=1, image_num=rb_num)
lbl_bg_square_black_val = np.zeros(shape=rb_num)
lbl_shape_square_black_val = np.ones(shape=rb_num)



# generate test unseen
cb_num = 100
image_circle_black = gen_shape(flag_back=0, flag_shape=0, image_num=cb_num)
lbl_bg_circle_black = np.zeros(shape=cb_num)
lbl_shape_circle_black = np.zeros(shape=cb_num)


#split data
train_data = np.concatenate([image_circle_wight, image_square_wight, image_square_black])
train_lbl_bg = np.concatenate([lbl_bg_circle_wight, lbl_bg_square_wight, lbl_bg_square_black])
train_lbl_shape = np.concatenate([lbl_shape_circle_wight, lbl_shape_square_wight, lbl_shape_square_black])

val_data = np.concatenate([image_circle_wight_val, image_square_wight_val, image_square_black_val])
val_lbl_bg = np.concatenate([lbl_bg_circle_wight_val, lbl_bg_square_wight_val, lbl_bg_square_black_val])
val_lbl_shape = np.concatenate([lbl_shape_circle_wight_val, lbl_shape_square_wight_val, lbl_shape_square_black_val])

test_data = np.concatenate([image_circle_wight_test, image_square_wight_test, image_square_black_test])
test_lbl_bg = np.concatenate([lbl_bg_circle_wight_test, lbl_bg_square_wight_test, lbl_bg_square_black_test])
test_lbl_shape = np.concatenate([lbl_shape_circle_wight_test, lbl_shape_square_wight_test, lbl_shape_square_black_test])


pdb.set_trace()

# shuffle the data
shuffle_idx = np.random.permutation(train_data.shape[0])
shuffle_train = train_data[shuffle_idx, :, :]
shuffle_train_bglbl = train_lbl_bg[shuffle_idx]
shuffle_train_shapelbl = train_lbl_shape[shuffle_idx]

shuffle_idx = np.random.permutation(val_data.shape[0])
shuffle_val = val_data[shuffle_idx, :, :]
shuffle_val_bglbl = val_lbl_bg[shuffle_idx]
shuffle_val_shapelbl = val_lbl_shape[shuffle_idx]

shuffle_idx = np.random.permutation(test_data.shape[0])
shuffle_test = test_data[shuffle_idx, :, :]
shuffle_test_bglbl = test_lbl_bg[shuffle_idx]
shuffle_test_shapelbl = test_lbl_shape[shuffle_idx]
print ('finish shuffle...')

np.savez('/home/test/test_syth.npz', img=shuffle_test, bglbl=shuffle_test_bglbl, shapelbl=shuffle_test_shapelbl)
np.savez('/home/validation/val_syth.npz', img=shuffle_val, bglbl=shuffle_val_bglbl, shapelbl=shuffle_val_shapelbl)
np.savez('/home/train/train_syth.npz', img=shuffle_train, bglbl=shuffle_train_bglbl, shapelbl=shuffle_train_shapelbl)

np.savez('/home/test/test_syth_unseen.npz', img=image_circle_black, bglbl=lbl_bg_circle_black, shapelbl=lbl_shape_circle_black)

print ('finish separat data')

#
# data = np.load('/data2/shadowData/datafortrail/train/train_syth.npz')
# shuffle_train = data['img']
# shuffle_train_bglbl = data['bglbl']
# shuffle_train_shapelbl = data['shapelbl']
generate_tfrecord(shuffle_train, shuffle_train_bglbl, shuffle_train_shapelbl, '/home/train/train_syth.tfrecords')


print ('finish generating')









