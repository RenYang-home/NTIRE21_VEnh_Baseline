import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import imageio
import os
import tensorflow.keras as tfkeras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, default='QECNN', choices=['ARCNN', 'DnCNN', 'QECNN'])
parser.add_argument("--raw_path", type=str, default='./003_raw') # path to raw frames
parser.add_argument("--com_path", type=str, default='./003') # path to compressed frames
parser.add_argument("--enh_path", type=str, default='./003_enh') # path to save enhanced frames
parser.add_argument("--frame_num", type=int, default=250)
parser.add_argument("--H", type=int, default=536)
parser.add_argument("--W", type=int, default=960)
args = parser.parse_args()

def model(input_tensor):

    if args.model == 'ARCNN':

        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=64, kernel_size=[9, 9], padding="same",
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=[1, 1], padding="same",
                                 activation=tf.nn.relu)
        output_tensor = tf.layers.conv2d(inputs=conv3, filters=3, kernel_size=[5, 5], padding="same",
                                         activation=None)
    elif args.model == 'DnCNN':

        conv = tf.layers.conv2d(inputs=input_tensor, filters=64, kernel_size=[3, 3], padding="same",
                                activation=tf.nn.relu, name='head')

        for i in range(18):
            with tf.variable_scope('layer_' + str(i + 1)):
                conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=[3, 3], padding="same",
                                        activation=None)
                conv = tf.layers.batch_normalization(conv, training=False)
                conv = tf.nn.relu(conv)

        conv = tf.layers.conv2d(inputs=conv, filters=3, kernel_size=[3, 3], padding="same",
                                activation=None, name='tail')

        output_tensor = input_tensor - conv

    else:

        conv_1 = tfkeras.layers.Conv2D(filters=128, kernel_size=[9, 9], padding="same", name='conv_1')(comp_tensor)
        conv_1 = tfkeras.layers.PReLU(name='prelu_1', shared_axes=[1, 2])(conv_1)

        conv_2 = tfkeras.layers.Conv2D(filters=64, kernel_size=[7, 7], padding="same", name='conv_2')(conv_1)
        conv_2 = tfkeras.layers.PReLU(name='prelu_2', shared_axes=[1, 2])(conv_2)

        conv_3 = tfkeras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", name='conv_3')(conv_2)
        conv_3 = tfkeras.layers.PReLU(name='prelu_3', shared_axes=[1, 2])(conv_3)

        conv_4 = tfkeras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding="same", name='conv_4')(conv_3)
        conv_4 = tfkeras.layers.PReLU(name='prelu_4', shared_axes=[1, 2])(conv_4)

        conv_11 = tfkeras.layers.Conv2D(filters=128, kernel_size=[9, 9], padding="same", name='conv_6')(comp_tensor)
        conv_11 = tfkeras.layers.PReLU(name='prelu_6', shared_axes=[1, 2])(conv_11)

        feat_11 = tf.concat([conv_1, conv_11], axis=-1)
        conv_22 = tfkeras.layers.Conv2D(filters=64, kernel_size=[7, 7], padding="same", name='conv_7')(feat_11)
        conv_22 = tfkeras.layers.PReLU(name='prelu_7', shared_axes=[1, 2])(conv_22)

        feat_22 = tf.concat([conv_2, conv_22], axis=-1)
        conv_33 = tfkeras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", name='conv_8')(feat_22)
        conv_33 = tfkeras.layers.PReLU(name='prelu_8', shared_axes=[1, 2])(conv_33)

        feat_33 = tf.concat([conv_3, conv_33], axis=-1)
        conv_44 = tfkeras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding="same", name='conv_9')(feat_33)
        conv_44 = tfkeras.layers.PReLU(name='prelu_9', shared_axes=[1, 2])(conv_44)

        feat_44 = tf.concat([conv_4, conv_44], axis=-1)
        conv_10 = tfkeras.layers.Conv2D(filters=3, kernel_size=[5, 5], padding="same", name='conv_out')(feat_44)

        output_tensor = comp_tensor + conv_10

    return output_tensor

comp_in = tf.placeholder(tf.float32, [args.H, args.W, 3])
comp_tensor = tf.expand_dims(comp_in/255.0, axis=0)

enha_tensor = tf.clip_by_value(model(input_tensor=comp_tensor), 0, 1)
enha_tensor = tf.cast(tf.round(enha_tensor[0] * 255.0), tf.uint8)

saver = tf.train.Saver(max_to_keep=None)
checkpoint ='./model/' + args.model + '/model.ckpt'
saver.restore(sess, save_path=checkpoint)

os.makedirs(args.enh_path, exist_ok=True)

PSNR_org = np.zeros([args.frame_num])
PSNR_enh = np.zeros([args.frame_num])

for f in range(args.frame_num):

    comp_frame = imageio.imread(args.com_path + '/' + str(f + 1).zfill(3) + '.png')

    enha_frame = sess.run(enha_tensor, feed_dict={comp_in: comp_frame})

    raw_frame = imageio.imread(args.raw_path + '/' + str(f + 1).zfill(3) + '.png')

    imageio.imwrite(args.enh_path + '/' + str(f + 1).zfill(3) + '.png', enha_frame)

    PSNR_org[f] = cal_psnr(comp_frame/255.0, raw_frame/255.0)
    PSNR_enh[f] = cal_psnr(enha_frame/255.0, raw_frame/255.0)

    print('Frame:', f + 1, 'Original PSNR:', PSNR_org[f], 'Enhanced PSNR:', PSNR_enh[f])

print('Average ', 'Original PSNR:', np.mean(PSNR_org), 'Enhanced PSNR:', np.mean(PSNR_enh))



