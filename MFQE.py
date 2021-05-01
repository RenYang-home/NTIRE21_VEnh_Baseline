import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import os
import motion
import imageio
import net
import tensorflow.contrib as tfcon

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
parser.add_argument("--raw_path", type=str, default='./003_raw') # path to raw frames
parser.add_argument("--com_path", type=str, default='./003') # path to compressed frames
parser.add_argument("--pqf_path", type=str, default='./003_qecnn') # path to save enhanced frames
parser.add_argument("--enh_path", type=str, default='./003_enh') # path to save enhanced frames
parser.add_argument("--frame_num", type=int, default=250)
parser.add_argument("--H", type=int, default=536)
parser.add_argument("--W", type=int, default=960)
args = parser.parse_args()

def model(input_tensor, left_tensor, right_tensor):

    with tf.variable_scope("flow_motion_1"):
        flow_tensor_1, _, _, _, _, _ = motion.optical_flow(left_tensor, input_tensor, 1, args.H, args.W)
        com_warp_1 = tfcon.image.dense_image_warp(left_tensor, flow_tensor_1)

    with tf.variable_scope("flow_motion_2"):
        flow_tensor_2, _, _, _, _, _ = motion.optical_flow(right_tensor, input_tensor, 1, args.H, args.W)
        com_warp_2 = tfcon.image.dense_image_warp(right_tensor, flow_tensor_2)

    output_tensor = net.network(com_warp_1, input_tensor, com_warp_2)

    return output_tensor

comp_in = tf.placeholder(tf.float32, [args.H, args.W, 3])
comp_tensor = tf.expand_dims(comp_in/255.0, axis=0)

pqf_l = tf.placeholder(tf.float32, [args.H, args.W, 3])
pqf_l_tensor = tf.expand_dims(pqf_l/255.0, axis=0)

pqf_r = tf.placeholder(tf.float32, [args.H, args.W, 3])
pqf_r_tensor = tf.expand_dims(pqf_r/255.0, axis=0)

enha_tensor = tf.clip_by_value(model(input_tensor=comp_tensor, left_tensor=pqf_l_tensor, right_tensor=pqf_r_tensor), 0, 1)
enha_tensor = tf.cast(tf.round(enha_tensor[0] * 255.0), tf.uint8)

var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

saver = tf.train.Saver(max_to_keep=None)
checkpoint ='./model/MFQE/model.ckpt'
saver.restore(sess, save_path=checkpoint)

os.makedirs(args.enh_path, exist_ok=True)

PSNR_org = np.zeros([args.frame_num])
PSNR_enh = np.zeros([args.frame_num])

gop = int(np.floor((args.frame_num - 1)/4))

g = 0
pqf_l_frame = imageio.imread(args.com_path + '/' + str(g * 4 + 1).zfill(3) + '.png')
pdf_enh_l = imageio.imread(args.pqf_path + '/' + str(g * 4 + 1).zfill(3) + '.png')
pdf_raw_l = imageio.imread(args.raw_path + '/' + str(g * 4 + 1).zfill(3) + '.png')

PSNR_org[g * 4] = cal_psnr(pqf_l_frame / 255.0, pdf_raw_l / 255.0)
PSNR_enh[g * 4] = cal_psnr(pdf_enh_l / 255.0, pdf_raw_l / 255.0)

print('Frame:', g * 4 + 1, 'Original PSNR:', PSNR_org[g * 4], 'Enhanced PSNR:', PSNR_enh[g * 4])

for g in range(gop):

    pqf_r_frame = imageio.imread(args.com_path + '/' + str((g + 1) * 4 + 1).zfill(3) + '.png')
    pdf_enh_r = imageio.imread(args.pqf_path + '/' + '/' + str((g + 1) * 4 + 1).zfill(3) + '.png')
    pdf_raw_r = imageio.imread(args.raw_path + '/' + str((g + 1) * 4 + 1).zfill(3) + '.png')

    for d in range(3):

        comp_frame = imageio.imread(args.com_path + '/' + str(g * 4 + 2 + d).zfill(3) + '.png')
        enha_frame = sess.run(enha_tensor, feed_dict={comp_in: comp_frame, pqf_l: pqf_l_frame, pqf_r: pqf_r_frame})
        raw_frame = imageio.imread(args.raw_path + '/' + str(g * 4 + 2 + d).zfill(3) + '.png')

        imageio.imwrite(args.enh_path + '/' + str(g * 4 + 2 + d).zfill(3) + '.png', enha_frame)

        PSNR_org[g * 4 + 1 + d] = cal_psnr(comp_frame/255.0, raw_frame/255.0)
        PSNR_enh[g * 4 + 1 + d] = cal_psnr(enha_frame/255.0, raw_frame/255.0)

        print('Frame:', g * 4 + 2 + d, 'Original PSNR:', PSNR_org[g * 4 + 1 + d], 'Enhanced PSNR:', PSNR_enh[g * 4 + 1 + d])

    PSNR_org[(g + 1) * 4] = cal_psnr(pqf_r_frame / 255.0, pdf_raw_r / 255.0)
    PSNR_enh[(g + 1) * 4] = cal_psnr(pdf_enh_r / 255.0, pdf_raw_r / 255.0)

    print('Frame:', (g + 1) * 4 + 1, 'Original PSNR:', PSNR_org[(g + 1) * 4], 'Enhanced PSNR:', PSNR_enh[(g + 1) * 4])

if args.frame_num > gop * 4 + 1:

    for f in range(gop * 4 + 1, args.frame_num):

        comp_frame = imageio.imread(args.com_path + '/' + str(f + 1).zfill(3) + '.png')
        raw_frame = imageio.imread(args.raw_path + '/' + str(f + 1).zfill(3) + '.png')
        enha_frame = imageio.imread(args.pqf_path + '/' + str(f + 1).zfill(3) + '.png')

        PSNR_org[f] = cal_psnr(comp_frame / 255.0, raw_frame / 255.0)
        PSNR_enh[f] = cal_psnr(enha_frame / 255.0, raw_frame / 255.0)

        print('Frame:', f + 1, 'Original PSNR:', PSNR_org[f], 'Enhanced PSNR:', PSNR_enh[f])




