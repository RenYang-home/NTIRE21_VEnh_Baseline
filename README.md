# Baseline models for NTIRE 2021 Video Enhancement Challenge

The codes and pre-trained models of the baseline methods for the [NTIRE 2021 Video Enhancement Challenge](https://github.com/RenYang-home/NTIRE21_VEnh). The baseline methods are trained for Track 1, including MFQE, QE-CNN, DnCNN and ARCNN.

## Dependency

TensorFlow 1.15

TFLearn

Pre-trained models ([Download](https://data.vision.ee.ethz.ch/reyang/model/model.zip) and unzip to the root path)

## Preparation

First convert the mkv videos in the [NTIRE 2021 Video Enhancement Challenge](https://github.com/RenYang-home/NTIRE21_VEnh) to png frames, e.g.,

```
ffmpeg -i path_to_compressed/003.mkv ./003/%3d.png
ffmpeg -i path_to_raw/003.mkv ./003_raw/%3d.png
```

## Single-frame baseline models

The single-frame baseline models include ARCNN, DnCNN and QE-CNN. The arguments include:

```
parser.add_argument("--model", type=str, default='QECNN', choices=['ARCNN', 'DnCNN', 'QECNN']) # choose the model
parser.add_argument("--raw_path", type=str, default='./003_raw') # path to raw frames
parser.add_argument("--com_path", type=str, default='./003') # path to compressed frames
parser.add_argument("--enh_path", type=str, default='./003_enh') # path to save enhanced frames
parser.add_argument("--frame_num", type=int, default=250) # frame number
parser.add_argument("--H", type=int, default=536) # Height
parser.add_argument("--W", type=int, default=960) # Width
```

For example,

```
python SingleFrame.py --model QECNN --raw_path ./003_raw --com_path ./003 --enh_path ./003_enh --frame_num 250 --H 536 --W 960
```
