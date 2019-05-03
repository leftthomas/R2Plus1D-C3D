import os
import random
import shutil
import zipfile

import rarfile

if not os.path.exists('data/temp'):
    os.mkdir('data/temp')

# preprocess ucf101 files
if not os.path.exists('data/ucf101'):
    os.mkdir('data/ucf101')
if not os.path.exists('data/temp/ucf101'):
    os.mkdir('data/temp/ucf101')

ucf101_splits = zipfile.ZipFile('data/UCF101TrainTestSplits-RecognitionTask.zip')
ucf101_splits.extractall('data/temp/ucf101')
ucf101_splits.close()

if not os.path.exists('data/ucf101_labels.txt'):
    with open('data/ucf101_labels.txt', 'w') as f:
        for line in open('data/temp/ucf101/ucfTrainTestlist/classInd.txt', 'r'):
            f.write(line.split(' ')[1])

train_video_files, test_video_files = [], []
for line in open('data/temp/ucf101/ucfTrainTestlist/trainlist01.txt', 'r'):
    train_video_files.append(line.split(' ')[0])

for line in open('data/temp/ucf101/ucfTrainTestlist/testlist01.txt', 'r'):
    test_video_files.append(line.replace('\n', ''))

val_video_files = random.sample(test_video_files, int(len(test_video_files) * 0.2))

ucf101_videos = rarfile.RarFile('data/UCF101.rar')
ucf101_videos.extractall('data/temp/ucf101')
ucf101_videos.close()

if not os.path.exists('data/ucf101/train'):
    os.mkdir('data/ucf101/train')
for video in train_video_files:
    if not os.path.exists('data/ucf101/train/{}'.format(video.split('/')[0])):
        os.mkdir('data/ucf101/train/{}'.format(video.split('/')[0]))
    shutil.copy('data/temp/ucf101/UCF-101/{}'.format(video), 'data/ucf101/train/{}'.format(video))

if not os.path.exists('data/ucf101/val'):
    os.mkdir('data/ucf101/val')
for video in val_video_files:
    if not os.path.exists('data/ucf101/val/{}'.format(video.split('/')[0])):
        os.mkdir('data/ucf101/val/{}'.format(video.split('/')[0]))
    shutil.copy('data/temp/ucf101/UCF-101/{}'.format(video), 'data/ucf101/val/{}'.format(video))

if not os.path.exists('data/ucf101/test'):
    os.mkdir('data/ucf101/test')
for video in test_video_files:
    if not os.path.exists('data/ucf101/test/{}'.format(video.split('/')[0])):
        os.mkdir('data/ucf101/test/{}'.format(video.split('/')[0]))
    shutil.copy('data/temp/ucf101/UCF-101/{}'.format(video), 'data/ucf101/test/{}'.format(video))

# preprocess hmdb51 files
if not os.path.exists('data/hmdb51'):
    os.mkdir('data/hmdb51')
if not os.path.exists('data/temp/hmdb51'):
    os.mkdir('data/temp/hmdb51')

hmdb51_splits = rarfile.RarFile('data/test_train_splits.rar')
hmdb51_splits.extractall('data/temp/hmdb51')
hmdb51_splits.close()

labels = []
for file in sorted(os.listdir('data/temp/hmdb51/testTrainMulti_7030_splits')):
    labels.append(file.split('_test_split')[0])
labels = sorted(set(labels))

if not os.path.exists('data/hmdb51_labels.txt'):
    with open('data/hmdb51_labels.txt', 'w') as f:
        for current_label in labels:
            f.write(current_label + '\n')

train_video_files, val_video_files, test_video_files = [], [], []
for file in sorted(os.listdir('data/temp/hmdb51/testTrainMulti_7030_splits')):
    if file.endswith('test_split1.txt'):
        for line in open('data/temp/hmdb51/testTrainMulti_7030_splits/{}'.format(file), 'r'):
            if line.split(' ')[1].replace('\n', '') == '1':
                train_video_files.append(file.split('_test_split')[0] + '/' + line.split(' ')[0])
            if line.split(' ')[1].replace('\n', '') == '2':
                test_video_files.append(file.split('_test_split')[0] + '/' + line.split(' ')[0])
            if line.split(' ')[1].replace('\n', '') == '0':
                val_video_files.append(file.split('_test_split')[0] + '/' + line.split(' ')[0])

hmdb51_videos = rarfile.RarFile('data/hmdb51_org.rar')
hmdb51_videos.extractall('data/temp/hmdb51')
hmdb51_videos.close()
for file in sorted(os.listdir('data/temp/hmdb51/')):
    if file.endswith('.rar'):
        rar_file = rarfile.RarFile('data/temp/hmdb51/{}'.format(file))
        rar_file.extractall('data/temp/hmdb51')
        rar_file.close()

if not os.path.exists('data/hmdb51/train'):
    os.mkdir('data/hmdb51/train')
for video in train_video_files:
    if not os.path.exists('data/hmdb51/train/{}'.format(video.split('/')[0])):
        os.mkdir('data/hmdb51/train/{}'.format(video.split('/')[0]))
    shutil.copy('data/temp/hmdb51/{}'.format(video), 'data/hmdb51/train/{}'.format(video))

if not os.path.exists('data/hmdb51/val'):
    os.mkdir('data/hmdb51/val')
for video in val_video_files:
    if not os.path.exists('data/hmdb51/val/{}'.format(video.split('/')[0])):
        os.mkdir('data/hmdb51/val/{}'.format(video.split('/')[0]))
    shutil.copy('data/temp/hmdb51/{}'.format(video), 'data/hmdb51/val/{}'.format(video))

if not os.path.exists('data/hmdb51/test'):
    os.mkdir('data/hmdb51/test')
for video in test_video_files:
    if not os.path.exists('data/hmdb51/test/{}'.format(video.split('/')[0])):
        os.mkdir('data/hmdb51/test/{}'.format(video.split('/')[0]))
    shutil.copy('data/temp/hmdb51/{}'.format(video), 'data/hmdb51/test/{}'.format(video))

# remove the temp dir to make the data dir more clear
shutil.rmtree('data/temp')
