import json
import os
import shutil
import zipfile

import rarfile
from sklearn.model_selection import train_test_split

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

if not os.path.exists('data/ucf101/labels.txt'):
    with open('data/ucf101/labels.txt', 'w') as f:
        for line in open('data/temp/ucf101/ucfTrainTestlist/classInd.txt', 'r'):
            f.write(line.split(' ')[1])

train_video_files, val_video_files, test_video_files, train_video_labels = [], [], [], []
for line in open('data/temp/ucf101/ucfTrainTestlist/trainlist01.txt', 'r'):
    train_video_files.append(line.split(' ')[0])
    train_video_labels.append(line.split(' ')[1].replace('\n', ''))
for line in open('data/temp/ucf101/ucfTrainTestlist/testlist01.txt', 'r'):
    test_video_files.append(line.replace('\n', ''))
train_video_files, val_video_files, _, _ = train_test_split(train_video_files, train_video_labels, test_size=0.2,
                                                            random_state=42)

ucf101_videos = rarfile.RarFile('data/UCF101.rar')
ucf101_videos.extractall('data/temp/ucf101')
ucf101_videos.close()

if not os.path.exists('data/ucf101/train'):
    os.mkdir('data/ucf101/train')
for video in train_video_files:
    if not os.path.exists('data/ucf101/train/{}'.format(video.split('/')[0])):
        os.mkdir('data/ucf101/train/{}'.format(video.split('/')[0]))
    shutil.move('data/temp/ucf101/UCF-101/{}'.format(video), 'data/ucf101/train/{}'.format(video))

if not os.path.exists('data/ucf101/val'):
    os.mkdir('data/ucf101/val')
for video in val_video_files:
    if not os.path.exists('data/ucf101/val/{}'.format(video.split('/')[0])):
        os.mkdir('data/ucf101/val/{}'.format(video.split('/')[0]))
    shutil.move('data/temp/ucf101/UCF-101/{}'.format(video), 'data/ucf101/val/{}'.format(video))

if not os.path.exists('data/ucf101/test'):
    os.mkdir('data/ucf101/test')
for video in test_video_files:
    if not os.path.exists('data/ucf101/test/{}'.format(video.split('/')[0])):
        os.mkdir('data/ucf101/test/{}'.format(video.split('/')[0]))
    shutil.move('data/temp/ucf101/UCF-101/{}'.format(video), 'data/ucf101/test/{}'.format(video))

# preprocess hmdb51 files
if not os.path.exists('data/hmdb51'):
    os.mkdir('data/hmdb51')
if not os.path.exists('data/temp/hmdb51'):
    os.mkdir('data/temp/hmdb51')

hmdb51_splits = rarfile.RarFile('data/test_train_splits.rar')
hmdb51_splits.extractall('data/temp/hmdb51')
hmdb51_splits.close()

labels = []
for file in os.listdir('data/temp/hmdb51/testTrainMulti_7030_splits'):
    labels.append(file.split('_test_split')[0])
labels = set(labels)

if not os.path.exists('data/hmdb51/labels.txt'):
    with open('data/hmdb51/labels.txt', 'w') as f:
        for label in labels:
            f.write(label + '\n')

train_video_files, val_video_files, test_video_files = [], [], []
for file in os.listdir('data/temp/hmdb51/testTrainMulti_7030_splits'):
    if file.endswith('test_split1.txt'):
        for line in open('data/temp/hmdb51/testTrainMulti_7030_splits/{}'.format(file), 'r'):
            if line.split(' ')[1].replace('\n', '') == '1':
                train_video_files.append(file.split('_')[0] + '/' + line.split(' ')[0])
            if line.split(' ')[1].replace('\n', '') == '2':
                test_video_files.append(file.split('_')[0] + '/' + line.split(' ')[0])
            if line.split(' ')[1].replace('\n', '') == '0':
                val_video_files.append(file.split('_')[0] + '/' + line.split(' ')[0])

hmdb51_videos = rarfile.RarFile('data/hmdb51_org.rar')
hmdb51_videos.extractall('data/temp/hmdb51')
hmdb51_videos.close()
for file in os.listdir('data/temp/hmdb51/'):
    if file.endswith('.rar'):
        rar_file = rarfile.RarFile('data/temp/hmdb51/{}'.format(file))
        rar_file.extractall('data/temp/hmdb51')
        rar_file.close()

if not os.path.exists('data/hmdb51/train'):
    os.mkdir('data/hmdb51/train')
for video in train_video_files:
    if not os.path.exists('data/hmdb51/train/{}'.format(video.split('/')[0])):
        os.mkdir('data/hmdb51/train/{}'.format(video.split('/')[0]))
    shutil.move('data/temp/hmdb51/{}'.format(video), 'data/hmdb51/train/{}'.format(video))

if not os.path.exists('data/hmdb51/val'):
    os.mkdir('data/hmdb51/val')
for video in val_video_files:
    if not os.path.exists('data/hmdb51/val/{}'.format(video.split('/')[0])):
        os.mkdir('data/hmdb51/val/{}'.format(video.split('/')[0]))
    shutil.move('data/temp/hmdb51/{}'.format(video), 'data/hmdb51/val/{}'.format(video))

if not os.path.exists('data/hmdb51/test'):
    os.mkdir('data/hmdb51/test')
for video in test_video_files:
    if not os.path.exists('data/hmdb51/test/{}'.format(video.split('/')[0])):
        os.mkdir('data/hmdb51/test/{}'.format(video.split('/')[0]))
    shutil.move('data/temp/hmdb51/{}'.format(video), 'data/hmdb51/test/{}'.format(video))

# preprocess ss174 files
if not os.path.exists('data/ss174'):
    os.mkdir('data/ss174')
if not os.path.exists('data/temp/ss174'):
    os.mkdir('data/temp/ss174')

if not os.path.exists('data/ss174/labels.txt'):
    with open('data/ss174/labels.txt', 'w') as f:
        with open('data/something-something-v2-labels.json', 'r') as load_f:
            labels = json.load(load_f)
        for label in labels.keys():
            f.write(label + '\n')

train_video_files, val_video_files, test_video_files = [], [], []
with open('data/something-something-v2-train.json', 'r') as load_f:
    videos = json.load(load_f)
    for video in videos:
        train_video_files.append(video['template'].replace('[', '').replace(']', '') + '/' + video['id'] + '.webm')
with open('data/something-something-v2-validation.json', 'r') as load_f:
    videos = json.load(load_f)
    for video in videos:
        val_video_files.append(video['template'].replace('[', '').replace(']', '') + '/' + video['id'] + '.webm')
with open('something-something-v2-test.json', 'r') as load_f:
    videos = json.load(load_f)
    for video in videos:
        test_video_files.append(video['id'] + '.webm')

if not os.path.exists('data/ss174/train'):
    os.mkdir('data/ss174/train')
for video in train_video_files:
    if not os.path.exists('data/ss174/train/{}'.format(video.split('/')[0])):
        os.mkdir('data/ss174/train/{}'.format(video.split('/')[0]))
    shutil.move('data/20bn-something-something-v2/{}'.format(video.split('/')[1]), 'data/ss174/train/{}'.format(video))

if not os.path.exists('data/ss174/val'):
    os.mkdir('data/ss174/val')
for video in val_video_files:
    if not os.path.exists('data/ss174/val/{}'.format(video.split('/')[0])):
        os.mkdir('data/ss174/val/{}'.format(video.split('/')[0]))
    shutil.move('data/20bn-something-something-v2/{}'.format(video.split('/')[1]), 'data/ss174/val/{}'.format(video))

if not os.path.exists('data/ss174/test'):
    os.mkdir('data/ss174/test')
for video in test_video_files:
    shutil.move('data/20bn-something-something-v2/{}'.format(video), 'data/ss174/test/{}'.format(video))

# remove these files to make the data dir more clear
os.remove('data/UCF101.rar')
os.remove('data/UCF101TrainTestSplits-RecognitionTask.zip')
os.remove('data/hmdb51_org.rar')
os.remove('data/test_train_splits.rar')
os.remove('data/something-something-v2-labels.json')
os.remove('data/something-something-v2-train.json')
os.remove('data/something-something-v2-validation.json')
os.remove('data/something-something-v2-test.json')
os.remove('data/20bn-something-something-v2-00')
os.remove('data/20bn-something-something-v2-01')
os.remove('data/20bn-something-something-v2-02')
os.remove('data/20bn-something-something-v2-03')
os.remove('data/20bn-something-something-v2-04')
os.remove('data/20bn-something-something-v2-05')
os.remove('data/20bn-something-something-v2-06')
os.remove('data/20bn-something-something-v2-07')
os.remove('data/20bn-something-something-v2-08')
os.remove('data/20bn-something-something-v2-09')
os.remove('data/20bn-something-something-v2-11')
os.remove('data/20bn-something-something-v2-12')
os.remove('data/20bn-something-something-v2-13')
os.remove('data/20bn-something-something-v2-14')
os.remove('data/20bn-something-something-v2-15')
os.remove('data/20bn-something-something-v2-16')
os.remove('data/20bn-something-something-v2-17')
os.remove('data/20bn-something-something-v2-18')
os.remove('data/20bn-something-something-v2-19')
shutil.rmtree('data/20bn-something-something-v2')
shutil.rmtree('data/temp')
