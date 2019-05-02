import os
import shutil
import subprocess
import zipfile
from collections import OrderedDict

import pandas as pd
from joblib import Parallel
from joblib import delayed

if not os.path.exists('data/temp'):
    os.mkdir('data/temp')
if not os.path.exists('data/kinetics600'):
    os.mkdir('data/kinetics600')
if not os.path.exists('data/temp/kinetics600'):
    os.mkdir('data/temp/kinetics600')

kinetics_train_split = zipfile.ZipFile('data/kinetics_600_train (1).zip')
kinetics_train_split.extractall('data/temp/kinetics600')
kinetics_train_split.close()

kinetics_val_split = zipfile.ZipFile('data/kinetics_600_val (1).zip')
kinetics_val_split.extractall('data/temp/kinetics600')
kinetics_val_split.close()

kinetics_test_split = zipfile.ZipFile('data/kinetics_600_test (2).zip')
kinetics_test_split.extractall('data/temp/kinetics600')
kinetics_test_split.close()


def parse_kinetics_annotations(input_csv):
    df = pd.read_csv(input_csv)
    columns = OrderedDict(
        [('youtube_id', 'video-id'), ('time_start', 'start-time'), ('time_end', 'end-time'), ('label', 'label-name')])
    df.rename(columns=columns, inplace=True)
    return df


def create_video_folders(dataset, output_dir, split):
    label_to_dir = {}
    if not os.path.exists(os.path.join(output_dir, split)):
        os.makedirs(os.path.join(output_dir, split))
    for label_name in dataset['label-name'].unique():
        this_dir = os.path.join(output_dir, split, label_name)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name] = this_dir
    if not os.path.exists('data/kinetics600_labels.txt'):
        with open('data/kinetics600_labels.txt', 'w') as f:
            for label_name in dataset['label-name'].unique():
                f.write(label_name + '\n')

    return label_to_dir


def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    dirname = label_to_dir[row['label-name']]
    output_filename = os.path.join(dirname, basename)
    return output_filename


def download_clip(video_identifier, output_filename, start_time, end_time, url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # construct command line for getting the direct video link
    command = ['youtube-dl',
               '--quiet', '--no-warnings',
               '-f', '18',  # 640x360 h264 encoded video
               '--get-url',
               '"%s"' % (url_base + video_identifier)]
    command = ' '.join(command)

    status, attempts, direct_download_url = False, 0, None
    while True:
        try:
            direct_download_url = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            direct_download_url = direct_download_url.strip().decode('utf-8')
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == 3:
                return status, err.output
            else:
                continue
        break
    # construct command to trim the videos (ffmpeg required, it should be compiled with openssl)
    command = ['/usr/local/bin/ffmpeg',
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-i', '"%s"' % direct_download_url,
               '-c:v', 'libx264',
               '-c:a', 'aac',
               '-loglevel', 'fatal',
               '"%s"' % output_filename]
    command = ' '.join(command)

    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, b'ERROR: Something is wrong with ffmpeg.'

    return True, 'Status: Downloaded.'


def download_clip_wrapper(row, label_to_dir, trim_format, index):
    output_filename = construct_video_filename(row, label_to_dir, trim_format)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]
    if os.path.exists(output_filename):
        print('Index: %-16s Clip-ID: %-31s %-50s' % (index, clip_id, 'Status: Exists.'))
    else:
        downloaded, log = download_clip(row['video-id'], output_filename, row['start-time'], row['end-time'])
        if downloaded:
            print('Index: %-16s Clip-ID: %-31s %-50s' % (index, clip_id, 'Status: Downloaded.'))
        else:
            print('Index: %-16s Clip-ID: %-31s %-50s' % (index, clip_id, log.strip().decode('utf-8')))


def download_kinetics(input_csv, split, output_dir='data/kinetics600', trim_format='%06d'):
    # read and parse Kinetics
    dataset = parse_kinetics_annotations(input_csv)

    # create folders where videos will be saved later
    label_to_dir = create_video_folders(dataset, output_dir, split)
    num_data = len(dataset)

    # download all clips
    Parallel(n_jobs=24)(
        delayed(download_clip_wrapper)(row, label_to_dir, trim_format, '{}/{}'.format(str(i + 1), str(num_data))) for
        i, row
        in dataset.iterrows())


for split_file in ['kinetics_val.csv', 'kinetics_600_test.csv', 'kinetics_train.csv']:
    split_mode = split_file.split('_')[-1].split('.')[0]
    print('Download {} part of kinetics600 dataset'.format(split_mode))
    download_kinetics('data/temp/kinetics600/{}'.format(split_file), split=split_mode)
    # clean the corrupted videos
    print('Check the videos about {} part of kinetics600 dataset, '
          'if the video is corrupted, it will be deleted'.format(split_mode))
    for label in sorted(os.listdir('data/kinetics600/{}'.format(split_mode))):
        for video in sorted(os.listdir('data/kinetics600/{}/{}'.format(split_mode, label))):

            command = ['/usr/local/bin/ffmpeg',
                       '-loglevel', 'error',
                       '-i', '"%s"' % 'data/kinetics600/{}/{}/{}'.format(split_mode, label, video),
                       '-f', 'null',
                       '- 1>data/temp/error.log']
            command = ' '.join(command)

            try:
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                print('Clip-Name: %-40s %-50s' % (video, 'Status: Success Saved.'))
            except subprocess.CalledProcessError as err:
                os.remove('data/kinetics600/{}/{}/{}'.format(split_mode, label, video))
                print('Clip-Name: %-40s %-50s' % (video, 'Status: Corrupted.'))

# clean tmp dir.
shutil.rmtree('data/temp')
