import os
import sys
import time
import argparse
import subprocess

allowed_set = set(['avi', 'mp4', 'mkv', 'flv', 'wmv', 'mov']) 

def allowed_file(filename, allowed_set):
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set
    return check

def main(args):

    ffmpegCmd   = "C:/Path/ffmpeg/bin/ffmpeg.exe"   # ffmpeg路径
    videoDir    = args.videoDir
    cropDir     = args.cropDir

    if not os.path.exists(cropDir):
            os.makedirs(cropDir)

    kol_name = []
    for i in os.listdir(videoDir):
        kol_name.append(i)

    for i in range(len(kol_name)):
        # count = 0
        dir_name = kol_name[i]
        kol_dir = os.path.join(videoDir, kol_name[i])
        if os.path.isdir(kol_dir):
            kol_crop_dir = os.path.join(cropDir, kol_name[i])
            if not os.path.exists(kol_crop_dir):
                    os.makedirs(kol_crop_dir)
            for kol_video_file in os.listdir(kol_dir):
                if allowed_file(filename=kol_video_file, allowed_set=allowed_set):   # | if (kol_video_file.endswith(".mp4")):
                    kol_video_path  = os.path.join(kol_dir, kol_video_file)
                    kol_crop_path   = os.path.join(kol_crop_dir, kol_video_file.rsplit('.')[0] + "_%04d.png")
                    video2framesCmd = ffmpegCmd + " -i " + kol_video_path + " -f image2 -vf fps=fps=3 -qscale:v 2 " + kol_crop_path
                    try:
                        subprocess.call(video2framesCmd, shell=True)
                        print('crop from %s to %s' % (kol_video_path, kol_crop_path))
                    except:
                        continue

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('videoDir', type=str, help='KOL video root dirctory which contains different KOL identity directories.')
    parser.add_argument('cropDir', type=str, help='KOL crops root dirctory which contains different KOL identity directories which containing labeled and aligned face thumbnails.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    start_time = time.time()
    print('Program.py start at: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main(parse_arguments(sys.argv[1:]))
    print('Program.py end at: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('Program.py all time: {0} seconds = {1} minutes = {2} hrs'.format((time.time() - start_time),
                                                                     (time.time() - start_time) / 60,
                                                                     (time.time() - start_time) / 3600))