import subprocess
import math
import os
import glob
import shlex
from pathlib import Path
from source.config import SEGMENTS_FOLDER, VIDEOS_FOLDER


def get_video_length(filename):

    output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filename)).strip()
    video_length = int(float(output))
    print("Video length in seconds: "+str(video_length))
    return video_length

def ceildiv(a, b):
    return int(math.ceil(a / float(b)))

def split_by_seconds(filename, segment_path, split_length, chunk, vcodec="copy", acodec="copy", extra="", **kwargs):
    if split_length and split_length <= 0:
        print("Split length can't be 0")
        raise SystemExit

    video_length = get_video_length(filename)
    split_count = ceildiv(video_length, split_length)
    if(split_count == 1):
        print("Video length is less then the target split length.")
        raise SystemExit

    split_cmd = ["ffmpeg", "-y", "-i", filename, "-vcodec", vcodec, "-acodec", acodec] + shlex.split(extra)
    try:
        if segment_path is None:
            Path(chunk).mkdir(parents=True, exist_ok=True)
            filebase = os.path.join(chunk, "".join(filename.split(".")[:-1]).split("/")[-1])
        else:
            Path(os.path.join(segment_path, chunk)).mkdir(parents=True, exist_ok=True)
            filebase = os.path.join(segment_path, chunk, "".join(filename.split(".")[:-1]).split("/")[-1])
        fileext = filename.split(".")[-1]
    except IndexError as e:
        raise IndexError("No . in filename. Error: " + str(e))
    for n in range(0, split_count):
        split_args = []
        if n == 0:
            split_start = 0
        else:
            split_start = split_length * n

        split_args += ["-ss", str(split_start), "-t", str(split_length), filebase + "-" + f"{n+1:03}" + "-of-" + f"{split_count:03}" + "." + fileext]
        print("About to run: "+" ".join(split_cmd + split_args))
        subprocess.check_output(split_cmd + split_args)


if __name__ == '__main__':
    for file in glob.glob(VIDEOS_FOLDER + "*.mp4"):
        seconds = 0.5
        chunk = "chunk_" + file.split("_")[1]
        split_by_seconds(filename=file, segment_path=SEGMENTS_FOLDER, split_length=0.5, chunk=chunk, vcodec="libx264", extra="-vf 'scale=1280:720' -threads 32")
