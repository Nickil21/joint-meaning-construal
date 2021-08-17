import os
import glob
import shutil
import argparse
import pandas as pd
from source.video_splitter import split_by_seconds
from source import inference

UPLOAD_FOLDER = 'static/uploads/'


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Hand gesture classification')

    # Add the arguments
    parser.add_argument('Path',
                        metavar='path',
                        type=str,
                        help='the path to the video file (.mp4 format)')

    # Execute parse_args()
    args = parser.parse_args()
    # delete uploaded files
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    filename = args.Path
    chunk = "chunk_" + filename.split("_")[1]
    split_by_seconds(filename=filename, segment_path=None, split_length=0.5, chunk=UPLOAD_FOLDER + chunk,
                      vcodec="libx264", extra="-vf 'scale=1280:720' -threads 32")
    lst = []
    for f in glob.glob(UPLOAD_FOLDER + "{}/*".format(chunk)):
        d = inference.evaluate(f)
        d['name'] = f.split("/")[-1]
        lst.append(d)
    df = pd.DataFrame(lst)
    df.sort_values('name').to_csv(UPLOAD_FOLDER + "hand_gestures_predicted.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()