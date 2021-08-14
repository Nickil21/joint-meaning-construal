import datetime
import glob
import os
import random
import subprocess

import h5py
import pandas as pd
from joblib import Parallel, delayed

from source.config import SPEAKERS_FOLDER, VIDEOS_FOLDER

# Fix random seed
random.seed(42)

DATA_DIR = "DATA"


def take_random_subset(data, num_chunks):
    unique_chunks = data['chunk'].unique().tolist()
    random_chunks = random.sample(unique_chunks, min(len(unique_chunks), num_chunks))
    return data[data['chunk'].isin(random_chunks)]


def crop_videos(video_link, output, trigger, start_time, end_time):
    command = """ffmpeg -y -loglevel panic \\
                        -ss {start_time} -to {end_time} -i "$(youtube-dl -f best --get-url {video_link})" \\
                        -c:v copy -c:a copy {VIDEOS_FOLDER}chunk_{output}_trigger_{trigger}.mp4
              """.format(video_link=video_link, output=output, start_time=start_time, end_time=end_time, trigger=trigger,
                         VIDEOS_FOLDER=VIDEOS_FOLDER)
    subprocess.call(command, shell=True)


class Dataset(object):
    """Dataloader for different speakers"""

    def __init__(self, speakers, triggers, num_chunks_per_trigger='all'):
        self.speakers = speakers
        self.triggers = triggers
        self.num_chunks_per_trigger = num_chunks_per_trigger

    def speaker_dataset(self, ROOT_DIR):
        missing = h5py.File(os.path.join(DATA_DIR, ROOT_DIR, "pats/data/missing_intervals.h5"))
        missing_intervals = [interval.decode("utf-8") for interval in missing['intervals']]
        files = os.path.join(DATA_DIR, ROOT_DIR, "pats/data/processed/{}/*.h5".format(ROOT_DIR))
        processed = []
        for idx, file in enumerate(glob.glob(files)):
            filename = file.split("/")[-1][:-3]
            if filename in missing_intervals:
                continue
            try:
                data = pd.read_hdf(file, key='text/meta')
            except KeyError as e:
                continue
            data['chunk'] = idx
            data['interval_id'] = filename
            processed.append(data)

        df = pd.concat(processed)
        df.to_csv("{}.tsv".format(SPEAKERS_FOLDER + ROOT_DIR), sep="\t", index=False)
        print("{} dataset saved successfully!".format(ROOT_DIR))

    def retrieve_results(self, data, trigger):
        trigger_chunks = []
        for chunk, g in data.groupby("chunk"):
            tokens = g['Word'].tolist()
            check1 = len(set(tokens) & set(trigger)) == 2
            if check1:
                check2 = tokens.index(trigger[0]) < tokens.index(trigger[1])
                check3 = tokens.index(trigger[1]) - tokens.index(trigger[0]) < trigger[2]
                if check2 and check3:
                    trigger_chunks.append(chunk)
        return trigger_chunks

    def store_results(self, data, trigger, ROOT_DIR):
        df_intervals = pd.read_csv(os.path.join(DATA_DIR, ROOT_DIR, "pats/data/cmu_intervals_df.csv"))
        data['interval_id'] = data['interval_id'].astype(str)
        df = data[data['chunk'].isin(self.retrieve_results(data, trigger))]
        df = df.merge(df_intervals, on='interval_id', how='left')
        return df

    def process_speakers(self, trigger):
        df_trigger_list = []
        for speaker in self.speakers:
            # self.speaker_dataset(speaker)
            df = pd.read_csv(SPEAKERS_FOLDER + "{}.tsv".format(speaker), sep="\t")
            df_trigger_list.append(self.store_results(data=df, trigger=trigger, ROOT_DIR=speaker))
        return pd.concat(df_trigger_list)

    def process_triggers(self):
        lst = []
        for trigger in self.triggers:
            df = self.process_speakers(trigger)
            df['trigger'] = "{}-{}".format(trigger[0], trigger[1])
            if self.num_chunks_per_trigger != 'all':
                df = take_random_subset(df, num_chunks=self.num_chunks_per_trigger)
            lst.append(df)
        return pd.concat(lst, ignore_index=True)


class TranscriptProcess(object):
    """Operations on a DataFrame"""

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.chunk = self.dataframe.groupby("chunk")
        if "start_time_mod" and "end_time_mod" not in self.dataframe.columns.tolist():
            start_time_mapper = self.chunk.apply(lambda x: self.process_time(x['start_time'].unique()[0]))
            end_time_mapper = self.chunk.apply(lambda x: self.process_time(x['end_time'].unique()[0], offset_seconds=2))
            self.dataframe["start_time_mod"] = self.dataframe['chunk'].map(start_time_mapper)
            self.dataframe["end_time_mod"] = self.dataframe['chunk'].map(end_time_mapper)

    def save_videos(self, item):
        print("Processing item {}".format(item))
        crop_videos(video_link=item['video_link'],
                    output=item['chunk'],
                    start_time=item['start_time_mod'],
                    end_time=item['end_time_mod'],
                    trigger=item['trigger'])

    def save_videos_batch(self, parallelize=True):
        metadata = self.chunk[['video_link', 'start_time_mod', 'end_time_mod', 'trigger']].agg('first').reset_index().to_dict("records")
        if parallelize:
            Parallel(n_jobs=-1)(delayed(self.save_videos)(item) for item in metadata)
        else:
            for item in metadata:
                self.save_videos(item)

    def remove_private_videos(self):
        chunks = [int(file.split("_")[1]) for file in glob.glob("TEMP/videos/*.mp4")]
        return self.dataframe[self.dataframe['chunk'].isin(chunks)]

    def process_time(self, time, offset_seconds=0):
        t = datetime.datetime.strptime(time.split(".")[0], "%Y-%m-%d %H:%M:%S")
        t_offset = t + datetime.timedelta(seconds=offset_seconds)
        hour = t_offset.hour
        minute = t_offset.minute
        second = t_offset.second
        return (f"{hour:02}" + ":" + f"{minute:02}" + ":" + f"{second:02}")

    def get_video_links(self):
        return self.chunk.apply(lambda x: x['video_link'].unique()[0]).tolist()

    def get_sentences(self):
        return self.chunk.apply(lambda x: " ".join(x['Word'].tolist())).tolist()

    def get_unique_chunks(self):
        return self.dataframe['chunk'].unique()

    def total_sentences(self):
        return len(self.get_sentences())

    def total_videos(self):
        return len(self.get_video_links())

    def add_hand_gestures(self, keep_relevant_cols=False):
        df_hand_gestures = pd.DataFrame(data={"handedness_left": 0, "handedness_right": 0, "handedness_both": 0,
                                              "shape_straight": 0, "shape_arced": 0,
                                              "direction_upward": 0, "direction_downward": 0, "directon_leftward": 0, "direction_rightward": 0,
                                              "axis_vertical": 0, "axis_horizontal": 0},
                                        index=self.dataframe.index.tolist())
        if keep_relevant_cols:
            relevant_cols = ['chunk', 'speaker', 'trigger', 'Word',
                             'start_time_mod', 'end_time_mod', 'video_link']
            self.dataframe = self.dataframe[relevant_cols]
        return pd.concat([self.dataframe, df_hand_gestures], axis=1)
