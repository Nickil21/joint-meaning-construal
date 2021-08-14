from source.util import Dataset, TranscriptProcess
from source.config import ANNOTATIONS_FOLDER


def prepare_dataset(proximity_small=5, proximity_large=15):
    speakers = ['fallon', 'ellen', 'oliver', 'seth', 'corden', 'colbert', 'conan']
    triggers = [("from", "to", proximity_small), ("here", "then", proximity_small),
                ("first", "second", proximity_large), ("firstly", "secondly", proximity_large)]
    print("preparing speakers dataset ...")
    df_speakers = Dataset(speakers=speakers, triggers=triggers, num_chunks_per_trigger='all').process_triggers()
    print("saving videos ...")
    TranscriptProcess(dataframe=df_speakers).save_videos_batch(parallelize=True)
    print("removing chunks having private YouTube videos ...")
    df_speakers = TranscriptProcess(dataframe=df_speakers).remove_private_videos()
    print("saving gestures ...")
    df_gestures = TranscriptProcess(dataframe=df_speakers).add_hand_gestures(keep_relevant_cols=True)
    df_gestures.to_csv(ANNOTATIONS_FOLDER + "gesture_annotations.tsv", sep="\t", index=False)

if __name__ == "__main__":
    prepare_dataset()
