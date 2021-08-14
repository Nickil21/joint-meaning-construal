import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras

from source.train import PositionalEmbedding, TransformerEncoder
from source.helper import predict_action


def evaluate(test_video):
    print("TEST VIDEO", test_video)
    model = keras.models.load_model("source/gesture_model.h5", custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                                               'TransformerEncoder': TransformerEncoder})
    return predict_action(model=model, path=test_video)

# print(evaluate("source/data/test/chunk_105_trigger_from-to-14-of-36.mp4"))