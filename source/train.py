import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from source.helper import prepare_all_videos, structure, read_datasets
from source.config import MAX_SEQ_LENGTH, NUM_FEATURES, EPOCHS, BATCH_SIZE, RAPID_ANNOTATOR_FILE


class PositionalEmbedding(layers.Layer):

    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_length': self.sequence_length,
            'output_dim': self.output_dim,
        })
        return config

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.3)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads
        })
        return config

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_experiment(train_data, train_labels):
    filepath = "source/checkpoints/"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=1)

    model = get_compiled_model()
    model.fit(train_data, train_labels, batch_size=BATCH_SIZE, validation_split=0.15, epochs=EPOCHS, callbacks=[checkpoint])

    model.load_weights(filepath)
    # _, accuracy = model.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    model.save('source/gesture_model.h5')


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    structure(data=pd.read_excel(RAPID_ANNOTATOR_FILE))
    train_df = read_datasets()
    classes = len(read_datasets(classnames=True))
    print(f"Total videos for training: {len(train_df)}")

    train_data, train_labels = prepare_all_videos(df=train_df, root_dir="source/data/train/")
    print(f"Frame features in train set: {train_data.shape}")
    run_experiment(train_data, train_labels)
