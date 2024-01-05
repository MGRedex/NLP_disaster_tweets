import tensorflow as tf
from tensorflow import keras

@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):

    def __init__(self, sentence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sentence_length = sentence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.token_embeddings = keras.layers.Embedding(input_dim = input_dim, output_dim = output_dim)
        self.position_embeddings = keras.layers.Embedding(input_dim = sentence_length, output_dim = output_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)

        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)

        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()

        config.update({
            "output_dim": self.output_dim,
            "sentence_length": self.sentence_length,
            "input_dim": self.input_dim,
        })

        return config

@keras.saving.register_keras_serializable()
class TransformerEncoder(keras.layers.Layer):

    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.attention = keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.feed_forward = keras.models.Sequential([
            keras.layers.Dense(ff_dim, activation = "relu"),
            keras.layers.Dense(embed_dim)
        ])
        self.norm_1 = keras.layers.LayerNormalization()
        self.norm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, mask = None):
        if mask is not None:
                    mask = mask[:, tf.newaxis, :]

        attention_out = self.attention(inputs, inputs, attention_mask = mask)
        norm1_out = self.norm_1(inputs + attention_out)
        ff_out = self.feed_forward(norm1_out)
        norm2_out = self.norm_2(ff_out + norm1_out)

        return norm2_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "num_heads": self.num_heads,
        })
        return config