import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample Data: English to French
data = [
    ("hello", "bonjour"),
    ("how are you?", "comment ça va?"),
    ("good morning", "bonjour"),
    ("thank you", "merci"),
    ("yes", "oui"),
    ("no", "non"),
]

# Preprocess data
english_sentences, french_sentences = zip(*data)

# Tokenize and Pad Sequences
def tokenize_and_pad(sentences, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, filters='')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding="post")
    return tokenizer, padded

english_tokenizer, english_padded = tokenize_and_pad(english_sentences)
french_tokenizer, french_padded = tokenize_and_pad(french_sentences)

# Vocabulary sizes
english_vocab_size = len(english_tokenizer.word_index) + 1
french_vocab_size = len(french_tokenizer.word_index) + 1

# Model Parameters
embedding_dim = 256
units = 512

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(english_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(units, return_state=True)(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(french_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Attention Mechanism
attention = tf.keras.layers.Attention()
context_vector = attention([decoder_outputs, encoder_embedding])

# Combine context with decoder outputs
decoder_combined_context = tf.concat([context_vector, decoder_outputs], axis=-1)

# Final Dense Layer
dense_output = Dense(french_vocab_size, activation="softmax")(decoder_combined_context)

# Model
model = Model([encoder_inputs, decoder_inputs], dense_output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Prepare Decoder Targets (shifted French sentences)
decoder_target = french_padded[:, 1:]
decoder_input = french_padded[:, :-1]

# Train the Model
model.fit(
    [english_padded, decoder_input],
    decoder_target[:, :, np.newaxis],
    batch_size=32,
    epochs=100,
)

# Translation Function
def translate(input_sentence):
    # Tokenize input sentence
    input_sequence = english_tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=english_padded.shape[1], padding="post")
    
    # Generate prediction
    decoder_input_seq = np.zeros((1, 1))  # Start with an empty sequence
    translation = []
    for _ in range(20):  # Max translation length
        prediction = model.predict([input_sequence, decoder_input_seq])
        predicted_token = np.argmax(prediction[0, -1, :])
        if predicted_token == 0:  # End of sequence token
            break
        translation.append(french_tokenizer.index_word[predicted_token])
        decoder_input_seq = np.append(decoder_input_seq, [[predicted_token]], axis=1)
    return " ".join(translation)

# Test Translation
print(translate("hello"))
