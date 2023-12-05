from flask import Flask, request, jsonify, render_template
import numpy as np
import re
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

# Load the trained models and word-to-index dictionaries
model_overgeneralization = load_model('Overgeneralization_model.h5')
model_all_or_none_thinking = load_model('All or none thinking_model.h5')
model_mind_reading = load_model('Mind Reading_model.h5')
model_fortune_telling_error = load_model('Fortune telling error_model.h5')
model_labeling = load_model('Labeling_model.h5')
model_should_statement = load_model('Should Statement_model.h5')
model_emotional_reasoning = load_model('Emotional reasoning_model.h5')
model_personalization = load_model('Personalization_model.h5')


word_index_overgeneralization = np.load('Overgeneralization_index.npy', allow_pickle=True).item()
word_index_all_or_none_thinking = np.load('All or none thinking_word_index.npy', allow_pickle=True).item()
word_index_mind_reading = np.load('Mind Reading_index.npy', allow_pickle=True).item()
word_index_fortune_telling_error = np.load('Fortune telling error_index.npy', allow_pickle=True).item()
word_index_labeling = np.load('Labeling_index.npy', allow_pickle=True).item()
word_index_should_statement = np.load('Should Statement_index.npy', allow_pickle=True).item()
word_index_emotional_reasoning = np.load('Emotional reasoning_index.npy', allow_pickle=True).item()
word_index_personalization = np.load('Personalization_index.npy', allow_pickle=True).item()


tokenizer_overgeneralization = Tokenizer()
tokenizer_overgeneralization.word_index = word_index_overgeneralization

tokenizer_all_or_none_thinking = Tokenizer()
tokenizer_all_or_none_thinking.word_index = word_index_all_or_none_thinking

tokenizer_mind_reading = Tokenizer()
tokenizer_mind_reading.word_index = word_index_mind_reading

tokenizer_fortune_telling_error = Tokenizer()
tokenizer_fortune_telling_error.word_index = word_index_fortune_telling_error

tokenizer_labeling = Tokenizer()
tokenizer_labeling.word_index = word_index_labeling

tokenizer_should_statement = Tokenizer()
tokenizer_should_statement.word_index = word_index_should_statement

tokenizer_emotional_reasoning = Tokenizer()
tokenizer_emotional_reasoning.word_index = word_index_emotional_reasoning

tokenizer_personalization = Tokenizer()
tokenizer_personalization.word_index = word_index_personalization


# Maximum sequence length
max_sequence_length = 300

# Make the prediction for Overgeneralization
def predict_overgeneralization(comment):
    sequence = tokenizer_overgeneralization.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_overgeneralization.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'Overgeneralization: Yes ' + str(prediction)
    else:
        result = 'Overgeneralization: No ' + str(prediction)
    return result

# Make the prediction for All or None Thinking
def predict_all_or_none_thinking(comment):
    sequence = tokenizer_all_or_none_thinking.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_all_or_none_thinking.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'All or None Thinking: Yes   ' + str(prediction)
    else:
        result = 'All or None Thinking: No   ' + str(prediction)
    return result

# Make the prediction for Mind Reading
def predict_mind_reading(comment):
    sequence = tokenizer_mind_reading.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_mind_reading.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'Mind Reading: Yes   ' + str(prediction)
    else:
        result = 'Mind Reading: No   ' + str(prediction)
    return result

# Make the prediction for Fortune Telling Error
def predict_fortune_telling_error(comment):
    sequence = tokenizer_fortune_telling_error.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_fortune_telling_error.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'Fortune Telling Error: Yes   ' + str(prediction)
    else:
        result = 'Fortune Telling Error: No   ' + str(prediction)
    return result


# Make the prediction for Labeling
def predict_labeling(comment):
    sequence = tokenizer_labeling.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_labeling.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'Labeling: Yes   ' + str(prediction)
    else:
        result = 'Labeling: No   ' + str(prediction)
    return result

# Make the prediction for Should Statement
def predict_should_statement(comment):
    sequence = tokenizer_should_statement.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_should_statement.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'Should Statement: Yes   ' + str(prediction)
    else:
        result = 'Should Statement: No   ' + str(prediction)
    return result

# Make the prediction for Emotional Reasoning
def predict_emotional_reasoning(comment):
    sequence = tokenizer_emotional_reasoning.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_emotional_reasoning.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'Emotional Reasoning: Yes   ' + str(prediction)
    else:
        result = 'Emotional Reasoning: No   ' + str(prediction)
    return result

# Make the prediction for Personalization
def predict_personalization(comment):
    sequence = tokenizer_personalization.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model_personalization.predict(padded_sequence)[0][0]

    if prediction >= 0.3:
        result = 'Personalization: Yes   ' + str(prediction)
    else:
        result = 'Personalization: No   ' + str(prediction)
    return result



# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data['comment']

    # Get predictions from all models
    result_overgeneralization = predict_overgeneralization(comment)
    result_all_or_none_thinking = predict_all_or_none_thinking(comment)
    result_mind_reading = predict_mind_reading(comment)
    result_fortune_telling_error = predict_fortune_telling_error(comment)
    result_labeling = predict_labeling(comment)
    result_should_statement = predict_should_statement(comment)
    result_emotional_reasoning = predict_emotional_reasoning(comment)
    result_personalization = predict_personalization(comment)


    return jsonify({
        'result_overgeneralization': result_overgeneralization,
        'result_all_or_none_thinking': result_all_or_none_thinking,
        'result_mind_reading': result_mind_reading,
        'result_fortune_telling_error': result_fortune_telling_error,
        'result_labeling': result_labeling,
        'result_should_statement': result_should_statement,
        'result_emotional_reasoning': result_emotional_reasoning,
        'result_personalization': result_personalization,
    })

# Define the route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
