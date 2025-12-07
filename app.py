import tensorflow as tf
from transformers import TFBertForTokenClassification, BertTokenizer
import numpy as np
from flask import Flask, jsonify, request
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

save_path = 'D:/Music_Tagging_AI/model'

tags = ['O', 'B-SONG', 'I-SONG', 'B-ARTIST', 'I-ARTIST']
id_to_tag = {i: tag for i, tag in enumerate(tags)} 
WEIGHTS_TENSOR = tf.constant(np.array([0.2, 2, 2, 2, 2], dtype=np.float32), dtype=tf.float32)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true_int = tf.cast(y_true, dtype=tf.int64)
    loss_mask = tf.cast(tf.not_equal(y_true_int, -100), dtype=y_pred.dtype)
    y_true_masked = tf.where(tf.equal(y_true_int, -100), tf.constant(0, dtype=y_true_int.dtype), y_true_int)
    sample_weights = tf.gather(WEIGHTS_TENSOR, y_true_masked)

    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction=tf.keras.losses.Reduction.NONE)

    token_loss = loss_fct(y_true_masked, y_pred)
    weighted_masked_loss = token_loss * sample_weights * loss_mask

    normalization_factor = tf.reduce_sum(sample_weights * loss_mask)
    normalization_factor = tf.where(tf.equal(normalization_factor, 0.), 1., normalization_factor)

    return tf.reduce_sum(weighted_masked_loss) / normalization_factor

def masked_accuracy(y_true, y_pred):
    y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
    y_true_int = tf.cast(y_true, dtype=tf.int64)

    mask = tf.cast(tf.not_equal(y_true_int, -100), dtype=tf.float32)
    matches = tf.cast(tf.equal(y_true_int, y_pred_argmax), dtype=tf.float32)

    masked_matches = matches * mask

    accuracy = tf.reduce_sum(masked_matches) / tf.reduce_sum(mask)

    return accuracy

try:
    loaded_model = tf.keras.models.load_model(
        save_path,
        custom_objects={
            'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy,
            'masked_accuracy': masked_accuracy
        }
    )
    print("모델 불러오기 성공.")
except Exception as e:
    print(f"로드 중 오류 발생: {e}")


def input_model(text, model, id_to_tag):
    label = []
    ARTIST_LIST = []
    SONG_LIST = []

    text_words = tokenizer.tokenize(text)
    attention_mask = [1] * len(text_words)
    token_type_ids = [0] * len(text_words)

    tokenizer_text = tokenizer.convert_tokens_to_ids(text_words)

    tokenizer_text = np.array(tokenizer_text)
    attention_mask = np.array(attention_mask)
    token_type_ids = np.array(token_type_ids)

    input_data = {
        'input_ids': tf.expand_dims(tokenizer_text, 0),
        'attention_mask': tf.expand_dims(attention_mask, 0),
        'token_type_ids': tf.expand_dims(token_type_ids, 0)
    }

    predictions = model(input_data)
    logits = predictions['logits'][0].numpy()
    for i in range(len(logits)):
        label.append(id_to_tag[np.argmax(logits[i])])

    arttist = ""
    song = ""

    for i, tag in enumerate(label):
        if tag == 'B-ARTIST':
            arttist = text_words[i]
        elif tag == "I-ARTIST":
            if "##" in text_words[i]:
                arttist += text_words[i]
            else:
                arttist += (" " + text_words[i])
        elif tag == 'B-SONG':
            song = text_words[i]
        elif tag == 'I-SONG':
            if "##" in text_words[i]:
                song += text_words[i]
            else:
                song += (" " + text_words[i])
        elif tag == 'O':
            if arttist != "":
                arttist.replace("##", "")
                ARTIST_LIST.append(arttist.replace("##", ""))
            elif song != "":
                SONG_LIST.append(song.replace("##", ""))
            arttist = ""
            song = ""
    
    return ARTIST_LIST, SONG_LIST

@app.route('/')
def home():
    return "Music_Tagging_AI is running."

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Music_Tagging_AI is running."}), 200

@app.route('/upload_json', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
    else:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    json_dict = {}
    text = data
    ARTIST, SONG  = input_model(text, loaded_model, id_to_tag)
    json_dict['원본 텍스트'] = text
    for i, s in enumerate(ARTIST):
        json_dict[f'아티스트 {i+1}'] = s
    for i, s in enumerate(SONG):
        json_dict[f'곡명 {i+1}'] = s

    return jsonify(json_dict), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)