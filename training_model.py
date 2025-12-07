import tensorflow as tf
from transformers import TFBertForTokenClassification, BertConfig, BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
import numpy as np
import keras

input_file = 'music_data.CSV'
music_df = pd.read_csv(input_file)

tags = ['O', 'B-SONG', 'I-SONG', 'B-ARTIST', 'I-ARTIST']
tag_to_id = {tag: i for i, tag in enumerate(tags)}
id_to_tag = {i: tag for tag, i in tag_to_id.items()}

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

input_ids = []
attention_mask_list = []
labels_list = []

for idx in range(len(music_df)):
    text_words = tokenizer.tokenize(music_df.loc[idx, 'title'] + " " + music_df.loc[idx, 'channel_name'])
    labels = [0] * len(text_words)
    attention_mask = [1] * len(text_words)

    original_song_name_words = tokenizer.tokenize(music_df.loc[idx, '원곡명'])
    artist_words = tokenizer.tokenize(music_df.loc[idx, '아티스트'])

    def assign_bio_tags(words, entity_name_words, tag_prefix):
        if not entity_name_words: 
            return
        start = -1
        for i in range(len(words)):
            if words[i:i + len(entity_name_words)] == entity_name_words:
                start = i
                for j in range(start, start + len(entity_name_words)):
                    if j == start:
                        labels[j] = tag_to_id[f'B-{tag_prefix}']
                    else:
                        labels[j] = tag_to_id[f'I-{tag_prefix}']

    assign_bio_tags(text_words, original_song_name_words, 'SONG')
    assign_bio_tags(text_words, artist_words, 'ARTIST')

    text_words = tokenizer.convert_tokens_to_ids(text_words)

    input_ids.append(text_words)
    attention_mask_list.append(attention_mask)
    labels_list.append(labels)

max_len = len(max(input_ids, key=len))

for i in range(len(input_ids)):
    if len(input_ids[i]) < max_len:
        for j in range(max_len - len(input_ids[i])):
            input_ids[i].append(0)
            attention_mask_list[i].append(0)
            labels_list[i].append(-100)

    input_ids[i] = np.array(input_ids[i])
    attention_mask_list[i] = np.array(attention_mask_list[i])
    labels_list[i] = np.array(labels_list[i])

RANDOM_SEED = 42

train_ids, temp_ids, train_masks, temp_masks, train_labels, temp_labels = train_test_split(
    input_ids, attention_mask_list, labels_list,
    test_size=0.2,
    random_state=RANDOM_SEED)



val_ids, test_ids, val_masks, test_masks, val_labels, test_labels = train_test_split(
    temp_ids, temp_masks, temp_labels,
    test_size=0.5,
    random_state=RANDOM_SEED)

train_inputs = {
        'input_ids': train_ids,
        'attention_mask': train_masks
}

val_inputs = {
        'input_ids': val_ids,
        'attention_mask': val_masks
}

test_inputs = {
        'input_ids': test_ids,
        'attention_mask': test_masks
}

train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

NUM_LABELS = 5
DROPOUT_RATE = 0.2
CLASS_WEIGHTS = np.array([0.2, 2, 2, 2, 2], dtype=np.float32)
WEIGHTS_TENSOR = tf.constant(CLASS_WEIGHTS, dtype=tf.float32)



config = BertConfig.from_pretrained('bert-base-multilingual-cased',
                                    num_labels=NUM_LABELS,
                                    hidden_dropout_prob=DROPOUT_RATE,
                                    attention_probs_dropout_prob=DROPOUT_RATE)



model = TFBertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                     config=config)

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

LEARNING_RATE = 2e-5

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=masked_sparse_categorical_crossentropy,
              metrics=[masked_accuracy])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

BATCH_SIZE = 16
EPOCHS = 20

train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
val_dataset = val_dataset.shuffle(1000).batch(BATCH_SIZE)

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stop])

test_dataset = test_dataset.shuffle(1000).batch(BATCH_SIZE)

all_preds = []
all_true_labels = []

for x_batch, y_batch in test_dataset: 
    preds = model.predict(x_batch, verbose=0)
    preds_logits = preds.logits
    pred_ids_batch = np.argmax(preds_logits, axis=2)
    
    true_labels_batch = y_batch.numpy()
    
    all_preds.append(pred_ids_batch)
    all_true_labels.append(true_labels_batch)

pred_ids = np.concatenate(all_preds, axis=0)
test_labels_final = np.concatenate(all_true_labels, axis=0)

def compute_f1_scores(pred_ids, labels, id_to_label):
    true_labels = []
    predicted_labels = []

    for i in range(len(labels)):
        true_tags = []
        pred_tags = []

        for j in range(len(labels[i])):
            label_id = labels[i][j]

            if label_id != -100:
                true_tags.append(id_to_label[label_id]) 
                pred_tags.append(id_to_label[pred_ids[i][j]]) 

        true_labels.append(true_tags)
        predicted_labels.append(pred_tags)

    filtered_true_labels = [l for l in true_labels if l]
    filtered_predicted_labels = [p for p, l in zip(predicted_labels, true_labels) if l]

    if not filtered_true_labels:
        return "No valid labels found in the test set after filtering."

    report = classification_report(filtered_true_labels, filtered_predicted_labels)
    
    return report

f1_report = compute_f1_scores(pred_ids, test_labels_final, id_to_tag)

print("\n--- F1-Score 평가 보고서 ---")
print(f1_report)

save_directory = "D:/Music_Tagging_AI/model"
model.save(save_directory)