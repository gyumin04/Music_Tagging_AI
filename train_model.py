import tensorflow as tf
from transformers import TFBertForTokenClassification, BertTokenizerFast, BertConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
import numpy as np
import keras

input_file = 'music_data.CSV'
music_df = pd.read_csv(input_file)

data = []
for idx in music_df.index:
    text_words = (music_df.loc[idx, 'title'] + " " + music_df.loc[idx, 'channel_name']).split()
    
    labels = ['O'] * len(text_words)
    
    original_song_name_words = music_df.loc[idx, '원곡명'].split()
    artist_words = music_df.loc[idx, '아티스트'].split()

    def assign_bio_tags(words, entity_name_words, tag_prefix):
        if not entity_name_words: return

        try:
            start_index = -1
            for i in range(len(words) - len(entity_name_words) + 1):
                if words[i:i + len(entity_name_words)] == entity_name_words:
                    start_index = i
                    break
            
            if start_index != -1:
                for i in range(len(entity_name_words)):
                    if i == 0:
                        labels[start_index + i] = f'B-{tag_prefix}'
                    else:
                        labels[start_index + i] = f'I-{tag_prefix}'
        except ValueError:
            pass

    assign_bio_tags(text_words, original_song_name_words, 'SONG')
    assign_bio_tags(text_words, artist_words, 'ARTIST')

    data.append({
        "words": text_words,
        "tags": labels
    })

tags = ['O', 'B-SONG', 'I-SONG', 'B-ARTIST', 'I-ARTIST']
tag_to_id = {tag: i for i, tag in enumerate(tags)}
id_to_tag = {i: tag for tag, i in tag_to_id.items()}
num_labels = len(tags)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

def tokenize_and_align_labels(examples):
    """
    WordPiece 토큰화 후, 레이블을 정렬하고 서브워드 토큰에 -100을 할당
    """
    tokenized_inputs = tokenizer(examples["words"], 
                                 truncation=True, 
                                 is_split_into_words=True,
                                 padding="max_length", 
                                 max_length=128)
    
    labels = []
    
    for i, word_tags in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        previous_word_idx = None
        label_ids = []

        tag_ids = [tag_to_id[tag] for tag in word_tags]
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx == previous_word_idx:
                label_ids.append(-100) 
            else:
                if word_idx < len(tag_ids):
                    label_ids.append(tag_ids[word_idx])
                else:
                    label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

words_list = [item["words"] for item in data]
tags_list = [item["tags"] for item in data]

processed_data = tokenize_and_align_labels({
    "words": words_list,
    "tags": tags_list
})

input_ids_np = np.array(processed_data['input_ids'])
attention_mask_np = np.array(processed_data['attention_mask'])
labels_np = np.array(processed_data['labels']) 

RANDOM_SEED = 42
train_ids, temp_ids, train_masks, temp_masks, train_labels, temp_labels = train_test_split(
    input_ids_np, attention_mask_np, labels_np,
    test_size=0.2, 
    random_state=RANDOM_SEED
)

val_ids, test_ids, val_masks, test_masks, val_labels, test_labels = train_test_split(
    temp_ids, temp_masks, temp_labels,
    test_size=0.5, 
    random_state=RANDOM_SEED
)

BATCH_SIZE = 16 

train_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': train_ids,
    'attention_mask': train_masks
}, train_labels)).shuffle(1000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': val_ids,
    'attention_mask': val_masks
}, val_labels)).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': test_ids,
    'attention_mask': test_masks
}, test_labels)).batch(BATCH_SIZE)


DROPOUT_RATE = 0.2
CLASS_WEIGHTS = np.array([0.05, 3.0, 3.0, 3.0, 3.0], dtype=np.float32) 
WEIGHTS_TENSOR = tf.constant(CLASS_WEIGHTS, dtype=tf.float32)

config = BertConfig.from_pretrained(
    'bert-base-multilingual-cased', 
    num_labels=num_labels,
    hidden_dropout_prob=DROPOUT_RATE,
    attention_probs_dropout_prob=DROPOUT_RATE
)

model = TFBertForTokenClassification.from_pretrained(
    'bert-base-multilingual-cased',
    config=config
)

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true_int = tf.cast(y_true, dtype=tf.int64)
    loss_mask = tf.cast(tf.not_equal(y_true_int, -100), dtype=y_pred.dtype)
    y_true_masked = tf.where(tf.equal(y_true_int, -100), tf.constant(0, dtype=y_true_int.dtype), y_true_int)
    
    sample_weights = tf.gather(WEIGHTS_TENSOR, y_true_masked) 
    
    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        reduction=tf.keras.losses.Reduction.NONE 
    )
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

LEARNING_RATE = 1e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=masked_sparse_categorical_crossentropy, 
              metrics=[masked_accuracy])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
EPOCHS = 20 

print("--- 모델 재학습 시작 (Fast Tokenizer 적용 및 레이블 정렬 문제 최종 해결) ---")
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    callbacks=[early_stop])

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

print("\n--- 최종 개체 기반 F1-Score 평가 보고서 (수정된 로직) ---")
print(f1_report)