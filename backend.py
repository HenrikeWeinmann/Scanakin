# Backend imports
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score
import random

'''
Das Datenset zum Training kann auf folgender Internetseite heruntergeladen werden:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
'''

# seeding
tf.random.set_seed(8)
np.random.seed(8)
random.seed(8)

# 0 for benign, 1 for malignant

klassifizierungen = ["benign", "malignant"]


# generate CSV metadata file to read img paths and labels from harvard metadata file
def create_csv(meta_path, data_path,name):
    with open(meta_path) as csvdatei:
        csv_reader_object = csv.reader(csvdatei, delimiter=',')
        i = 0
        df = pd.DataFrame(columns=["filepath", "label"])
        for row in csv_reader_object:
            if row[1] == 'image_id':
                continue
            df.loc[i] = [data_path + str(row[1])+'.jpg', benign_or_malign(row[2])]
            i += 1

        output_file = name
        print("Saving", output_file)
        df.to_csv(output_file)


def benign_or_malign(tag):
    result = 1
    if tag == 'bkl' or tag == 'df':
        result = 0
    return result

# uncomment to create new csv
'''
path_to_training_set = 'hvd-dataset/train_dataset/'
path_to_training_set_metadata = 'hvd-dataset/train_metadata.csv'
create_csv(path_to_training_set_metadata,path_to_training_set,"hvd-dataset/new_train_metadata.csv")

path_to_training_set2 = 'hvd-dataset/valid_dataset/'
path_to_training_set_metadata2 = 'hvd-dataset/valid_metadata.csv'
create_csv(path_to_training_set_metadata2, path_to_training_set2, "hvd-dataset/new_valid_metadata.csv")
'''

# loading data
train_metadata_filename = "hvd-dataset/new_train_metadata_04.csv"
valid_metadata_filename = "hvd-dataset/new_valid_metadata.csv"

# load CSV files as DataFrames
train = pd.read_csv(train_metadata_filename)
valid = pd.read_csv(valid_metadata_filename)
length_samples_train = len(train)
length_samples_valid = len(valid)
print("# Training samples:", length_samples_train)
print("# Validation samples:", length_samples_valid)


# reshape data to tensor
def decode_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  return tf.image.resize(image, [299, 299])


# create data set from tf slices
train_ds = tf.data.Dataset.from_tensor_slices((train["filepath"], train["label"]))
valid_ds = tf.data.Dataset.from_tensor_slices((valid["filepath"], valid["label"]))


def read_path(filepath, label):
  img = tf.io.read_file(filepath)  # load the raw data from the file as a string
  img = decode_image(img)
  return img, label


valid_ds = valid_ds.map(read_path)
train_ds = train_ds.map(read_path)
# test_ds = test_ds

for image, label in train_ds.take(1):
    print("Image shape:", image.shape)
    print("Label:", label.numpy())

# training parameters
batch_size = 64  # or 32
optimizer = "rmsprop"


def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  # reorder data set
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # Repeat forever
  ds = ds.repeat()
  # split to batches
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds


valid_ds = prepare_for_training(valid_ds, batch_size=batch_size, cache="valid-cached-data")
train_ds = prepare_for_training(train_ds, batch_size=batch_size, cache="train-cached-data")


# building the model
# InceptionV3 model & pre-trained weights

module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

m = tf.keras.Sequential([
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

m.build([None, 299, 299, 3])
m.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
m.summary()
'''

model_name = f"benign-vs-malignant_{batch_size}_{optimizer}"
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs", model_name))
# saves model checkpoint whenever we reach better weights
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint( model_name + "_{val_loss:.3f}.h5", save_best_only=True, verbose=1)

history = m.fit(train_ds, validation_data=valid_ds,
                steps_per_epoch=n_training_samples // batch_size,
                validation_steps=n_validation_samples // batch_size, verbose=1, epochs=100,
                callbacks=[tensorboard, modelcheckpoint])
'''

# model  evaluation
# load testing set
test_metadata_filename = "hvd-dataset/test_metadata.csv"
df_test = pd.read_csv(test_metadata_filename)
n_testing_samples = len(df_test)
print("Number of testing samples:", n_testing_samples)
test_ds = tf.data.Dataset.from_tensor_slices((df_test["filepath"], df_test["label"]))

def prepare_for_testing(ds, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  return ds


test_ds = test_ds.map(read_path)
test_ds = prepare_for_testing(test_ds, cache="test-cached-data")

# convert testing set to numpy array
y_test = np.zeros((n_testing_samples,))
X_test = np.zeros((n_testing_samples, 299, 299, 3))
for i, (img, label) in enumerate(test_ds.take(n_testing_samples)):
  # print(img.shape, label.shape)
  X_test[i] = img
  y_test[i] = label.numpy()

print("X_test.shape:", X_test.shape)
print("Y_test.shape:", y_test.shape)

# load the weights with the least loss
m.load_weights("benign-vs-malignant_64_rmsprop_0.430.h5")  #430 426

loss, accuracy = m.evaluate(X_test, y_test, verbose=0)
print("Loss:", loss, "  Accuracy:", accuracy)

threshold = 0.23
# get predictions with 20% threshold
# which means if the model is 20% sure or more that is malignant,
# it's assigned as malignant, otherwise it's benign


def get_predictions(threshold=None):
  y_pred = m.predict(X_test)
  if not threshold:
    threshold = 0.45
  result = np.zeros((n_testing_samples,))
  for i in range(n_testing_samples):
    # test melanoma probability
    if y_pred[i][0] >= threshold:
      result[i] = 1
    # else, it's 0 (benign)
  return result


y_pred = get_predictions(threshold)
print('y_pred.shape:', y_pred.shape)
y_val_cat_prob = m.predict(X_test).ravel()
FPR, TPR, thresholds = roc_curve(y_test, y_val_cat_prob)
plt.plot(FPR,TPR)
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
auc_keras = auc(FPR, TPR)
print(auc_keras)

def plot_roc_auc(y_true, y_pred):
    """
    This function plots the ROC curves and provides the scores.
    """
    # prepare for figure
    plt.figure()
    fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    # print score
    print('fpr'+ str(fpr) + 'tpr' + str(tpr))
    print(f"ROC AUC: {roc_auc:.3f}")
    plt.plot(fpr, tpr, color="blue", lw=2,
                label='ROC curve (area = {f:.2f})'.format(d=1, f=roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()


plot_roc_auc(y_test, y_pred)


def actual_vs_predicted(y_test, y_pred):
  cmn = confusion_matrix(y_test, y_pred)
  # Normalise
  cmn = cmn.astype('float') / cmn.sum(axis=1)[:, np.newaxis]
  # print it
  print(cmn)
  fig, ax = plt.subplots(figsize=(10,10))
  sns.heatmap(cmn, annot=True, fmt='.2f',
              xticklabels=[f"pred_{c}" for c in klassifizierungen],
              yticklabels=[f"true_{c}" for c in klassifizierungen],
              cmap="Blues"
              )
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.show()


actual_vs_predicted(y_test, y_pred)

sensitivity = sensitivity_score(y_test, y_pred)
specificity = specificity_score(y_test, y_pred)

print("Melanoma Sensitivity:", sensitivity)
print("Melanoma Specificity:", specificity)


def photo_prediction(filepath):
    result = 0

    test_img = tf.io.read_file(filepath)
    test_img = decode_image(test_img)
    test_img = tf.expand_dims(test_img,0)
    test_img_result = m.predict(test_img)
    print("test result: " + str(test_img_result))
    if test_img_result >= 0.45:
        result = 1

    return result

'''
# save model for further use
m.save('model/model_neu')
# Convert the model
#converter = tf.lite.TFLiteConverter.from_saved_model("model/model_neu")  # path to the SavedModel directory
tflite_model = converter.convert()

# Save the tflite-model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
'''
