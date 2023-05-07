import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from sklearn.calibration import calibration_curve
import os, requests, zipfile
!pip install GitPython
from git import Repo

print('Ready to go!')
batch_size = 64
photo_height = 227
photo_width = 227

filepath = 'temp_concrete_crack'

train_path = r'{temp_concrete_crack}/data/concrete_images/train'.format(temp_concrete_crack=filepath)
val_path = r'{temp_concrete_crack}/data/concrete_images/val'.format(temp_concrete_crack=filepath)
test_path = r'{temp_concrete_crack}/data/concrete_images/test'.format(temp_concrete_crack=filepath)

train = tf.keras.preprocessing.image_dataset_from_directory(
  train_path,
  validation_split=None,
  subset=None,
  seed=42,
  image_size=(photo_height, photo_width),
  batch_size=batch_size)

val = tf.keras.preprocessing.image_dataset_from_directory(
  val_path,
  validation_split=None,
  subset=None,
  seed=42,
  image_size=(photo_height, photo_width),
  batch_size=batch_size)

classes = train.class_names
print('classes: ' +str(classes))
plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classes[labels[i]])
    plt.axis("off")
    
    
plt.show()
tf.keras.backend.clear_session()

num_classes = 2

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
    
    
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

print('compiled!')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)

history = model.fit(
  train,
  validation_data=val,
  epochs=30,
    callbacks=[model_checkpoint, early_stopping]
)

h = pd.DataFrame(history.history)
h['epoch'] = h.index + 1


plt.subplot(2, 1, 1)
plt.plot(h['epoch'], h['accuracy'], h['epoch'], h['val_accuracy'])
plt.title('Model History')
plt.ylabel('accuracy')
plt.grid(True)
plt.legend(('train accuracy', 'val accuracy'),
           loc='lower right')
ax1 = plt.gca()
ax1.set_xticks(range(1, max(h['epoch'])+1))


plt.subplot(2, 1, 2)
plt.plot(h['epoch'], h['loss'], h['epoch'], h['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend(('train loss', 'val loss'),
           loc='upper right')
ax2 = plt.gca()
ax2.set_xticks(range(1, max(h['epoch'])+1))

plt.show()


loaded_model = tf.keras.models.load_model(filepath)
results = pd.DataFrame(columns=['predicted', 'actual', 'probability', 'file', 'raw_logits'])

at = 0
for label in classes: 
    path = test_path+'/'+label
    print(path)
    for file in glob.iglob(path + '/*'):
        img = keras.preprocessing.image.load_img(
        file, target_size=(photo_height, photo_width)
        )
        
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        results = results.append({'predicted': classes[np.argmax(score)], 'actual': label, 
                                  'probability': 100 * np.max(score), 'file': file,
                                 'raw_logits': score}, ignore_index=True)
        at +=1
        if at % 200 == 0:
            print(str(round(((at / 4000) * 100), 1))+'%')

        

score = accuracy_score(results['actual'], results['predicted'])
print('overall test accuracy: '+str(score))
for i in classes:
   mistakes = results.loc[(results['predicted'] != i) & (results['actual'] == i)]
   files = mistakes['file'].tail(10).values
   fig,ax = plt.subplots(2,int(len(files) / 2) + (int(len(files) % 2 > 0)))
   fig.set_size_inches(11, 5)
   prob = mistakes['probability'].tail(10).values
   act = mistakes['actual'].tail(10).values
   pred = mistakes['predicted'].tail(10).values
   fig.suptitle('True Label: '+str(i) + '\n' + ' ', weight='bold')
   
   for i in range(0, len(files)):
       with open(files[i],'rb') as f:
           image=Image.open(f)
           ax[i%2][i//2].imshow(image)
   ax_list = fig.axes
   for ax in range(len(files)):

      ax_list[ax].set_title('predicted: '+pred[ax]+ "\n" + ' prob: '+str(round(prob[ax], 1)), fontsize=8)
   for ax in range(len(ax_list)):
      ax_list[ax].set_xticks([])
      ax_list[ax].set_yticks([])
   
plt.show()
pd.set_option('precision', 1)
errors = results.loc[(results['predicted'] != results['actual'])]
for row in range(len(errors)):
    print(errors.iloc[row, 0:3].values)
print('False positive mean prob: '+str(np.mean(errors['probability'].loc[(results['predicted'] != 'no_crack')])))
print('False negative mean prob: '+str(np.mean(errors['probability'].loc[(results['predicted'] == 'no_crack')])))
report = classification_report(results['actual'], results['predicted'], output_dict=True)

plt.figure(figsize=(8,8))
plt.subplot(211)
barWidth = 0.3
 
bars1 = [report['crack']['precision'], report['no_crack']['precision']]
bars2 = [report['crack']['recall'], report['no_crack']['recall']]
 
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = np.append(r1, r2)
plt.bar(r1, bars1, width = barWidth, color = 'cornflowerblue', edgecolor = 'black', capsize=7, label='precision')
plt.bar(r2, bars2, width = barWidth, color = 'olivedrab', edgecolor = 'black', capsize=7, label='recall')

plt.text(-.1, 1.05, r'CV Accuracy: '+str(round(report['accuracy']*100, 1))+'%', fontsize=9, fontweight='bold')
ax1 = plt.gca()
ax1.set_xticks([r + (barWidth / 2) for r in range(len(bars1))])
ax1.set_xticklabels(['Crack', 'No Crack'])
plt.ylabel('accuracy')
plt.ylim([0.5, 1.1])

label = [bars1[0], bars2[0], bars1[1], bars2[1]]
plt.title('Classification Report')
plt.legend()

plt.hlines(0.95, xmin=min(r3), xmax=max(r3), color='black', linestyles='dashed', label='95%', linewidth=0.6)
plt.hlines(0.98, xmin=min(r3), xmax=max(r3), color='black', linestyles='dashed', label='98%', linewidth=0.6)
plt.text(0.57, 0.995, '98%', ha='left', va='center')
plt.text(0.57, 0.93, '95%', ha='left', va='center')




plt.subplot(212)

predicted_dist = [(results['predicted'] == 'crack').sum() / 4000, (results['predicted'] == 'no_crack').sum() / 4000]

actual_dist = [(results['actual'] == 'crack').sum() / 4000, (results['actual'] == 'no_crack').sum() / 4000]

barWidth = 0.3
 
bars1 = predicted_dist
bars2 = actual_dist
 
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars1, width = barWidth, color = 'tab:blue', edgecolor = 'black', capsize=7, label='predicted')
plt.bar(r2, bars2, width = barWidth, color = 'tab:orange', edgecolor = 'black', capsize=7, label='actual')
ax2 = plt.gca()
ax2.set_xticks([r + (barWidth / 2) for r in range(len(bars1))])
ax2.set_xticklabels(['Crack', 'No Crack'])
plt.ylabel('pct of dataset')
plt.legend()
plt.title('Class Distributions')


plt.show()
crack = results.loc[(results['predicted'] == 'crack')]
no_crack = results.loc[(results['predicted'] == 'no_crack')]
probs = pd.DataFrame({'Crack': crack['probability'], 'No Crack': no_crack['probability']})

ax = probs.plot.kde(ind=[i for i in np.linspace(start = 60, stop = 110, num = 1000)], 
                    title='Probability Distribution', xlabel='Probability')
prob_no_crack = []
for i in range(len(results)):
    logits = results['raw_logits'][i]
    prob_no_crack.append(logits[1])

prob_true_binary, prob_pred_binary = calibration_curve(
        results['actual'].map({'crack': 0, 'no_crack': 1}), 
        prob_no_crack, n_bins=4, normalize=False)


fig = plt.figure()
ax = plt.gca()

plt.plot([0, 1], [0, 1], color='tab:red', linestyle=":", label="Calibrated Model")
plt.plot(prob_pred_binary, prob_true_binary, label='Classifier', color="tab:green")

plt.ylabel('positive rate')
plt.xlabel('predicted value')

plt.legend()
plt.yticks()
plt.grid(True)
plt.tight_layout()

plt.show()
thresholds = [i for i in np.linspace(0.0, 100.0, num=1000)]
prob_tuning = pd.DataFrame(columns=[
    'threshold', 'false_negatives', 'false_refrains', 'under_threshold'])

for threshold in thresholds:
    false_neg = 0
    false_refrain = 0
    under_threshold = 0
    for z in range(len(results)):
        true = results['actual'][z]
        pred = results['predicted'][z]
        prob = results['probability'][z]
        if prob <= threshold:
            under_threshold += 1
        if prob <= threshold and pred == true:
            false_refrain += 1
        if prob > threshold and pred == 'no_crack' and true == 'crack':
            false_neg += 1
    prob_tuning = prob_tuning.append(
       {'threshold': threshold, 'false_negatives': false_neg, 'false_refrains': false_refrain, 
        'under_threshold': under_threshold}, ignore_index=True) 


            
            
fig, ax1 = plt.subplots()

color = 'tab:orange'
ax1.set_xlabel('probability threshold (%)')
ax1.set_ylabel('false negatives', color=color, fontweight='bold')
ax1.plot(prob_tuning['threshold'], prob_tuning['false_negatives'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlim(50, 110)
ax1.grid(True)
ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('false refrains', color=color, fontweight='bold')  
ax2.plot(prob_tuning['threshold'], prob_tuning['false_refrains'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')

fig.suptitle('Finding an Appropriate Probability Threshold')
plt.show()    

tf.keras.backend.clear_session()
from git import rmtree


rmtree('./'+filepath)
