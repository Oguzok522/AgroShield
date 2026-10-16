import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

modelp = "agroshield_v2.keras"
logsp = "learning_v2.csv"
datasetp = "dataset"
imagesh = (224,224)
batch = 16
print("1")
train = tf.keras.utils.image_dataset_from_directory(
    datasetp,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=imagesh,
    batch_size=batch
)
validation = tf.keras.utils.image_dataset_from_directory(
    datasetp,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=imagesh,
    batch_size=batch
)
class_names = train.class_names
num_classes = len(class_names)
print(f"Найдено {num_classes} классов: {class_names}")
print(2)
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess(inputs)
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base.trainable = False
x = base(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
out = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
weights = {
    0: 6.58,
    1: 1.0,
    2: 1.0,

}

print("3")
logi = model.fit(train, validation_data=validation, epochs=10,class_weight=weights)
model.save(modelp)

df = pd.DataFrame(logi.history)
df.to_csv(logsp, index=False)

