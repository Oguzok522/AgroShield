import os
import tensorflow as tf
from pathlib import Path

dataset_dir = "dataset"
removed_count = 0


for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        file_path = str(Path(root) / file)


        if file.startswith('.'):
            os.remove(file_path)
            removed_count += 1
            continue

        try:

            image_bytes = tf.io.read_file(file_path)

            _ = tf.image.decode_image(image_bytes, expand_animations=False)
        except Exception as e:
            print(f"🗑 Уничтожен битый файл-самозванец: {file}")
            os.remove(file_path)