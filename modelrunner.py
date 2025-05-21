import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('apple_classifier.h5')

img_width, img_height = 150, 150

img_path = '/Users/jayeshpatil/Desktop/data/ro-apple-test.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0

predictions = model.predict(img_array)


class_labels = ['ripe', 'rotten', 'unripe']

predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class = class_labels[predicted_class_index]

print(f'Predicted class: {predicted_class}')
