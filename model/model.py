import os
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/cnn'))

    def predict(self, X):
        
        # Insert your preprocessing here
        X = preprocess_input(X)

        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out