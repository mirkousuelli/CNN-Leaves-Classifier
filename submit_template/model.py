import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/cnn_1'))

    def predict(self, X):
        
        # Insert your preprocessing here
        X = X / 255.

        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out