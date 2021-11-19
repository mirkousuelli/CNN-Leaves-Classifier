import os
import tensorflow as tf

tfk = tf.keras

def f1_score(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = tfk.backend.sum(tfk.backend.round(tfk.backend.clip(y_true * y_pred, 0, 1)))
        Positives = tfk.backend.sum(tfk.backend.round(tfk.backend.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+tfk.backend.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = tfk.backend.sum(tfk.backend.round(tfk.backend.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = tfk.backend.sum(tfk.backend.round(tfk.backend.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+tfk.backend.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+tfk.backend.epsilon()))

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'), custom_objects={'f1_score': f1_score})

    def predict(self, X):
        
        # Insert your preprocessing here

        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out
