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
