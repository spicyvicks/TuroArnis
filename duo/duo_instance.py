import joblib
import tensorflow as tf
import numpy as np
import pandas as pd

class ArnisClassifiers:
    def __init__(self, tf_model_path, rf_model_path, encoder_path):
        print("Loading models...")
        self.tf_model = tf.keras.models.load_model(tf_model_path)
        self.rf_classifier = joblib.load(rf_model_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Hybrid model ready.")

    def predict(self, live_features_df):
        
        #tensorFlow 
        tf_pred_proba = self.tf_model.predict(live_features_df, verbose=0)[0]
        tf_pred_index = np.argmax(tf_pred_proba)
        tf_confidence = tf_pred_proba[tf_pred_index]
        
        #random Forest 
        rf_pred_proba = self.rf_classifier.predict_proba(live_features_df)[0]
        rf_pred_index = np.argmax(rf_pred_proba)
        rf_confidence = rf_pred_proba[rf_pred_index]

        combined_proba = (tf_pred_proba + rf_pred_proba) / 2.0
        final_index = np.argmax(combined_proba)
        final_confidence = combined_proba[final_index]

        final_class = self.label_encoder.inverse_transform([final_index])[0]

        return {
            'predicted_class': final_class,
            'confidence': final_confidence
        }