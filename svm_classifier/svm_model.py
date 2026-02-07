"""
Modelo SVM para clasificación.
"""

import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SVMModel:
    """
    Modelo SVM para clasificación de personas.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, 
                        probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.classes_ = None
    
    def train(self, features, labels):
        import time
        start_time = time.time()
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        
        features_scaled = self.scaler.fit_transform(features)
        
        self.model.fit(features_scaled, encoded_labels)
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'num_samples': len(labels),
            'num_classes': len(self.classes_),
            'classes': list(self.classes_)
        }
    
    def predict(self, features):
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        features_scaled = self.scaler.transform(features)
        predictions_encoded = self.model.predict(features_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_with_confidence(self, features):
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)
        
        predictions_encoded = np.argmax(probabilities, axis=1)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, confidence
    
    def save(self, model_path):
        if not self.is_trained:
            raise ValueError("No se puede guardar un modelo no entrenado")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classes': self.classes_,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        return True
    
    @staticmethod
    def load(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        svm = SVMModel(
            kernel=model_data['kernel'],
            C=model_data['C'],
            gamma=model_data['gamma']
        )
        
        svm.model = model_data['model']
        svm.scaler = model_data['scaler']
        svm.label_encoder = model_data['label_encoder']
        svm.classes_ = model_data['classes']
        svm.is_trained = True
        
        return svm
    
    def get_person_id(self, class_index):
        if self.classes_ is None:
            raise ValueError("El modelo no tiene clases definidas")
        return self.classes_[class_index]


