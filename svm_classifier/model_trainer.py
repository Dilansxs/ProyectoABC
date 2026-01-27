from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .svm_model import SVMModel
from .model_evaluator import ModelEvaluator


class ModelTrainer:
    """
    Entrenador para el modelo SVM.
    """
    
    def __init__(self, train_test_split_ratio=0.8):
        self.train_test_split_ratio = train_test_split_ratio
    
    def train(self, features, labels, validate=True):
        print(f"\n{'='*60}")
        print("ENTRENAMIENTO DEL MODELO SVM")
        print(f"{'='*60}")
        
        model = SVMModel(kernel='rbf', C=1.0, gamma='scale')
        
        if validate:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels,
                test_size=(1 - self.train_test_split_ratio),
                stratify=labels,
                random_state=42
            )
            
            print(f"\nDatos de entrenamiento: {len(X_train)}")
            print(f"Datos de prueba: {len(X_test)}")
            
            train_info = model.train(X_train, y_train)
            
            print(f"\n✓ Entrenamiento completado en {train_info['training_time']:.2f}s")
            print(f"  - Muestras: {train_info['num_samples']}")
            print(f"  - Clases: {train_info['num_classes']}")
            
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            print(f"\nMétricas de evaluación:")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - Precision (promedio): {metrics['precision_avg']:.4f}")
            print(f"  - Recall (promedio): {metrics['recall_avg']:.4f}")
            print(f"  - F1-Score (promedio): {metrics['f1_avg']:.4f}")
            
            return model, metrics
        else:
            train_info = model.train(features, labels)
            
            print(f"\n✓ Entrenamiento completado en {train_info['training_time']:.2f}s")
            print(f"  - Muestras totales: {train_info['num_samples']}")
            print(f"  - Clases: {train_info['num_classes']}")
            
            return model, {'training_info': train_info}
