from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt

class ModelEvaluator:
    """
    Evaluador de modelos SVM.
    """
    
    def evaluate(self, model, features, labels):
        predictions = model.predict(features)
        
        accuracy = accuracy_score(labels, predictions)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        conf_matrix = confusion_matrix(labels, predictions, labels=model.classes_)
        
        # Calcular probabilidades para la curva ROC (One-vs-Rest)
        try:
            decision_function = model.decision_function(features)
            roc_data = self._calculate_roc_curves(labels, decision_function, model.classes_)
        except:
            roc_data = None
        
        return {
            'accuracy': float(accuracy),
            'precision_avg': float(precision_avg),
            'recall_avg': float(recall_avg),
            'f1_avg': float(f1_avg),
            'precision_per_class': {cls: float(p) for cls, p in zip(model.classes_, precision)},
            'recall_per_class': {cls: float(r) for cls, r in zip(model.classes_, recall)},
            'f1_per_class': {cls: float(f) for cls, f in zip(model.classes_, f1)},
            'support_per_class': {cls: int(s) for cls, s in zip(model.classes_, support)},
            'confusion_matrix': conf_matrix.tolist(),
            'classes': list(model.classes_),
            'roc_data': roc_data
        }
    
    def _calculate_roc_curves(self, labels, decision_function, classes):
        """
        Calcula curvas ROC para cada clase (One-vs-Rest).
        """
        try:
            n_classes = len(classes)
            
            # Binarizar las etiquetas
            y_bin = label_binarize(labels, classes=classes)
            if n_classes == 2:
                y_bin = np.column_stack([1 - y_bin, y_bin])
            
            # Asegurar que decision_function es 2D
            if decision_function.ndim == 1:
                decision_function = decision_function.reshape(-1, 1)
            
            # Si hay menos columnas que clases, rellenar con ceros
            if decision_function.shape[1] < n_classes:
                padding = np.zeros((decision_function.shape[0], n_classes - decision_function.shape[1]))
                decision_function = np.hstack([decision_function, padding])
            
            roc_curves = {}
            
            for i, cls in enumerate(classes):
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], decision_function[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_curves[cls] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': float(roc_auc)
                    }
                except Exception as e:
                    # Silenciosamente fallar para esta clase
                    roc_curves[cls] = {
                        'fpr': [],
                        'tpr': [],
                        'auc': 0.0
                    }
            
            return roc_curves
        except Exception as e:
            print(f"⚠️  Error calculando curvas ROC: {e}")
            return None
    
    def plot_roc_curves(self, metrics, save_path=None):
        """
        Grafica las curvas ROC para todas las clases.
        
        Args:
            metrics: Diccionario de métricas de evaluate()
            save_path: Ruta para guardar la imagen (opcional)
        """
        if not metrics.get('roc_data'):
            print("❌ No hay datos ROC disponibles")
            return
        
        plt.figure(figsize=(12, 8))
        
        for cls, roc_info in metrics['roc_data'].items():
            fpr = roc_info['fpr']
            tpr = roc_info['tpr']
            auc_score = roc_info['auc']
            plt.plot(fpr, tpr, label=f'{cls} (AUC = {auc_score:.3f})', linewidth=2)
        
        # Línea diagonal (clasificador aleatorio)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Curvas ROC - One-vs-Rest', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Curva ROC guardada: {save_path}")
        
        plt.show()
    
    def generate_report(self, metrics):
        report = "\n" + "="*60 + "\n"
        report += "   REPORTE DE EVALUACIÓN\n"
        report += "="*60 + "\n"
        
        report += f"\nMétricas Globales:\n"
        report += f"  • Accuracy:  {metrics['accuracy']:.4f}\n"
        report += f"  • Precision: {metrics['precision_avg']:.4f}\n"
        report += f"  • Recall:    {metrics['recall_avg']:.4f}\n"
        report += f"  • F1-Score:  {metrics['f1_avg']:.4f}\n"
        
        report += f"\n{'─'*60}\n"
        report += "Métricas por Clase:\n"
        report += f"{'─'*60}\n"
        
        for cls in metrics['classes']:
            report += f"\n{cls}:\n"
            report += f"  Precision: {metrics['precision_per_class'][cls]:.4f}\n"
            report += f"  Recall:    {metrics['recall_per_class'][cls]:.4f}\n"
            report += f"  F1-Score:  {metrics['f1_per_class'][cls]:.4f}\n"
            report += f"  Muestras:  {metrics['support_per_class'][cls]}\n"
        
        # Agregar AUC-ROC si está disponible
        if metrics.get('roc_data'):
            report += f"\n{'─'*60}\n"
            report += "Áreas bajo la Curva ROC (AUC):\n"
            report += f"{'─'*60}\n"
            for cls, roc_info in metrics['roc_data'].items():
                report += f"  {cls}: {roc_info['auc']:.4f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report