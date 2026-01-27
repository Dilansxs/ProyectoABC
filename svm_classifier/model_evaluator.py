from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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
            'classes': list(model.classes_)
        }
    
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
        
        report += "\n" + "="*60 + "\n"
        
        return report