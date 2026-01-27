class ViewClassification:
    """
    Clasificador de vista frontal/posterior (placeholder).
    
    En el sistema actual, la clasificación de vista se realiza mediante
    la estructura de carpetas (front/back), no mediante un modelo ML.
    """
    
    def __init__(self, model='cnn', confidence_threshold=0.7):
        self.model = model
        self.confidence_threshold = confidence_threshold
        print("[INFO] ViewClassification es un placeholder.")
    
    def classify(self, image):
        """Placeholder: retorna 'front' por defecto."""
        return ('front', 0.5)
    
    def classify_batch(self, images):
        """Placeholder: clasifica batch."""
        return [self.classify(img) for img in images]
    
    def separate_by_view(self, images):
        """Placeholder: separa por vista."""
        # Por simplicidad, retorna todas en 'front'
        return {
            'front': images,
            'back': []
        }
    
    def process_and_save(self, images, output_path):
        """Placeholder: procesa y guarda."""
        import os
        os.makedirs(os.path.join(output_path, 'front'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'back'), exist_ok=True)
        
        return {
            'front_count': len(images),
            'back_count': 0,
            'total': len(images)
        }