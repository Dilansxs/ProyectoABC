# Sistema de Reidentificación de Personas

## Descripción

Sistema integrado de reidentificación de personas basado en visión por computador, centrado en el análisis de apariencia corporal y clasificación mediante SVM para identificar personas en videos.

## Estructura del Proyecto

```
ABCsistema/
├── preprocessing/          # Extracción de fotogramas y data augmentation
│   ├── __init__.py
│   ├── frame_extraction.py
│   ├── data_augmentation.py
│   └── preprocessing_pipeline.py
├── detection/              # Detección de cuerpos
│   ├── __init__.py
│   ├── body_detection.py
│   └── view_classification.py
├── feature_extraction/     # Extracción de características
│   ├── __init__.py
│   ├── body_features.py
│   └── feature_vector.py
├── svm_classifier/         # Clasificador SVM
│   ├── __init__.py
│   ├── svm_model.py
│   ├── model_trainer.py
│   └── model_evaluator.py
├── interface/              # Interfaz de línea de comandos
│   ├── __init__.py
│   └── command_handler.py
├── data/                   # Directorio para datasets
├── models/                 # Directorio para modelos entrenados
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── .gitignore             # Configuración de Git
└── README.md              # Este archivo
```

## Características Principales

### 1. Preprocesamiento
- Extracción de fotogramas a ~10 fps desde videos
- Data augmentation (rotaciones, brillo, reflexiones)
- Segmentación automática de cuerpos

### 2. Detección
- Detección de cuerpos completos
- Recorte y organización de regiones detectadas

### 3. Extracción de Características
- Extracción de características corporales
- Vectores normalizados para clasificación
- Cálculo de distancias y similitudes

### 4. Clasificación SVM
- Entrenamiento de modelo SVM con características corporales
- Validación cruzada y evaluación
- Ajuste de hiperparámetros
- Predicción con confianza

### 5. Interfaz CLI
- Comando de preprocesamiento
- Entrenamiento de modelos
- Evaluación de desempeño
- Identificación de personas

## Requisitos

- Python 3.8 o superior
- Librerías especificadas en `requirements.txt`

## Instalación

1. **Clonar el repositorio**
```bash
git clone <url-repo>
cd ABCsistema
```

2. **Crear entorno virtual**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## Uso

### Modo Interactivo
```bash
python main.py
```

### Comandos Disponibles

#### 1. Preprocesamiento
```
> preprocess <dataset_path> <output_path> [--fps 10] [--augmentation]
```
Extrae fotogramas de videos y aplica data augmentation.

#### 2. Entrenar Modelo SVM
```
> train_svm <dataset_path> <model_output_path> [--augmented]
```
Entrena el clasificador SVM para reidentificación corporal.

#### 4. Evaluar Modelo
```
> evaluate <model_path> <test_dataset_path>
```
Evalúa el desempeño del modelo con dataset de prueba.

#### 5. Identificar Persona
```
> identify <image_path> <model_path> [--threshold 0.7]
```
Identifica a una persona en una imagen.

## Estructura de Datos

### Dataset Original
```
dataset/
├── persona1/
│   └── video1.mp4
├── persona2/
│   └── video2.mp4
└── ...
```

### Dataset Procesado
```
datasetPros/
├── persona1/
│   └── body/
│       ├── img1.jpg
│       └── ...
├── persona2/
│   └── ...
└── ...
```

## Flujo de Procesamiento

1. **Preprocesamiento**: Extrae fotogramas → Aplica augmentation
2. **Detección**: Detecta cuerpos
3. **Extracción**: Genera características corporales
4. **Entrenamiento**: Entrena modelo SVM con características
5. **Evaluación**: Valida performance con métricas
6. **Predicción**: Identifica personas en nuevas imágenes

## Modelos y Versiones

- **Detección de Cuerpos**: YOLO / Faster R-CNN
- **Características Corporales**: ResNet50 / EfficientNet
- **Clasificación**: SVM (kernel RBF por defecto)

## Métricas de Evaluación

- Exactitud (Accuracy)
- Precisión y Recall por clase
- F1-Score
- Matriz de Confusión
- Curva ROC y AUC

## Contribuciones

Se aceptan contribuciones. Por favor:
1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit cambios
4. Push a la rama
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo licencia MIT.

## Contacto

Para preguntas o sugerencias, contactar al equipo de desarrollo.

---

**Última actualización**: 23 de enero de 2026
