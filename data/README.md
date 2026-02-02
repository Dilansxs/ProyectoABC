# Estructura esperada en `data/`

Este proyecto espera el siguiente layout para el dataset original:

```
data/
  raw/                # (opcional) videos sin distribuir
  dataset/            # dataset organizado por persona (UN VIDEO POR CARPETA)
    PERSON_001/
      video1.mp4      # coloque aquí el video con audio para la persona
    PERSON_002/
      video1.mp4

  datasetPros/        # salida del preprocesamiento (body/, audio/)
  features/           # vectores de características para entrenamiento
```

Reglas simples para nombres de archivos (recomendado):
- Use nombres que incluyan la persona: `Angeli_video1.mp4` o `Ivan_take1.mov`.

Para crear las carpetas vacías, use el comando `prepare_dataset` del CLI o cree las carpetas manualmente:

```bash
# Crear carpetas para personas
python main.py --command prepare_dataset --persons Angeli Ivan Mateo
```

Si necesita ayuda, use el comando `help` en la CLI del proyecto (comando `help`).