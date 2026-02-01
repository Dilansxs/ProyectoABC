# Estructura esperada en `data/`

Este proyecto espera el siguiente layout para el dataset original:

```
data/
  raw/                # (opcional) videos sin distribuir
  dataset/            # dataset organizado por persona
    PERSON_001/
      front/          # videos frontales (imágenes/frames/rostros)
      back/           # videos posteriores (audio para MFCC / voz)
    PERSON_002/
      front/
      back/

  datasetPros/        # salida del preprocesamiento (frames, audios procesados, etc.)
  features/           # vectores de características para entrenamiento
```

Reglas simples para nombres de archivos (recomendado):
- Use nombres que incluyan la persona y la vista: `AngeliMicroFrente.mov` o `IvanMicroEspalda.mov`.
- Palabras clave reconocidas (case-insensitive): `frente`, `front`, `espalda`, `back`.

Para crear la estructura vacía y distribuir videos automáticamente desde `data/raw`, use:

```bash
python -m data.prepare_dataset --create Angeli Ivan Mateo
python -m data.prepare_dataset --auto-distribute
```

Si necesita ayuda, puede ejecutar `prepare_dataset` desde el CLI del proyecto (comando `prepare_dataset`).