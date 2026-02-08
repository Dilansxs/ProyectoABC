"""
Sistema de reidentificación de personas - Módulo principal.

Este programa implementa un sistema integrado de reconocimiento y reidentificación
de personas basado en características corporales.

Soporta tres tipos de preprocesadores:
- BLP: Binary Local Patterns
- HSH: Histogram of Spatial Hue  
- LBP: Local Binary Patterns

Rutas fijas del sistema:
- Dataset: data/dataset/
- Dataset procesado: data/datasetPros/
- Modelos: models/
"""


import sys
import os
from interface.command_handler import CommandHandler


def main():
    """
    Función principal que inicia la interfaz de línea de comandos.
    """
    handler = CommandHandler()
    show_welcome_message(handler)
    run_interactive_mode(handler)


def show_welcome_message(handler: CommandHandler):
    """
    Muestra un mensaje de bienvenida con información del sistema.
    """
    print("\n" + "=" * 60)
    print("   SISTEMA DE REIDENTIFICACIÓN DE PERSONAS")
    print("   Basado en características corporales")
    print("=" * 60)
    print("\nRUTAS DEL SISTEMA (fijas):")
    print(f"   • Dataset origen:     {handler.DATASET_PATH}")
    print(f"   • Dataset procesado:  {handler.DATASET_PROCESSED_PATH}")
    print(f"   • Modelos:            {handler.MODELS_PATH}")
    print(f"   • Características:    {handler.FEATURES_PATH}")
    print("\nPREPROCESADORES DISPONIBLES:")
    print("HSV - Histograma de Hue, Saturation, Value")
    print("\n" + "-" * 60)
    print("\nNOTA: El pipeline ahora procesa solo 'body' y 'audio' por persona.")


def show_menu(handler: CommandHandler) -> str:
    """
    Muestra el menú principal interactivo.
    
    Returns:
        str: Opción seleccionada por el usuario.
    """
    print(f"\n┌─────────────────────────────────────────────────┐")
    print(f"│            MENÚ PRINCIPAL                         │")
    print(f"│  Preprocesador actual: {handler.preprocessor_type:3}                    │")
    print(f"├───────────────────────────────────────────────── ┤")
    print(f"│  1. Preprocesar dataset                          │")
    print(f"│  2. Detectar cuerpos/rostros                     │")
    print(f"│  3. Extraer características                      │")
    print(f"│  3b. 🎵 Augmentar dataset de audios              │")
    print(f"│  4. Entrenar modelo SVM                          │")
    print(f"│  5. Ver evaluación                               │")
    print(f"│  ─────────────────────────────────────────────   │")
    print(f"│  6. 🚀 AUTOMÁTICO (ejecutar todo)               │")
    print(f"│  ─────────────────────────────────────────────   │")
    print(f"│  7. Ver estado del sistema                       │")
    print(f"│  8. Ayuda                                        │")
    print(f"│  0. Salir                                        │")
    print(f"└─────────────────────────────────────────────────┘")
    
    return input("\nSeleccione una opción: ").strip()


def print_result(result: dict):
    """
    Imprime el resultado de una operación de forma legible.
    """
    if not result:
        return
    
    for key, value in result.items():
        if key in ['success', 'exit']:
            continue
        if isinstance(value, dict):
            print(f"\n  {key}:")
            for k, v in value.items():
                print(f"    • {k}: {v}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {', '.join(map(str, value))}")
        else:
            print(f"  {key}: {value}")


def run_interactive_mode(handler: CommandHandler):
    """
    Ejecuta el sistema en modo interactivo.
    """
    while True:
        option = show_menu(handler)
        
        try:
            if option == '1':
                # Preprocesar
                print("\n" + "─" * 50)
                result = handler.preprocess()
                print_result(result)
            
            elif option == '2':
                # Detectar
                print("\n" + "─" * 50)
                result = handler.detect()
                print_result(result)
            
            elif option == '3':
                # Extraer características
                print("\n" + "─" * 50)
                result = handler.extract_features()
                print_result(result)
            
            elif option == '3b':
                # Augmentar audios
                print("\n" + "─" * 50)
                result = handler.augment_audio_dataset()
                print_result(result)
            
            elif option == '4':
                # Entrenar SVM
                print("\n" + "─" * 50)
                modality = input("¿Entrenar modelo para (1) imagen, (2) audio o (3) FUSIÓN? [1/2/3]: ").strip()
                if modality == '2' or modality.lower() == 'audio':
                    result = handler.train_svm(modality='audio')
                elif modality == '3' or modality.lower() == 'fusion':
                    result = handler.train_svm(modality='fusion')
                else:
                    result = handler.train_svm(modality='image')
                print_result(result)
            
            elif option == '5':
                # Ver evaluación
                print("\n" + "─" * 50)
                result = handler.evaluate()
                print_result(result)
                
                if result.get('success') and 'metrics' in result:
                    print("\n  ╔═══════════════════════════════════════╗")
                    print("  ║       MÉTRICAS DE EVALUACIÓN          ║")
                    print("  ╠═══════════════════════════════════════╣")
                    metrics = result['metrics']
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                print(f"  ║  {metric:20} : {value:>10.4f}  ║")
                            else:
                                print(f"  ║  {metric:20} : {str(value):>10}  ║")
                    print("  ╚═══════════════════════════════════════╝")
            
            elif option == '6':
                # Automático
                confirm = input("\n¿Ejecutar pipeline completo automáticamente? (s/n): ").strip().lower()
                if confirm == 's':
                    result = handler.run_automatic()
                    
                    # Mostrar resumen final
                    if result.get('success'):
                        print("\n✅ Pipeline completado exitosamente")
                    else:
                        print("\n⚠️ Pipeline completado con algunos errores")
                else:
                    print("Operación cancelada.")
            
            
            elif option == '7':
                # Ver estado
                status = handler.get_status()
                print("\n" + "─" * 50)
                print("  ESTADO DEL SISTEMA")
                print("─" * 50)
                print(f"\n  Preprocesador: {status['preprocessor_type']}")
                print("\n  Rutas:")
                for name, path in status['paths'].items():
                    print(f"    • {name}: {path}")
                print("\n  Estado de componentes:")
                for component, ready in status['status'].items():
                    icon = "✓" if ready else "✗"
                    print(f"    {icon} {component}")
            
            elif option == '8':
                # Ayuda
                print(handler.help())
            
            elif option == '0':
                # Salir
                result = handler.exit_system()
                print(f"\n{result.get('message')}")
                break
            
            else:
                print("\n✗ Opción no válida. Intente de nuevo.")
        
        except KeyboardInterrupt:
            print("\n\nOperación cancelada por el usuario.")
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("   Gracias por usar el sistema. ¡Hasta pronto!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
