# 🌱 EstacaScan

Aplicación web para el conteo automático de estacas utilizando Inteligencia Artificial (YOLOv8) directamente en el navegador.

🔗 **Demo en vivo:** [https://estacascan.netlify.app/](https://estacascan.netlify.app/)

## Características

- **Detección Automática:** Cuenta estacas en segundos subiendo una foto o usando la cámara.
- **Funcionamiento Local:** El modelo se ejecuta en tu dispositivo, no requiere internet para procesar las imágenes una vez cargado.
- **Corrección Manual:**
  - Click en una estaca para descartarla.
  - Click en "Agregar" (o Ctrl+Click) para añadir estacas faltantes.
- **Modo Revisión:** Interfaz para verificar detecciones dudosas.
- **Zoom Suave:** Inspecciona la imagen con detalle.

## Cómo Usar

1. Abre la aplicación.
2. Sube una imagen o toma una foto.
3. Espera el conteo automático.
4. Corrige si es necesario (agrega o quita estacas).
5. ¡Listo! Tienes el total confirmado.

## Tecnologías

- YOLOv8 (Modelo de detección)
- ONNX Runtime Web (Ejecución en navegador)
- HTML5 / CSS3 / JavaScript (Vanilla)
