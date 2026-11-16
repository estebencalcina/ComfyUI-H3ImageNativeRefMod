# ComfyUI-H3ImageNativeRefMod — 8 imágenes (experimental)

Nodo nuevo para **MiniMax H3-Image** (NO H3 Video). Permite preparar y guardar una identidad con **1 a 8 fotografías**, y reutilizarla desde texto sin volver a subir las ocho fotos en cada generación. Está basado en el workflow de 8 referencias proporcionado por el usuario.

## Importante: alcance real

- Guarda **hasta 8 referencias latentes separadas y sus imágenes para Qwen**, en UN SOLO archivo llamado `Personaje_01.safetensors`. **No aprende ni fusiona un embedding compacto**, ni implementa el algoritmo comprimido original MiniMaxH3Mod.
- La generación posterior utiliza las ocho imágenes guardadas para Qwen y las ocho referencias latentes para el DiT. Esto **reduce la repetición de VAE encoding y de subir fotos**, pero puede aumentar VRAM/tokens y NO está probado que mejore la identidad de forma fiable.
- Esta implementación es una **prueba de concepto no ejecutada en GPU/ComfyUI**; código validado sintácticamente y enlaces revisados a nivel JSON. Si la identidad no se conserva, necesitamos logs y una comparación con las referencias directas antes de prometer un RefMod funcional.
- No uses archivos de `ComfyUI-MiniMaxH3Mod` (H3 Video) ni los archivos `h3-image-native-full-ref-v1` del prototipo anterior: son formatos distintos.

## Instalación

1. Copia la carpeta del repositorio a `ComfyUI/custom_nodes/ComfyUI-H3ImageNativeRefMod` (o clónala desde GitHub). Reinicia ComfyUI.
2. Abre `workflows/01_CREAR_REFMOD_8_IMAGENES.json`. Selecciona el VAE **MiniMax H3-Image**, NO el de video. Carga 8 imágenes (o al menos la primera), escribe `identity_name`, y ejecuta. Se guardará en `ComfyUI/models/h3_image_native_refmods/<nombre>.safetensors`.
3. Abre `workflows/02_T2I_USAR_IDENTIDAD_GUARDADA.json`. En el nodo `Load H3-Image Native Identity`, selecciona el archivo. Si no aparece, refresca ComfyUI o reinicia.
4. El prompt puede indicar `same character as <Picture 1> ... <Picture 8>`; el número de Picture corresponde al orden de las fotos suministradas. Para escenas de una sola persona, indica explícitamente que las ocho imágenes muestran al mismo sujeto.
5. Primera prueba: 1024x1024, 6 pasos, CFG 1, semilla fija, comparación con y sin referencias. Reduce `reference_short_edge` a 512 si tienes presión de VRAM. Para conservar mejor detalle de rostro, compara 768 vs 1024.

## Componentes

- `SaveH3ImageNativeIdentity`: entrada obligatoria `image_1`, opcionales `image_2...image_8`, `vae`, nombre y tamaño de referencia.
- `LoadH3ImageNativeIdentity`: carga el archivo desde la lista y muestra vista previa de la primera referencia.
- `H3ImageNativeRefModConditioning`: integra varias referencias en el acondicionamiento estándar de H3-Image.

## GitHub

Publica `__init__.py`, `nodes.py`, `README.md`, `.gitignore` y `workflows/`; no subas fotos privadas, modelos grandes, ni identidades `.safetensors`.

## Validación pendiente

No se ha confirmado todavía el efecto sobre fidelidad de identidad ni el consumo exacto de memoria en la GPU. El sistema no reemplaza al RefMod comprimido de H3 Video. La comparación técnica correcta es contra `MiniMaxH3ImageConditioning` usando las mismas ocho imágenes y semilla.
