# Benchmark SkyEye — niveles YOLO y VLM ("Mimir")

Amplía el benchmark de 1 nivel (CLIP) a la cascada de 3 que corre en producción:

```
frame ─► YOLO (puerta + pose + detector) ─► CLIP (score) ─► Mimir/VLM (juicio)
```

## Archivos

| Archivo | Qué es |
|---|---|
| `skyeye_core.py` | CLIP con ventana deslizante. **Sin cambios.** |
| `yolo_gate.py` | **NUEVO.** YOLOv8-Pose (puerta de personas + caída por pose) + YOLO-World (detector open-vocab que se fusiona con CLIP). |
| `mimir.py` | **NUEVO.** El VLM de juicio, corriendo **local con Ollama** (sin AWS). Mismas preguntas que `cascade/vlm.py`. |
| `cascada.py` | **NUEVO.** Orquesta los 3 niveles y funde sus señales en 3 columnas de score comparables. |
| `evaluar_video.py` | Ampliado. Sin `--cascade` funciona **exactamente igual que antes**. Con `--cascade` corre YOLO + (opcional) VLM. |
| `calcular_metricas.py` | Ampliado. Detecta el CSV de cascada y compara CLIP vs CLIP+YOLO vs cascada completa. |

## Instalación

Ya tienes lo básico en `../venv` (CLIP, torch, opencv, sklearn). Falta:

```bash
../venv/Scripts/pip install ultralytics
```

Para el VLM (opcional, solo si usas `--vlm`):

1. Instala Ollama: https://ollama.com
2. `ollama pull llava`   (o `moondream` / `llava-phi3`, más livianos para CPU)

## Uso

**1) Base (igual que siempre) — solo CLIP:**
```bash
../venv/Scripts/python evaluar_video.py --video Fighting003_x264A.mp4 --prompt knife
```

**2) Cascada con YOLO (sin VLM):**
```bash
../venv/Scripts/python evaluar_video.py --video Fighting003_x264A.mp4 \
    --prompt "a person hitting another person" --concept violencia --cascade
```

**3) Cascada completa con VLM local:**
```bash
../venv/Scripts/python evaluar_video.py --video Fighting003_x264A.mp4 \
    --prompt "a person hitting another person" --concept violencia --cascade --vlm
```

**4) Métricas (compara las 3 variantes):**
```bash
../venv/Scripts/python calcular_metricas.py --selftest
../venv/Scripts/python calcular_metricas.py --scores scores_Fighting003_x264A.csv \
    --anomaly-start 90 --anomaly-end 240 --fps 30 --threshold 0.5
```

> En un CSV de cascada los scores van normalizados a 0..1 → usa `--threshold 0.5`
> para la latencia (en el CSV básico se sigue usando 0.268).

## Cómo encajan señales binarias en un AUC

El benchmark necesita un score por frame. YOLO da booleanos y el VLM un sí/no,
así que `cascada.py` los combina en tres columnas:

- **`score_clip`** — CLIP solo (línea base).
- **`score_clip_yolo`** — `max(clip_normalizado, confianza_YOLO-World)`.
- **`score_cascade`** — lo anterior, más:
  - evento abstracto sin persona → `0` (puerta de personas)
  - `caidas` con pose horizontal → `1` (lo resuelve YOLOv8-Pose, sin VLM)
  - zona ambigua + VLM confirma → `0.85` · VLM rechaza → `0.10`

`calcular_metricas.py` saca AUC y latencia de cada una y además informa cuántas
llamadas al VLM se hicieron (el coste).

## Flags útiles de `evaluar_video.py --cascade`

| Flag | Default | Para qué |
|---|---|---|
| `--concept` | se infiere del prompt | Fuerza `knife`/`caidas`/`robos`/`violencia`/`persona` |
| `--vlm` | off | Activa las llamadas al VLM local |
| `--vlm-model` | `llava` | Modelo de Ollama |
| `--vlm-budget` | 400 | Tope de llamadas al VLM por vídeo |
| `--no-yolo-world` | off | Desactiva la fusión con YOLO-World (solo puerta+pose) |
| `--clear` / `--floor` | 0.30 / 0.24 | Banda de score CLIP que se considera "ambigua". Los scores reales rondan 0.24 → ajústalos tras ver la distribución. |
