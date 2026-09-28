"""
ci_smoke_test.py — chequeo rápido de CI (Nivel 1), sin videos reales ni AWS.

No mide AUC ni reemplaza la Suite Manual (correr_suite.py) -- esos números
solo tienen sentido contra los 15 videos reales de ground_truth.json, que no
viven en el repo (pesan 212MB) y por ahora solo existen en un banco local.

Lo que SÍ hace, y por qué sirve como red de seguridad en cada push:
  1. Corre el autotest de calcular_metricas.py (el cálculo de AUC/latencia
     debe ser correcto ANTES de confiar en cualquier número).
  2. Genera un video sintético de unos segundos (un cuadrado moviéndose sobre
     fondo negro) -- sin depender de ningún archivo externo -- y lo corre
     por evaluar_video_v3.evaluar() completo: MotionDetector real (Capa 0),
     ClipScorer real con pesos de CLIP reales (Capa 1), decidir() real
     (tiers.py). Esto no prueba que el sistema "detecte violencia" (el
     cuadrado no es una persona ni un evento real) -- prueba que la cadena
     de imports y el pipeline completo corren sin romperse: si alguien rompe
     una firma de función en vision.py/tiers.py/common.py/transport.py, esto
     falla en minutos, no cuando alguien corra la Suite Manual completa a
     mano.
  3. Confirma que el CSV de salida tiene las columnas esperadas.

Uso: python ci_smoke_test.py
"""
import csv
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))


def generar_video_sintetico(path, segundos=3, fps=10, size=(320, 240)):
    """Cuadrado blanco moviéndose en diagonal sobre fondo negro -- movimiento
    real para que MotionDetector (MOG2) tenga algo que restar del fondo."""
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_frames = segundos * fps
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int((i / n_frames) * (w - 40))
        y = int((i / n_frames) * (h - 40))
        cv2.rectangle(frame, (x, y), (x + 40, y + 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def paso_selftest_metricas():
    print("[1/3] Autotest de calcular_metricas.py ...")
    # PYTHONIOENCODING=utf-8: calcular_metricas.py imprime "✓"/"✗"; sin esto,
    # una consola no-UTF-8 (cp1252 en Windows) revienta con UnicodeEncodeError
    # aunque el cálculo en sí esté bien -- no es un fallo real del selftest.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, os.path.join(AQUI, "calcular_metricas.py"), "--selftest"], env=env)
    if r.returncode != 0:
        sys.exit("FALLÓ el autotest de métricas -- no se puede confiar en el cálculo de AUC/latencia.")
    print("      OK")


def paso_pipeline_sintetico():
    print("[2/3] Pipeline completo (Capa 0+1+decisión) sobre video sintético ...")
    sys.path[:0] = [AQUI, os.path.join(AQUI, "..", "cascade")]
    import evaluar_video_v3 as ev  # import tardío: necesita el sys.path de arriba

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "sintetico.mp4")
        csv_path = os.path.join(tmp, "smoke.csv")
        generar_video_sintetico(video_path)
        # "persona" ejercita ClipScorer real (descarga/usa los pesos de CLIP)
        # sin necesitar que el cuadrado sea reconocido como nada -- lo que
        # importa es que la cadena completa corra sin excepciones.
        ev.evaluar(video_path, "persona", sample_every=1, clear_margin=ev.CLEAR_MARGIN_DEFAULT,
                   out_csv=csv_path, use_vlm=False)

        with open(csv_path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        esperado = ["frame_idx", "time_seconds", "score", "decision", "detected",
                    "vlm_called", "vlm_confirmed", "vlm_reason"]
        if header != esperado:
            sys.exit(f"FALLÓ: columnas del CSV cambiaron.\n  esperado: {esperado}\n  real: {header}")
    print("      OK")


def paso_imports_capa2():
    print("[3/3] Import de vlm.py (Capa 2, sin llamar a Bedrock) ...")
    import vlm as vlm_mod  # noqa: F401 -- solo confirma que importa sin credenciales AWS
    if not hasattr(vlm_mod, "judge") or not hasattr(vlm_mod, "QUESTION"):
        sys.exit("FALLÓ: vlm.py no expone judge()/QUESTION como se esperaba.")
    print("      OK")


if __name__ == "__main__":
    paso_selftest_metricas()
    paso_pipeline_sintetico()
    paso_imports_capa2()
    print("\nSmoke test de CI: TODO OK.")
