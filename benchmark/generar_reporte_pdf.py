# -*- coding: utf-8 -*-
"""Genera el reporte final en PDF de la Suite Manual de Heimdall (15 videos,
Mímir real conectado) -- para mandarle a Brayham."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, Image)
from reportlab.lib.enums import TA_CENTER
import csv
import glob
import os
from PIL import Image as PILImage

styles = getSampleStyleSheet()


def _img(path, width_cm=6.5):
    """Imagen con la relación de aspecto REAL del archivo (no asume 320x240 --
    algunos clips de Wikimedia son 1920x1080 o 1056x744)."""
    w_px, h_px = PILImage.open(path).size
    w = width_cm * cm
    h = w * h_px / w_px
    return Image(path, width=w, height=h)

ACCENT = colors.HexColor("#0d7a8c")
AMBER = colors.HexColor("#b8791a")
OK = colors.HexColor("#1f8a6f")
DANGER = colors.HexColor("#b8452f")
MUTE = colors.HexColor("#5c6e70")
BORDER = colors.HexColor("#c4ced1")
SURFACE2 = colors.HexColor("#eef1f2")

title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=22,
                              textColor=colors.HexColor("#16211f"), spaceAfter=4)
eyebrow_style = ParagraphStyle("Eyebrow", parent=styles["Normal"], fontSize=10,
                                textColor=ACCENT, fontName="Helvetica-Bold",
                                spaceAfter=6)
lede_style = ParagraphStyle("Lede", parent=styles["Normal"], fontSize=11.5,
                             textColor=MUTE, leading=16, spaceAfter=14)
h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=15,
                           textColor=colors.HexColor("#16211f"), spaceBefore=16,
                           spaceAfter=8)
h3_style = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5,
                           textColor=colors.HexColor("#16211f"), spaceBefore=10,
                           spaceAfter=4)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                              leading=14.5, spaceAfter=8)
note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=9,
                              textColor=MUTE, leading=13, spaceAfter=6)
callout_style = ParagraphStyle("Callout", parent=styles["Normal"], fontSize=10,
                                 leading=14.5, textColor=colors.HexColor("#16211f"),
                                 leftIndent=10, spaceAfter=4)

story = []

# --- Header ---------------------------------------------------------------
story.append(Paragraph("REPORTE · SUITE MANUAL DE HEIMDALL", eyebrow_style))
story.append(Paragraph("Benchmark de detección con Mímir real", title_style))
story.append(Paragraph(
    "15 videos, 4 conceptos (violencia, robos, caidas, cuchillo), Amazon Nova Lite "
    "vía Bedrock conectado de verdad. Corrido el 17 de septiembre de 2026.",
    lede_style))
story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=14))

# --- Resumen ejecutivo -----------------------------------------------------
story.append(Paragraph("Resumen ejecutivo", h2_style))
story.append(Paragraph(
    "De los 11 videos con un evento real, el sistema confirmó 6 (55%) y nunca "
    "generó una falsa alarma en los 4 videos de control (negativos genuinos o "
    "casos donde no debía alertar). <b>Persona</b> y <b>caidas</b> funcionan de "
    "forma consistente; <b>violencia</b> funciona en 3 de 4 casos; <b>robos</b> "
    "es el punto débil real, con 0 de 3 confirmaciones a pesar de que la Capa 1 "
    "(CLIP) sí distinguía el evento en los tres. Costo total real: "
    "<b>USD 0.008</b> (109 llamadas a Mímir). Tiempo total: ~90 minutos.",
    body_style))

# --- Cómo funciona el pipeline: qué hace cada capa, y dónde entra ---------
story.append(Paragraph("Cómo funciona el pipeline: qué hace cada capa", h2_style))
story.append(Paragraph(
    "Un frame pasa por tres capas en orden. Cada una solo entra en acción "
    "sobre lo que le dejó pasar la anterior -- ninguna ve el video completo "
    "de una.", body_style))

capas = [
    (ACCENT, "Capa 0 -- Movimiento", "vision.py, clase MotionDetector",
     "Entra primero, en TODO frame, sin excepción (evaluar_video_v3.py línea "
     "143). Resta el fondo (MOG2) y propone recuadros (ROIs) donde hubo "
     "movimiento real. Si no encuentra ninguno, el frame se descarta ahí "
     "mismo -- columna \"Sin movim.\" de la tabla de trazabilidad -- sin "
     "gastar CLIP ni Mímir."),
    (ACCENT, "Capa 1 -- CLIP", "vision.py, clase ClipScorer",
     "Entra solo sobre los recortes que dejó pasar la Capa 0 -- nunca ve el "
     "frame completo. Puntúa cada recorte contra los prompts del concepto "
     "(ej. \"a person fallen on the floor\") con margen contrastivo. Si el "
     "score no llega al umbral, se descarta (\"Mov+CLIP bajo\"); si lo "
     "supera, la decisión pasa a \"ambiguous\" (eventos abstractos) o "
     "\"clear\" directo (conceptos concretos con margen alto, ej. persona)."),
    (AMBER, "Capa 2 -- Mímir", "vlm.py, Nova Lite vía Bedrock",
     "Entra SOLO cuando la decisión queda en \"ambiguous\" -- es la más cara, "
     "por eso las dos capas anteriores ya filtraron todo lo que se pudo "
     "filtrar gratis. Recibe una ráfaga de 4 frames (1.5s de historia, no un "
     "solo instante) y responde SI/NO a una pregunta específica del "
     "concepto. Su respuesta decide si el evento se confirma (\"clear\") o "
     "se descarta (\"none\")."),
]
for color, nombre, archivo, texto in capas:
    story.append(Paragraph(
        f'<font color="{color.hexval()}">●</font> <b>{nombre}</b> '
        f'<font face="Courier" size="8.5" color="{MUTE.hexval()}">({archivo})</font>',
        h3_style))
    story.append(Paragraph(texto, body_style))

story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=12))

# --- Tabla de resultados ----------------------------------------------------
story.append(Paragraph("Resultado por video", h2_style))

data = [
    ["Video", "Concepto", "AUC", "Latencia", "Detectó", "Llam. VLM"],
    ["Fighting003_x264A.mp4", "violencia", "0.5342", "27.97s", "Sí", "17"],
    ["Shooting002_x264A.mp4", "violencia", "0.7618", "2.00s", "Sí", "7"],
    ["Assault011_x264.mp4", "violencia", "0.4622", "n/a", "No", "13"],
    ["Fighting042_x264A.mp4", "violencia", "0.6673", "15.30s", "Sí", "11"],
    ["Normal_Videos_015.mp4", "violencia (negativo)", "n/a", "n/a", "No (correcto)", "2"],
    ["Fighting042_x264A.mp4", "persona", "n/a", "n/a", "Sí", "11"],
    ["Normal_Videos_015.mp4", "persona", "n/a", "n/a", "Sí", "2"],
    ["Robbery048_x264.mp4", "robos", "0.7558", "n/a", "No", "7"],
    ["Robbery102_x264.mp4", "robos", "0.6123", "n/a", "No", "8"],
    ["Burglary024_x264A.mp4", "robos", "0.7677", "n/a", "No", "6"],
    ["Normal_Videos_015.mp4", "robos (negativo)", "n/a", "n/a", "No (correcto)", "2"],
    ["caida_escaleras.mp4", "caidas", "0.6693", "7.92s", "Sí", "10"],
    ["caida_judo_src.webm", "caidas (negativo difícil)", "n/a", "n/a", "No (correcto)", "2"],
    ["cuchillo_afilado.mp4", "cuchillo", "n/a", "n/a", "No", "2"],
    ["cuchillo_piedra.mp4", "cuchillo", "n/a", "n/a", "Sí", "3"],
]

col_widths = [4.6*cm, 3.6*cm, 1.7*cm, 1.9*cm, 2.3*cm, 2.0*cm]
tabla = Table(data, colWidths=col_widths, repeatRows=1)

estilo_tabla = [
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
    ("BACKGROUND", (0, 0), (-1, 0), SURFACE2),
    ("TEXTCOLOR", (0, 0), (-1, 0), MUTE),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
]
# Colorear la columna "Detectó" según el resultado
for i, row in enumerate(data[1:], start=1):
    detecto = row[4]
    if detecto.startswith("Sí"):
        estilo_tabla.append(("TEXTCOLOR", (4, i), (4, i), OK))
    elif "correcto" in detecto:
        estilo_tabla.append(("TEXTCOLOR", (4, i), (4, i), OK))
    else:
        estilo_tabla.append(("TEXTCOLOR", (4, i), (4, i), DANGER))
    estilo_tabla.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))

tabla.setStyle(TableStyle(estilo_tabla))
story.append(tabla)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "AUC/latencia salen \"n/a\" cuando el video no tiene un tramo negativo "
    "comparable dentro del mismo clip (negativos puros, o el objeto/persona "
    "presente en todo el video) -- no es un cálculo que falló.",
    note_style))

story.append(PageBreak())

# --- Hallazgos por concepto -------------------------------------------------
story.append(Paragraph("Hallazgos por concepto", h2_style))

hallazgos = [
    ("persona", OK, "Funciona de forma muy consistente",
     "13 de 13 llamadas confirmaron correctamente, en dos videos distintos. "
     "Es el concepto más confiable de los cuatro.",
     []),
    ("caidas", OK, "Funciona bien, incluyendo un caso límite",
     "Confirmó la caída real en escaleras (AUC 0.6693, latencia 7.92s) y "
     "rechazó correctamente un lanzamiento de judo controlado -- reconociendo "
     "por sí solo que era \"una posición de lucha controlada\", sin que la "
     "pregunta se lo pidiera explícitamente.",
     [("thumbs/esc2_1000.jpg", "caida_escaleras -- confirmado (Sí)."),
      ("thumbs/judo_0160.jpg", "caida_judo -- rechazado a propósito (No, correcto).")]),
    ("violencia", AMBER, "Funciona cuando el golpe es visualmente claro",
     "3 de 4 videos con evento real confirmaron, con 0 falsos positivos. El "
     "único que falló (Assault011) no fue por Mímir: CLIP (Capa 1) nunca "
     "encontró señal de violencia en ese video en particular, ni dentro ni "
     "fuera del evento -- es un caso sin señal, no un caso difícil.",
     [("thumbs/assault_peak_1170.jpg",
       "Assault011 -- el frame con el score de CLIP más alto de TODO el video "
       "(fuera del evento real): una vara en la mano generó más señal que la "
       "propia agresión.")]),
    ("robos", DANGER, "El punto débil real -- 0 de 3 confirmó",
     "Se probaron tres tipos distintos de robo (arma oculta/pequeña, forcejeo "
     "sin despojo visible, hurto silencioso sin fuerza) y ninguno confirmó, "
     "a pesar de que CLIP sí distinguía el evento en los tres (AUC 0.61-0.77). "
     "El cuello de botella está en Mímir: necesita ver fuerza, despojo o "
     "forcejeo claro, evidencia que estos tres robos reales no mostraban en "
     "los frames muestreados.",
     [("thumbs/r_0540.jpg", "Robbery048 -- arma apuntada, pero muy pequeña/oculta para confirmar."),
      ("thumbs/rob102_1150.jpg", "Robbery102 -- forcejeo claro, pero sin el momento del despojo."),
      ("thumbs/burg_0300.jpg", "Burglary024 -- hurto silencioso, indistinguible de un empleado trabajando.")]),
    ("cuchillo", AMBER, "Depende de si el objeto está oculto",
     "Falló cuando la mano tapaba el cuchillo contra la piedra de afilar, y "
     "confirmó (2 de 3 llamadas) con el mismo tipo de objeto claramente "
     "visible en otro video -- confirmando que el problema es la oclusión, "
     "no la resolución ni el tamaño del objeto en sí.",
     [("thumbs/knife_call_0186.jpg", "cuchillo_afilado -- la mano tapa la hoja (No)."),
      ("thumbs/piedra_0600.jpg", "cuchillo_piedra -- la hoja queda visible (Sí).")]),
]

for nombre, color, titulo, texto, fotos in hallazgos:
    story.append(Paragraph(
        f'<font color="{color.hexval()}">●</font> <b>{nombre}</b> — {titulo}',
        h3_style))
    story.append(Paragraph(texto, body_style))
    for path, pie in fotos:
        if os.path.exists(path):
            story.append(_img(path, width_cm=5.2))
            story.append(Paragraph(pie, note_style))
            story.append(Spacer(1, 6))

story.append(Spacer(1, 8))
story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=12))

# --- Opciones de mejora ------------------------------------------------------
story.append(Paragraph("Opciones de mejora identificadas", h2_style))
mejoras = [
    ("Ráfaga más larga para robos", "Un robo silencioso es un evento sostenido, "
     "no instantáneo. Probar 8 frames en 4-5s (en vez de 4 frames en 1.5s) "
     "solo para este concepto, sin costo adicional real por llamada."),
    ("Usar la hora quemada en el video", "Varios videos de vigilancia traen "
     "fecha/hora sobreimpresa en la imagen. Pedirle a Mímir que la lea y "
     "considere si es una hora inusual para la actividad observada le da "
     "contexto que hoy no tiene."),
    ("Detector de manos + zoom (cuchillo/armas ocultas)", "Usar el detector de "
     "pose (yolo_gate.py, hoy en experimental/) para ubicar las manos y "
     "recortar/ampliar esa región antes de mandarla a Mímir. Requiere más "
     "trabajo de implementación que las dos anteriores."),
]
for titulo, texto in mejoras:
    story.append(Paragraph(f"<b>{titulo}.</b> {texto}", body_style))

story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=10))
story.append(Paragraph(
    "Costo total de esta corrida: USD 0.008 (109 llamadas reales a Mímir, "
    "$0.000076 cada una). Tiempo total: ~90 minutos, corrido en 5 tramos para "
    "esquivar el vencimiento de la sesión de credenciales AWS. Reporte generado "
    "por la Suite Manual de Heimdall (correr_suite.py) -- fase 1 de la "
    "automatización propuesta, sin GitHub Actions ni infraestructura nueva "
    "todavía.", note_style))

story.append(PageBreak())

# --- Prueba visual de la Capa 0 (Movimiento) --------------------------------
story.append(Paragraph("Prueba visual: la Capa 0 (Movimiento) sí recorta de verdad", h2_style))
story.append(Paragraph(
    "MotionDetector.rois() (vision.py, idéntico al repo real salvo el fix de "
    "MIN_MOTION_AREA) corrió sobre Fighting042_x264A.mp4. Los recuadros rojos "
    "son los ROIs reales que produjo -- exactamente lo que después se recorta "
    "y se manda a CLIP, no una simulación aparte.", body_style))

imagenes = [
    ("thumbs/motion_roi_0300.jpg", "Frame 300 -- 2 ROIs, siguiendo a la persona que camina."),
    ("thumbs/motion_roi_0450.jpg", "Frame 450 -- 2 ROIs, sobre el forcejeo en la puerta."),
    ("thumbs/motion_roi_0850.jpg", "Frame 850 -- 5 ROIs, el momento más caótico de la escena."),
]
for path, pie in imagenes:
    if os.path.exists(path):
        story.append(_img(path, width_cm=7))
        story.append(Paragraph(pie, note_style))
        story.append(Spacer(1, 8))

story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=12))
story.append(Paragraph(
    "Esta es la misma línea de código (evaluar_video_v3.py, línea 143: "
    "<font face=\"Courier\">rois = motion.rois(frame)</font>) que corrió, sin "
    "excepción, en cada frame de los 15 videos del reporte -- no es una "
    "demostración aparte.", note_style))

story.append(PageBreak())

# --- Trazabilidad por capa ---------------------------------------------------
story.append(Paragraph("Trazabilidad por capa: dónde se descarta cada frame", h2_style))
story.append(Paragraph(
    "Cada fila de los CSV de resultado permite separar cuántos frames se "
    "descartaron por <b>falta de movimiento</b> (Capa 0, score=0.0 porque "
    "MotionDetector.rois() no encontró nada) de cuántos tuvieron movimiento "
    "pero CLIP los descartó igual (Capa 1, score por debajo del umbral). Es "
    "la prueba de que ambas capas están filtrando de verdad, no solo la "
    "última.", body_style))


def _leer_desglose(csv_path):
    none_sin_mov = none_con_mov = ambiguous = clear = 0
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = r["decision"]
            score = float(r["score"])
            if d == "none":
                if score == 0.0:
                    none_sin_mov += 1
                else:
                    none_con_mov += 1
            elif d == "ambiguous":
                ambiguous += 1
            elif d == "clear":
                clear += 1
    return none_sin_mov, none_con_mov, ambiguous, clear


body_cell_left_style = ParagraphStyle("BodyCellLeft", parent=styles["Normal"], fontSize=7.8,
                                        leading=9.5)

encabezado = ["Video / concepto", "Sin movim.", "Mov+CLIP bajo", "Ambiguous", "Clear"]
desglose_data = [encabezado]
for f in sorted(glob.glob("resultados_suite/*.csv")):
    nombre = os.path.basename(f).replace(".csv", "").replace("_", " ")
    n_sin_mov, n_con_mov, n_amb, n_clr = _leer_desglose(f)  # nunca reusar "cm": pisa reportlab.lib.units.cm
    desglose_data.append([Paragraph(nombre, body_cell_left_style),
                          str(n_sin_mov), str(n_con_mov), str(n_amb), str(n_clr)])

tabla2 = Table(desglose_data, colWidths=[6.5*cm, 2.4*cm, 2.9*cm, 2.4*cm, 1.8*cm], repeatRows=1)
tabla2.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 8.2),
    ("FONTSIZE", (1, 1), (-1, -1), 7.8),
    ("BACKGROUND", (0, 0), (-1, 0), SURFACE2),
    ("TEXTCOLOR", (0, 0), (-1, 0), MUTE),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
]))
story.append(tabla2)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "\"Sin movimiento\" y \"con movimiento, CLIP bajo\" juntos forman el total "
    "de \"none\" del reporte anterior -- aquí se separan para mostrar que la "
    "Capa 0 (movimiento) descarta una porción real y distinta en cada video "
    "(desde 3 frames en cuchillo_piedra hasta 637 en Burglary024), no un "
    "número fijo o simulado.", note_style))

doc = SimpleDocTemplate(
    "Reporte_Suite_Manual_Heimdall_2026-09-17.pdf",
    pagesize=letter,
    topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm,
)
doc.build(story)
print("PDF generado.")
