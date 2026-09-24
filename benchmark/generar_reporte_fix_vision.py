# -*- coding: utf-8 -*-
"""Reporte: propuesta de fix en la Capa 0 (MIN_MOTION_AREA escalado por
resolución) -- comparación línea base (producción) vs candidato, sobre la
suite completa de 15 entradas / 12 videos, sin Mímir (--vlm off), corrido el
2026-09-22."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER

styles = getSampleStyleSheet()

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
code_style = ParagraphStyle("Code", parent=styles["Normal"], fontName="Courier",
                             fontSize=8.3, leading=11.5, textColor=colors.HexColor("#16211f"),
                             backColor=SURFACE2, borderPadding=8, spaceAfter=10)
metric_num_style = ParagraphStyle("MetricNum", parent=styles["Normal"], fontSize=20,
                                    fontName="Helvetica-Bold", alignment=TA_CENTER,
                                    textColor=OK, leading=23)
metric_num_small_style = ParagraphStyle("MetricNumSmall", parent=metric_num_style,
                                          fontSize=13, leading=15)
metric_lbl_style = ParagraphStyle("MetricLbl", parent=styles["Normal"], fontSize=8.3,
                                    alignment=TA_CENTER, textColor=MUTE, leading=10.5)

story = []

# --- Header ------------------------------------------------------------
story.append(Paragraph("PROPUESTA · CAPA 0 (MOVIMIENTO)", eyebrow_style))
story.append(Paragraph("Escalar MIN_MOTION_AREA por resolución", title_style))
story.append(Paragraph(
    "Comparación con la Suite Manual: mismos 15 casos, mismo código de Capa 1 y "
    "decisión (tiers.py real), un solo cambio de una línea en la Capa 0. Sin "
    "Mímir (--vlm apagado) para aislar el efecto y no gastar en Bedrock. "
    "Corrido el 22 de septiembre de 2026.", lede_style))
story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=14))

# --- Resumen ejecutivo ---------------------------------------------------
story.append(Paragraph("Resumen ejecutivo", h2_style))
story.append(Paragraph(
    "El umbral de movimiento (<font face=\"Courier\">MIN_MOTION_AREA</font>) "
    "está fijo en 500 píxeles², calibrado para 1080p. La mayoría de los videos "
    "de este banco son 320×240 -- casi 26 veces menos área -- así que ese "
    "mismo umbral les exige, en proporción, mucho más movimiento del que "
    "debería para considerar algo relevante. El fix escala el umbral según el "
    "área real del frame. Resultado: <b>el AUC promedio de los 8 casos con "
    "evento real sube de 0.514 a 0.654 (+0.140)</b>, mejora en 7 de 8 videos, "
    "y <b>cero</b> falsas alarmas confirmadas nuevas en los 7 casos de "
    "control -- aunque sube el número de frames \"ambiguous\" en negativos "
    "(detalle en la página siguiente), lo que en producción con Mímir "
    "prendido significa más llamadas a Bedrock, no más falsas alarmas.",
    body_style))

metrics = [
    ("0.514 <font size=12>→</font> 0.654", "AUC promedio\n(8 videos con evento)", True),
    ("7 / 8", "videos que mejoraron", False),
    ("+0.140", "ganancia promedio de AUC", False),
    ("0", "falsas alarmas\nconfirmadas nuevas", False),
]
metric_cells = []
for num, lbl, chico in metrics:
    estilo_num = metric_num_small_style if chico else metric_num_style
    metric_cells.append([Paragraph(num, estilo_num),
                          Paragraph(lbl.replace("\n", "<br/>"), metric_lbl_style)])
metric_table = Table([[c[0] for c in metric_cells], [c[1] for c in metric_cells]],
                      colWidths=[4.25*cm]*4)
metric_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), SURFACE2),
    ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
    ("TOPPADDING", (0, 0), (-1, 0), 12),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
    ("TOPPADDING", (0, 1), (-1, 1), 2),
    ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
]))
story.append(metric_table)
story.append(Spacer(1, 12))

# --- El cambio exacto ------------------------------------------------------
story.append(Paragraph("El cambio, línea por línea", h2_style))
story.append(Paragraph(
    "En <font face=\"Courier\">cascade/vision.py</font>, dentro de "
    "<font face=\"Courier\">MotionDetector</font>: en vez de comparar el área "
    "de cada contorno de movimiento contra un número fijo, se compara contra "
    "ese número ajustado a la resolución real del frame.", body_style))

story.append(Paragraph(
    "<font color='#b8452f'>- MIN_MOTION_AREA = 500</font><br/>"
    "<font color='#1f8a6f'>+ MIN_MOTION_AREA = 500  # calibrado para 1080p (1920x1080 px)</font><br/>"
    "<font color='#1f8a6f'>+ REFERENCE_AREA = 1920 * 1080</font><br/>"
    "&nbsp;&nbsp;...<br/>"
    "<font color='#1f8a6f'>+ area_frame = w * h</font><br/>"
    "<font color='#1f8a6f'>+ min_area_ajustada = MIN_MOTION_AREA * (area_frame / REFERENCE_AREA)</font><br/>"
    "<font color='#b8452f'>- if cv2.contourArea(c) < MIN_MOTION_AREA:</font><br/>"
    "<font color='#1f8a6f'>+ if cv2.contourArea(c) < min_area_ajustada:</font>",
    code_style))
story.append(Paragraph(
    "Para un video de 320×240, min_area_ajustada queda en ~17px² en vez de "
    "500px² -- exige mucho menos movimiento absoluto para generar un ROI, "
    "algo correcto porque en un frame tan chico un objeto real ocupa muy "
    "pocos píxeles de por sí. No toca ni CLIP (Capa 1) ni la decisión "
    "(tiers.py) -- solo qué recuadros le llegan a la Capa 1 para puntuar.",
    note_style))

story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=12))

# --- Tabla comparativa ------------------------------------------------------
story.append(Paragraph("Resultado, video por video", h2_style))
story.append(Paragraph(
    "Mismo video, mismo concepto, mismo umbral de CLIP y misma decisión -- "
    "solo cambia qué recuadros le llegaron a puntuar.", body_style))

data = [
    ["Video", "Concepto", "AUC base", "AUC candidato", "Diferencia"],
    ["Fighting003_x264A.mp4", "violencia", "0.4270", "0.5342", "+0.107"],
    ["Shooting002_x264A.mp4", "violencia", "0.5352", "0.7618", "+0.227"],
    ["Assault011_x264.mp4", "violencia", "0.4913", "0.4622", "-0.029"],
    ["Fighting042_x264A.mp4", "violencia", "0.5739", "0.6673", "+0.093"],
    ["Robbery048_x264.mp4", "robos", "0.4973", "0.7558", "+0.259"],
    ["Robbery102_x264.mp4", "robos", "0.3857", "0.6123", "+0.227"],
    ["Burglary024_x264A.mp4", "robos", "0.5492", "0.7677", "+0.218"],
    ["caida_escaleras.mp4", "caidas", "0.6492", "0.6693", "+0.020"],
    ["Promedio (8 videos)", "", "0.5136", "0.6538", "+0.140"],
]
col_widths = [5.4*cm, 3.0*cm, 2.2*cm, 2.6*cm, 2.2*cm]
tabla = Table(data, colWidths=col_widths, repeatRows=1)
estilo_tabla = [
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ("BACKGROUND", (0, 0), (-1, 0), SURFACE2),
    ("BACKGROUND", (0, -1), (-1, -1), SURFACE2),
    ("TEXTCOLOR", (0, 0), (-1, 0), MUTE),
    ("FONTSIZE", (0, 0), (-1, -1), 8.6),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("LINEABOVE", (0, -1), (-1, -1), 1, BORDER),
]
for i, row in enumerate(data[1:-1], start=1):
    diff = row[4]
    color = OK if diff.startswith("+") else DANGER
    estilo_tabla.append(("TEXTCOLOR", (4, i), (4, i), color))
    estilo_tabla.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
estilo_tabla.append(("TEXTCOLOR", (4, len(data)-1), (4, len(data)-1), OK))
tabla.setStyle(TableStyle(estilo_tabla))
story.append(tabla)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "Los otros 7 casos de la suite (Fighting042/persona, Normal_Videos_015 en "
    "sus 3 conceptos, caida_judo, cuchillo_afilado, cuchillo_piedra) no tienen "
    "AUC porque son negativos puros o el objeto está presente en todo el "
    "video. Ahí <b>0 llegaron a \"clear\"</b> en las dos corridas -- ninguna "
    "falsa alarma confirmada nueva. Pero hay un costo real que no hay que "
    "esconder: el umbral más bajo hace que varios de estos negativos generen "
    "muchas más filas \"ambiguous\" (ver tabla abajo) -- en producción, con "
    "Mímir prendido, eso significa más llamadas a Bedrock sobre video "
    "tranquilo, no más falsas alarmas confirmadas.",
    body_style))

data2 = [
    ["Video / concepto", "\"ambiguous\" base", "\"ambiguous\" candidato"],
    ["Fighting042 / persona", "229", "1365"],
    ["Normal_Videos_015 / violencia", "0", "250"],
    ["Normal_Videos_015 / robos", "0", "264"],
    ["Normal_Videos_015 / persona", "1", "266"],
    ["caida_judo / caidas", "130", "142"],
    ["cuchillo_afilado / cuchillo", "0", "24"],
    ["cuchillo_piedra / cuchillo", "418", "491"],
]
tabla_amb = Table(data2, colWidths=[6.5*cm, 3.7*cm, 3.7*cm], repeatRows=1)
tabla_amb.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("BACKGROUND", (0, 0), (-1, 0), SURFACE2),
    ("TEXTCOLOR", (0, 0), (-1, 0), MUTE),
    ("FONTSIZE", (0, 0), (-1, -1), 8.6),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("TEXTCOLOR", (1, 0), (1, -1), MUTE),
    ("TEXTCOLOR", (2, 0), (2, -1), AMBER),
]))
story.append(tabla_amb)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Con Mímir apagado (--vlm, como se corrió esta comparación) esto no "
    "cuesta nada. Con Mímir prendido en producción, cada fila \"ambiguous\" "
    "es una posible llamada real (limitada a 1 cada 6s por cámara+etiqueta, "
    "~$0.000076 cada una) -- así que antes de llevar este fix a producción "
    "conviene correr al menos estos 7 negativos <b>con --vlm</b> y confirmar "
    "que Mímir sigue rechazando todo, no solo que la Capa 0+1 se puso más "
    "sensible.", note_style))

story.append(PageBreak())

# --- Lectura por concepto ----------------------------------------------------
story.append(Paragraph("Lectura por concepto", h2_style))

lecturas = [
    ("robos", OK, "El que más se beneficia",
     "Los tres videos de robo suben entre +0.22 y +0.26 de AUC -- el salto "
     "más grande de la tabla. Coincide con lo que ya sabíamos de la Suite "
     "Manual: <b>robos</b> tenía 0 de 3 confirmaciones de Mímir a pesar de que "
     "CLIP sí distinguía el evento (AUC 0.61-0.77 ya en la corrida anterior). "
     "Con el umbral viejo, la Capa 0 probablemente estaba dejando pasar menos "
     "recuadros justo en los momentos clave (forcejeo, despojo), lo que le "
     "restaba oportunidades a CLIP de puntuar bien. No garantiza que Mímir "
     "vaya a confirmar más -- eso exige correr con --vlm -- pero le da a la "
     "Capa 1 mejor materia prima para trabajar."),
    ("violencia", AMBER, "Mejora en 3 de 4, sin sorpresas",
     "Fighting003, Shooting002 y Fighting042 mejoran. Assault011 baja "
     "levemente (-0.029), pero ya estaba documentado en la Suite Manual como "
     "un caso <i>sin señal</i>: CLIP nunca distinguió el evento ahí, ni "
     "dentro ni fuera del rango anómalo, con el umbral viejo tampoco. Este "
     "fix no iba a resolver eso -- es un problema de la Capa 1, no de la "
     "Capa 0."),
    ("caidas", OK, "Mejora chica, pero sin riesgo",
     "caida_escaleras sube +0.020 (0.6492 -> 0.6693). Es el único video de "
     "este grupo con resolución ya alta (720x576, más cerca del 1080p de "
     "referencia), así que el fix lo toca poco -- esperable, y consistente "
     "con la explicación del cambio."),
]
for nombre, color, titulo, texto in lecturas:
    story.append(Paragraph(
        f'<font color="{color.hexval()}">●</font> <b>{nombre}</b> — {titulo}',
        h3_style))
    story.append(Paragraph(texto, body_style))

story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=12))

# --- Cómo se corrió / honestidad metodológica -------------------------------
story.append(Paragraph("Cómo se corrió esta comparación", h2_style))
story.append(Paragraph(
    "Dos corridas de <font face=\"Courier\">correr_suite.py</font> sobre las "
    "mismas 15 entradas de <font face=\"Courier\">ground_truth.json</font>, "
    "sin <font face=\"Courier\">--vlm</font>: la primera con "
    "<font face=\"Courier\">vision.py</font> tal cual está en producción "
    "(<font face=\"Courier\">cascade/</font>, umbral fijo), la segunda con el "
    "candidato (umbral escalado). <font face=\"Courier\">tiers.py</font>, "
    "<font face=\"Courier\">common.py</font> y "
    "<font face=\"Courier\">transport.py</font> son exactamente los mismos de "
    "producción en ambas corridas -- lo único que cambia es la línea de "
    "vision.py de arriba. Línea base: 13.3 min. Candidato: 97.2 min (con el "
    "umbral más bajo, muchos más recuadros llegan a puntuarse con CLIP, sobre "
    "todo en los videos de 320×240).", body_style))
story.append(Paragraph(
    "Límites honestos de esta prueba: (1) el AUC de arriba es de Capa 0+1 "
    "solamente -- no mide por sí solo si Mímir confirma más eventos reales; "
    "eso se validó por separado con Mímir real sobre los 7 casos de control "
    "(siguiente sección) pero no sobre los 8 videos con evento -- ahí el "
    "efecto en la confirmación final de Mímir sigue sin medirse con --vlm; "
    "(2) 8 videos con evento real es una muestra chica -- es una señal "
    "fuerte, no una prueba estadística; (3) el aumento de tiempo de cómputo "
    "(7x más lento en estos videos chicos) y de frames \"ambiguous\" en "
    "negativos son costos reales a considerar si esto corre en vivo sobre "
    "muchas cámaras simultáneas, no solo en el benchmark.", note_style))

story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=12))

# --- Validación con Mímir real -----------------------------------------------
story.append(Paragraph("Validación: se corrieron los 7 negativos con Mímir real", h2_style))
story.append(Paragraph(
    "Para cerrar la duda de la sección anterior, se corrieron los mismos 7 "
    "casos de control con <font face=\"Courier\">--vlm</font> (Nova Lite vía "
    "Bedrock, real, no simulado). Resultado: <b>Mímir se comportó exactamente "
    "como se esperaba en los 7</b> -- confirmó donde debía confirmar (persona "
    "presente, cuchillo visible) y rechazó donde debía rechazar (lanzamiento "
    "controlado, cuchillo tapado por la mano, video sin evento). El umbral "
    "más sensible de la Capa 0 le manda más consultas a Mímir, pero no le "
    "hace confirmar nada que no debería.", body_style))

data3 = [
    ["Video / concepto", "Esperado", "Mímir confirmó"],
    ["Fighting042 / persona", "Sí (persona presente)", "Sí (11/11)"],
    ["caida_judo / caidas", "No (lanzamiento controlado)", "No (0/2)"],
    ["cuchillo_afilado / cuchillo", "No (mano tapa la hoja)", "No (0/2)"],
    ["cuchillo_piedra / cuchillo", "Sí (hoja visible)", "Sí (2/3)"],
    ["Normal_Videos_015 / violencia", "No (negativo genuino)", "No (0/2)"],
    ["Normal_Videos_015 / robos", "No (negativo genuino)", "No (0/2)"],
    ["Normal_Videos_015 / persona", "Sí (hay gente)", "Sí (2/2)"],
]
tabla_val = Table(data3, colWidths=[5.6*cm, 4.7*cm, 3.6*cm], repeatRows=1)
tabla_val.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("BACKGROUND", (0, 0), (-1, 0), SURFACE2),
    ("TEXTCOLOR", (0, 0), (-1, 0), MUTE),
    ("FONTSIZE", (0, 0), (-1, -1), 8.6),
    ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("TEXTCOLOR", (2, 1), (2, -1), OK),
    ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
]))
story.append(tabla_val)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "24 llamadas reales a Mímir en total, costo real: USD 0.0018. Tiempo: "
    "15.2 min. Con esto, el punto (1) de los límites de la página anterior "
    "queda cerrado para estos 7 casos -- el fix no genera falsas alarmas "
    "confirmadas ni con Mímir real prendido, solo más tráfico hacia él.",
    note_style))

story.append(Spacer(1, 8))
story.append(HRFlowable(width="100%", color=BORDER, thickness=0.7, spaceAfter=10))
story.append(Paragraph(
    "Reporte generado por benchmark/generar_reporte_fix_vision.py sobre los "
    "resultados de benchmark/resultados_completa_baseline/, "
    "benchmark/resultados_completa_candidato/ y "
    "benchmark/resultados_negativos_vlm/ (22 de septiembre de 2026).",
    note_style))

doc = SimpleDocTemplate(
    "Reporte_Fix_MotionDetector_2026-09-22.pdf",
    pagesize=letter,
    topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm,
)
doc.build(story)
print("PDF generado.")
