# Material de prueba: procedencia y licencias

Todo el material del banco de pruebas procede de **Wikimedia Commons** y tiene
licencia libre verificada por API (`imageinfo.extmetadata.LicenseShortName`) antes
de usarse. No se usa material privado, con derechos reservados ni de fuentes no
verificables.

Licencias aceptadas: dominio público, CC0, CC BY, CC BY-SA. Se rechaza cualquier
cosa con cláusula NC (no comercial) o ND (sin obras derivadas), porque el material
se recorta y se recodifica.

## Escenas negativas — no debería alertar

| Escena | Fichero original | Licencia | Autor |
|---|---|---|---|
| `calle_peatones` | Scramble Crossing at Robinson Road in Singapore - September 2022 | CC BY 4.0 | Jun Jie Yam |
| `naturaleza_vacia` | Mount Rainier Weather Timelapse | Dominio público | Mount Rainier National Park (NPS) |
| `obra_normal` | Building construction Moira Close Broadwater Farm Haringey 2025 21 | CC BY-SA 4.0 | Philafrenzy |

## Escenas positivas — debería alertar

| Escena | Fichero original | Licencia | Autor |
|---|---|---|---|
| `caida_escaleras` | Bits & Pieces - BP152 Falling down the stairs - EYE FLM7636 | Dominio público | EYE Filmmuseum |
| `caida_judo` | Tai-otoshi in detail by Laszlo Horvath edited 0 | CC BY-SA 4.0 | Rodrigo |
| `disturbios_saqueo` | Jacked at London riots - 8th August | CC BY 3.0 | michele bonechi |
| `disturbios_calle` | Medan-Indonesia omnibus law riots | CC BY-SA 4.0 | (ver página del fichero) |
| `accidente_laboral` | Las Caídas Cuestan - La Historia de un Safety Man | Dominio público | NIOSH (gobierno de EE. UU.) |

## Obligaciones de atribución

Las escenas CC BY y CC BY-SA exigen citar autor y licencia, cosa que hace esta
tabla. Las derivadas de CC BY-SA (los recortes normalizados que sirve la cámara)
heredan CC BY-SA si se distribuyeran; **no se distribuyen**: viven en el disco de
la instancia de pruebas y en un bucket privado, y solo se usan para evaluar el
sistema de detección.

## Criterio de selección

Se descartó material que, siendo de licencia libre, mostraba a personas
identificables en contextos delictivos o de violencia grave contra ellas —por
ejemplo grabaciones policiales de EE. UU. en dominio público—. Para probar
detección de robo bastan las escenas de disturbios, sin exponer a víctimas
concretas.

## Cómo reproducir la verificación

```bash
python3 buscar_clips.py "termino de busqueda"
```

Marca `LIBRE` cada resultado cuya licencia esté entre las aceptadas. La lista
definitiva de escenas, con su URL, offset y duración, está en el array `ESCENAS`
de `camera_userdata.sh`.
