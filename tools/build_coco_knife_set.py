#!/usr/bin/env python3
"""
Arma un set etiquetado de cuchillos desde COCO val2017 para eval_detection.py.
No requiere cámaras: descarga imágenes reales anotadas.

Positivos = imágenes con "knife". Negativos = sin cuchillo, separados en:
  neg_hard : contienen objetos confundibles (tenedor, cuchara, tijeras, celular…)
  neg_easy : escenas sin cuchillo ni confundibles

Requisitos: annotations/instances_val2017.json (de
http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

Uso: python3 build_coco_knife_set.py --out knife_eval --n 140
"""
import argparse, json, os, random, urllib.request
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="annotations/instances_val2017.json")
ap.add_argument("--out", default="knife_eval")
ap.add_argument("--n", type=int, default=140)
args = ap.parse_args()
random.seed(42)

ann = json.load(open(args.ann))
cats = {c["name"]: c["id"] for c in ann["categories"]}
KNIFE = cats["knife"]
HARD = [cats[n] for n in ["fork", "spoon", "scissors", "cell phone", "remote",
                          "toothbrush", "wine glass", "cup"] if n in cats]
img_cats = defaultdict(set)
for a in ann["annotations"]:
    img_cats[a["image_id"]].add(a["category_id"])
images = {im["id"]: im for im in ann["images"]}

pos = [i for i, cs in img_cats.items() if KNIFE in cs]
hard = [i for i, cs in img_cats.items() if KNIFE not in cs and (cs & set(HARD))]
easy = [i for i, cs in img_cats.items() if KNIFE not in cs and not (cs & set(HARD))]
for lst in (pos, hard, easy):
    random.shuffle(lst)

def fetch(ids, folder):
    os.makedirs(folder, exist_ok=True)
    ok = 0
    for iid in ids:
        try:
            urllib.request.urlretrieve(images[iid]["coco_url"], f'{folder}/{images[iid]["file_name"]}')
            ok += 1
        except Exception:
            pass
    return ok

print("pos:", fetch(pos[: args.n], f"{args.out}/pos"))
print("neg_hard:", fetch(hard[: args.n // 2], f"{args.out}/neg_hard"))
print("neg_easy:", fetch(easy[: args.n // 2], f"{args.out}/neg_easy"))
