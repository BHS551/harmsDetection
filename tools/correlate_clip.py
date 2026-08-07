import sys, os, glob, json
sys.path.insert(0,'/home/user/harmsDetection/tools')
import numpy as np, cv2, torch, clip
from eval_detection import score_image, PROMPT_MAP
from collections import defaultdict

ann=json.load(open('annotations/instances_val2017.json'))
catname={c['id']:c['name'] for c in ann['categories']}
images={im['id']:im for im in ann['images']}
anns_by_img=defaultdict(list)
for a in ann['annotations']: anns_by_img[a['image_id']].append(a)
KNIFE=next(c['id'] for c in ann['categories'] if c['name']=='knife')
FOOD={'dining table','fork','spoon','bowl','cup','wine glass','bottle','sandwich',
      'pizza','cake','donut','hot dog','broccoli','carrot','banana','apple','orange',
      'sink','refrigerator','oven','microwave','sandwich','sports ball'}

device='cpu'; model,preprocess=clip.load("ViT-B/32",device=device); model.eval()
with torch.no_grad():
    temb=model.encode_text(clip.tokenize(PROMPT_MAP['knife']).to(device)); temb/=temb.norm(dim=-1,keepdim=True)
def iid(p): return int(os.path.splitext(os.path.basename(p))[0])

def feats(path,label):
    i=iid(path); im=images[i]; A=im['width']*im['height']
    cats=set(catname[a['category_id']] for a in anns_by_img[i])
    ks=[a['bbox'][2]*a['bbox'][3] for a in anns_by_img[i] if a['category_id']==KNIFE]
    bgr=cv2.imread(path); g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    return dict(
        label=label,
        knife_size=(max(ks)/A if ks else 0.0),
        n_objects=len(anns_by_img[i]),
        has_food=1 if (cats & FOOD) else 0,
        bright=g.mean(), contrast=g.std(), sat=hsv[:,:,1].mean(),
        edge=cv2.Canny(g,100,200).mean())

rows=[]
for sub,label in [('pos',1),('neg_hard',0),('neg_easy',0)]:
    for f in sorted(glob.glob(f'knife_eval/{sub}/*.jpg')):
        s=score_image(f,model,preprocess,temb,device,'sliding')
        if s is None: continue
        d=feats(f,label); d['clip']=s; rows.append(d)

clip_s=np.array([r['clip'] for r in rows])
def corr(key):
    x=np.array([r[key] for r in rows], float)
    if x.std()==0: return 0.0
    return np.corrcoef(x, clip_s)[0,1]

print(f"n={len(rows)}  (140 pos / 140 neg)")
print("\n=== Correlación del SCORE de CLIP con cada variable ===")
print(f"  ¿hay cuchillo? (label real)      r = {corr('label'):+.2f}   <- lo que DEBERÍA predecir")
print(f"  contexto de comida/cocina        r = {corr('has_food'):+.2f}   <- lo que REALMENTE predice")
print(f"  nº de objetos en escena          r = {corr('n_objects'):+.2f}")
print(f"  tamaño del cuchillo              r = {corr('knife_size'):+.2f}")
print(f"  brillo                           r = {corr('bright'):+.2f}")
print(f"  contraste                        r = {corr('contrast'):+.2f}")
print(f"  saturación                       r = {corr('sat'):+.2f}")
print(f"  detalle/bordes                   r = {corr('edge'):+.2f}")

# prueba directa: score medio segun contexto de comida, con y sin cuchillo
def mean_where(f): 
    v=[r['clip'] for r in rows if f(r)]; return (np.mean(v), len(v))
print("\n=== Score medio de CLIP por grupo ===")
for lbl,cond in [
    ("SIN cuchillo y SIN comida", lambda r: r['label']==0 and r['has_food']==0),
    ("SIN cuchillo y CON comida", lambda r: r['label']==0 and r['has_food']==1),
    ("CON cuchillo y SIN comida", lambda r: r['label']==1 and r['has_food']==0),
    ("CON cuchillo y CON comida", lambda r: r['label']==1 and r['has_food']==1),
]:
    m,n=mean_where(cond); print(f"  {lbl:32} score {m:.3f}  (n={n})")
