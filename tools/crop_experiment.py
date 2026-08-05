import sys, os, glob, json
sys.path.insert(0,'/home/user/harmsDetection/tools')
import numpy as np, cv2, torch, clip
from eval_detection import enhance_frame, PROMPT_MAP
from PIL import Image
from collections import defaultdict

ann=json.load(open('annotations/instances_val2017.json'))
catname={c['id']:c['name'] for c in ann['categories']}
anns_by_img=defaultdict(list)
for a in ann['annotations']: anns_by_img[a['image_id']].append(a)
KNIFE=next(c['id'] for c in ann['categories'] if c['name']=='knife')
device='cpu'; model,preprocess=clip.load("ViT-B/32",device=device); model.eval()
with torch.no_grad():
    temb=model.encode_text(clip.tokenize(PROMPT_MAP['knife']).to(device)); temb/=temb.norm(dim=-1,keepdim=True)
def iid(p): return int(os.path.splitext(os.path.basename(p))[0])

def score_crop(path, bbox):
    bgr=cv2.imread(path); H,W=bgr.shape[:2]
    x,y,w,h=bbox
    # padding 40% alrededor del objeto
    px,py=w*0.4,h*0.4
    x1,y1=int(max(0,x-px)),int(max(0,y-py)); x2,y2=int(min(W,x+w+px)),int(min(H,y+h+py))
    crop=bgr[y1:y2, x1:x2]
    if crop.size==0: return 0.0
    crop=enhance_frame(crop)
    pil=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
    t=preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        e=model.encode_image(t); e/=e.norm(dim=-1,keepdim=True)
        return float((e@temb.T).max().item())

def biggest_bbox(i, exclude=None):
    b=[(a['bbox'][2]*a['bbox'][3], a['bbox']) for a in anns_by_img[i] if a['category_id']!=exclude and a['bbox'][2]*a['bbox'][3]>0]
    return max(b)[1] if b else None

scores=[]  # (score, label)
for sub,label in [('pos',1),('neg_hard',0),('neg_easy',0)]:
    for f in sorted(glob.glob(f'knife_eval/{sub}/*.jpg')):
        i=iid(f)
        if label==1:
            ks=[(a['bbox'][2]*a['bbox'][3],a['bbox']) for a in anns_by_img[i] if a['category_id']==KNIFE]
            if not ks: continue
            bbox=max(ks)[1]                      # recorte AL CUCHILLO
        else:
            bbox=biggest_bbox(i)                 # recorte al objeto más grande
            if bbox is None: continue
        scores.append((score_crop(f,bbox), label))

pos=[s for s,y in scores if y==1]; neg=[s for s,y in scores if y==0]
wins=sum((a>b)+0.5*(a==b) for a in pos for b in neg); auc=wins/(len(pos)*len(neg))
print(f"=== CLIP con recorte AISLADO al objeto (ROI perfecto) ===")
print(f"n={len(scores)} | AUC={auc:.3f}")
print(f"score medio  cuchillo aislado={np.mean(pos):.3f}  objeto-no-cuchillo aislado={np.mean(neg):.3f}  (sep {np.mean(pos)-np.mean(neg):+.3f})")
print(f"(referencia: CLIP escena completa AUC 0.64 / sep +0.011 ; YOLO-World AUC 0.87)")
