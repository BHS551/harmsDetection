import sys, zipfile, json, random
sys.path.insert(0,'/home/user/harmsDetection/tools')
import numpy as np, cv2, torch, clip
from eval_detection import enhance_frame
from PIL import Image
random.seed(7)

device='cpu'; model,preprocess=clip.load("ViT-B/32",device=device); model.eval()
def et(ps):
    with torch.no_grad():
        e=model.encode_text(clip.tokenize(ps).to(device)); e/=e.norm(dim=-1,keepdim=True)
    return e
WEAPON=et(["a photo of a knife","a photo of a handgun"])
NEG=et(["a photo of a smartphone","a photo of a wallet","a photo of a banknote",
        "a photo of a credit card","a person holding a phone","a close-up of a hand","a person"])

z=zipfile.ZipFile('granada_weapons.zip')
BASE='OD-WeaponDetection-master/Weapons and similar handled objects/Sohas_weapon-Detection'
img_index={}
for n in z.namelist():
    if ('/images/' in n or '/images_test/' in n) and n.lower().endswith('.jpg'):
        img_index[n.split('/')[-1]]=n
lab=json.load(open('sohas_det_labels.json'))
random.shuffle(lab['pos']); random.shuffle(lab['neg'])
items=[(f,1) for f in lab['pos'][:120]]+[(f,0) for f in lab['neg'][:120]]

def tiles(w,h,size,stride):
    xs=list(range(0,max(1,w-size)+1,stride)) or [0]; ys=list(range(0,max(1,h-size)+1,stride)) or [0]
    if xs[-1]!=w-size and w>size: xs.append(w-size)
    if ys[-1]!=h-size and h>size: ys.append(h-size)
    return [(x,y,min(x+size,w),min(y+size,h)) for y in ys for x in xs]

def analyze(fn):
    n=img_index.get(fn)
    if not n: return None
    arr=cv2.imdecode(np.frombuffer(z.read(n),np.uint8),cv2.IMREAD_COLOR)
    if arr is None: return None
    h,w=arr.shape[:2]; scale=896/max(w,h)
    if scale<1: arr=cv2.resize(arr,(int(w*scale),int(h*scale)))
    arr=enhance_frame(arr); H,W=arr.shape[:2]
    pil=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB))
    full=[(0,0,W,H)]
    base=tiles(W,H,256,224)                    # ~ enfoque worker actual (sliding)
    fine=tiles(W,H,192,128)+tiles(W,H,320,224) # SAHI multiescala
    boxes=full+base+fine
    tags=['full']*1+['base']*len(base)+['fine']*len(fine)
    crops=[pil.crop(b) for b in boxes]
    t=torch.stack([preprocess(c) for c in crops]).to(device)
    with torch.no_grad():
        e=model.encode_image(t); e/=e.norm(dim=-1,keepdim=True)
        wk=(e@WEAPON.T).max(dim=1).values.cpu().numpy()
        ng=(e@NEG.T).max(dim=1).values.cpu().numpy()
    tags=np.array(tags)
    return {
        'full_abs': float(wk[tags=='full'].max()),
        'slide_abs': float(wk[(tags=='full')|(tags=='base')].max()),
        'new_contrast': float((wk-ng)[:].max()),   # SAHI(todos)+contrastivo
    }

res={'full_abs':[], 'slide_abs':[], 'new_contrast':[]}; labels=[]
for fn,lb in items:
    a=analyze(fn)
    if a is None: continue
    for k in res: res[k].append(a[k])
    labels.append(lb)
labels=np.array(labels); print("evaluadas:",len(labels),"| pos",int(labels.sum()),"neg",int((labels==0).sum()))

def report(name,scores):
    scores=np.array(scores); pos=scores[labels==1]; neg=scores[labels==0]
    wins=sum((a>b)+0.5*(a==b) for a in pos for b in neg); auc=wins/(len(pos)*len(neg))
    grid=sorted(set(np.round(scores,4)))
    best=None
    for t in grid:
        rec=(pos>=t).mean()
        if rec>=0.95:
            best=(t,rec,(neg>=t).mean())
    line=f"### {name} | AUC {auc:.3f}"
    if best: line+=f" | @recall>=0.95: recall {best[1]:.2f}, FALSAS ALARMAS {best[2]*100:.0f}%"
    else: line+=" | no alcanza recall 0.95"
    print(line)

print("\n===== ESCENA REALISTA (arma ~1% del frame) : ACTUAL vs MEJORAS =====")
report("ACTUAL a) CLIP escena completa, absoluto", res['full_abs'])
report("ACTUAL b) CLIP sliding, absoluto (worker hoy)", res['slide_abs'])
report("NUEVO  ) SAHI multiescala + contrastivo", res['new_contrast'])
