"""Post-hoc analyses from stored per-image predictions (zero extra training).

Computes, from the committed/downloaded test_predictions.npz files:
  1. Layered vs flat classifier on matched metrics (accuracy AND macro-F1).
  2. Measured clean-subset accuracy (joins the leakage flags from
     07_leakage_audit.py to the layered per-image correctness).
  3. Stage-1 (script routing) error breakdown.
  4. Cross-script homograph confusion (flat model).
  5. Per-class F1 vs class frequency.
  6. Per-script classical (HOG 1-NN) vs deep.

Inputs: artifacts_colab/results/main/*/test_predictions.npz (layered + flat),
artifacts_local_full HOG caches + leakage arrays, artifacts_colab/manifest.csv.
Figures 7 and 8 are produced by make_figures.ipynb from the arrays saved here.

    python scripts/08_prediction_analysis.py
"""
import sys, glob; sys.path.insert(0,"src"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, csv
from collections import Counter
from statistics import mean, stdev
from sklearn.metrics import f1_score
MAIN="artifacts_colab/results/main"

def relkey(p):
    p=str(p).replace("\\","/")
    for a in ["data/clean/","content/data/","/data/"]:
        if a in p: p=p.split(a,1)[1]; break
    # cache appended ".png" to non-png originals (a1.jpg -> a1.jpg.png); normalize
    for ext in (".jpg",".jpeg",".bmp",".tif",".tiff",".webp"):
        if p.endswith(ext+".png"): p=p[:-4]; break
    return p

def load_seed(seed):
    d=np.load(f"{MAIN}/resnet18__script_id__sz64__aug-medium__pt__s{seed}/test_predictions.npz",allow_pickle=True)
    scr=[str(x) for x in d["class_names"]]; pred=d["logits"].argmax(1)
    route={relkey(p):(scr[t],scr[pred[i]]) for i,(p,t) in enumerate(zip(d["paths"],d["targets"]))}
    stage2={}
    for f in glob.glob(f"{MAIN}/resnet18__per_script__*__sz64__aug-medium__pt__s{seed}/test_predictions.npz"):
        script=f.split("per_script__")[1].split("__")[0]
        z=np.load(f,allow_pickle=True); cn=[str(x) for x in z["class_names"]]; pr=z["logits"].argmax(1)
        for i,(p,t) in enumerate(zip(z["paths"],z["targets"])):
            stage2[relkey(p)]=(f"{script}/{cn[t]}", cn[pr[i]], script, cn[t])  # (true_label, pred_char, script, true_char)
    return route, stage2

seed_acc=[]; seed_mf1=[]; stage1_err=Counter(); per_img_correct={}
for seed in (0,1,2):
    route,stage2=load_seed(seed)
    yt,yp,corr={},{},{}
    for k,(true_lab,pchar,tscript,tchar) in stage2.items():
        if k not in route: continue
        rt,rp=route[k]; routed=(rp==tscript)
        ok = routed and (pchar==tchar)
        yt[k]=true_lab; yp[k]= f"{tscript}/{pchar}" if routed else f"__MIS_{rp}"; corr[k]=ok
        if not routed: stage1_err[(tscript,rp)]+=1
    keys=list(yt)
    seed_acc.append(mean(corr[k] for k in keys)*100)
    seed_mf1.append(f1_score([yt[k] for k in keys],[yp[k] for k in keys],average="macro",zero_division=0)*100)
    if seed==0: per_img_correct=dict(corr)

print("="*62); print("ITEM 1 — LAYERED vs FLAT, matched metrics @64px"); print("="*62)
print(f"  LAYERED  accuracy {mean(seed_acc):.2f} ± {stdev(seed_acc):.2f}   macro-F1 {mean(seed_mf1):.2f} ± {stdev(seed_mf1):.2f}")
flat=[r for r in csv.DictReader(open("artifacts_kaggle/ablation_results_complete.csv"))
      if r["model"]=="resnet18" and r["image_size"]=="64" and r["augmentation"]=="medium" and r["pretrained"]=="True"]
fa=[float(r["accuracy"])*100 for r in flat]; fm=[float(r["macro_f1"])*100 for r in flat]
print(f"  FLAT     accuracy {mean(fa):.2f} ± {stdev(fa):.2f}   macro-F1 {mean(fm):.2f} ± {stdev(fm):.2f}")
print(f"  Δ (layered-flat): accuracy {mean(seed_acc)-mean(fa):+.2f}   macro-F1 {mean(seed_mf1)-mean(fm):+.2f}")

print("\n"+"="*62); print("ITEM 3 — STAGE-1 routing errors (3 seeds pooled)"); print("="*62)
tot=sum(stage1_err.values())
print(f"  {tot} misroutings / {len(per_img_correct)*3} decisions ({100*tot/(len(per_img_correct)*3):.3f}%)")
for (a,b),n in stage1_err.most_common(10): print(f"    {a:14} -> {b:14} {n}")

print("\n"+"="*62); print("ITEM 2 — MEASURED clean-subset accuracy (layered, seed 0)"); print("="*62)
from aksara.data.dataset import load_split_frame
frame=load_split_frame("artifacts_local_full/splits.csv","artifacts_local_full/manifest.csv")
te=frame[frame.split=="test"].reset_index(drop=True)
maxsim=np.load("artifacts_local_full/leak_hog_maxsim.npy")
leak={relkey(te.iloc[i]["path"]):maxsim[i] for i in range(len(te))}
matched=[(per_img_correct[k],leak[k]) for k in per_img_correct if k in leak]
print(f"  matched {len(matched)}/{len(per_img_correct)} images to leak flags")
for thr in (0.95,0.99):
    lk=[c for c,s in matched if s>=thr]; cl=[c for c,s in matched if 0<=s<thr]
    print(f"  thr {thr}: full {mean(c for c,_ in matched)*100:.2f} | leaked {mean(lk)*100:.2f} (n={len(lk)}) | CLEAN {mean(cl)*100:.2f} (n={len(cl)})")



# ---------- ITEM 5: per-class F1 vs total class count (layered stage-2 @64, seed 0) ----------
import pandas as pd
man=pd.read_csv("artifacts_colab/manifest.csv",encoding="utf-8")
total_count=man.groupby("label").size().to_dict()   # total images per class
f1_by_class={}
for f in glob.glob(f"{MAIN}/resnet18__per_script__*__sz64__aug-medium__pt__s0/test_predictions.npz"):
    script=f.split("per_script__")[1].split("__")[0]
    z=np.load(f,allow_pickle=True); cn=[str(x) for x in z["class_names"]]; pr=z["logits"].argmax(1); tg=z["targets"]
    labels=[f"{script}/{c}" for c in cn]
    per=f1_score(tg,pr,labels=range(len(cn)),average=None,zero_division=0)
    for i,lab in enumerate(labels): f1_by_class[lab]=per[i]
pairs=[(total_count.get(l,0),v*100) for l,v in f1_by_class.items() if l in total_count]
cnts=np.array([p[0] for p in pairs]); f1s=np.array([p[1] for p in pairs])
print("="*62); print("ITEM 5 — per-class F1 vs class frequency"); print("="*62)
print(f"  {len(pairs)} classes; count range {cnts.min()}-{cnts.max()}")
r=np.corrcoef(np.log(cnts),f1s)[0,1]
print(f"  correlation(log count, F1) = {r:.3f}")
for lo,hi in [(0,30),(30,60),(60,120),(120,10000)]:
    m=(cnts>=lo)&(cnts<hi)
    print(f"  count [{lo:4},{hi:5}): {m.sum():3} classes, mean F1 {f1s[m].mean():.2f}, min {f1s[m].min():.1f}")
np.save("artifacts_local_full/item5_counts.npy",cnts); np.save("artifacts_local_full/item5_f1.npy",f1s)

# ---------- ITEM 6: per-script classical (HOG kNN within script) vs deep ----------
print("\n"+"="*62); print("ITEM 6 — per-script classical (HOG 1-NN) vs deep macro-F1"); print("="*62)
from aksara.data.dataset import load_split_frame
frame=load_split_frame("artifacts_local_full/splits.csv","artifacts_local_full/manifest.csv")
tr=frame[frame.split=="train"].reset_index(drop=True); te=frame[frame.split=="test"].reset_index(drop=True)
Xtr=np.load("artifacts_local_full/results/classical/_feat_hog_64_train_69559.npy")
Xte=np.load("artifacts_local_full/results/classical/_feat_hog_64_test_13912.npy")
Xtr/=np.linalg.norm(Xtr,axis=1,keepdims=True)+1e-8; Xte/=np.linalg.norm(Xte,axis=1,keepdims=True)+1e-8
# deep per-script macro-F1 from per_script result.json (mean 3 seeds)
deep=defaultdict(list)
import json
for f in glob.glob(f"{MAIN}/resnet18__per_script__*/result.json"):
    d=json.load(open(f)); deep[d["experiment"]["script_filter"]].append(d["test_metrics"]["macro_f1"]*100)
print(f"  {'script':14}{'classical F1':>13}{'deep F1':>10}{'gap':>8}")
rows=[]
for s in sorted(set(tr.script)):
    trm=tr.script==s; tem=te.script==s
    Xa,ya=Xtr[trm.values],tr.character[trm].values; Xb,yb=Xte[tem.values],te.character[tem].values
    # 1-NN within script
    pred=[ya[np.argmax(Xa@Xb[i])] for i in range(len(Xb))]
    from sklearn.metrics import f1_score as f1
    cf=f1(yb,pred,average="macro",zero_division=0)*100
    df=np.mean(deep[s]) if deep.get(s) else float("nan")
    rows.append((s,cf,df,df-cf))
for s,cf,df,gp in sorted(rows,key=lambda x:x[3]):
    print(f"  {s:14}{cf:13.2f}{df:10.2f}{gp:8.2f}")

# ---------- ITEM 4: cross-script homograph confusion (flat model, best available) ----------
print("\n"+"="*62); print("ITEM 4 — cross-script homograph confusion (flat sz48, seed 0)"); print("="*62)
z=np.load(f"{MAIN}/resnet18__unified__sz48__aug-medium__pt__s0/test_predictions.npz",allow_pickle=True)
cn=[str(x) for x in z["class_names"]]; pr=z["logits"].argmax(1); tg=z["targets"]
def charname(lab): return lab.split("/",1)[1]  # strip script
name_to_scripts=defaultdict(set)
for lab in cn: name_to_scripts[charname(lab)].add(lab.split("/")[0])
recurring={n for n,s in name_to_scripts.items() if len(s)>1}
print(f"  {len(recurring)} character names recur across scripts (of {len(name_to_scripts)})")
homo=reg=correct=0
for i in range(len(tg)):
    tl=cn[tg[i]]; pl=cn[pr[i]]
    if charname(tl) not in recurring: continue
    if pl==tl: correct+=1
    elif charname(pl)==charname(tl) and pl.split("/")[0]!=tl.split("/")[0]: homo+=1  # same name, wrong script
    else: reg+=1
tot=homo+reg+correct
print(f"  among {tot} test images with a recurring name:")
print(f"    correct              : {correct} ({100*correct/tot:.1f}%)")
print(f"    homograph confusion  : {homo} ({100*homo/tot:.1f}%)  (right character, WRONG script)")
print(f"    other error          : {reg} ({100*reg/tot:.1f}%)")
print(f"  homograph share of all errors on recurring names: {100*homo/(homo+reg):.1f}%")
