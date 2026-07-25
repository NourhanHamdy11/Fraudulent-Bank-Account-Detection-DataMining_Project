"""Stage C: evaluate all models on the held-out test set using cached probabilities.
Produces comparison table (CSV+JSON) and ROC/PR/confusion/bar figures."""
import json, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

ART="/home/claude/proj/artifacts"; TAB="/home/claude/proj/tables"; FIG="/home/claude/proj/figures"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":130,"font.size":10})
d=np.load(f"{ART}/arrays.npz"); yte=d["yte"]
meta=json.load(open(f"{ART}/prep_meta.json"))

MODELS=["LogReg","DecisionTree","KNN","SVM","RandomForest","HistGB","Stacking"]
def tpr_at_fpr(y,s,tf=0.05):
    fpr,tpr,_=roc_curve(y,s); i=max(np.searchsorted(fpr,tf,side="right")-1,0); return float(tpr[i])

rows=[]; roc_data={}; pr_data={}; cms={}
for name in MODELS:
    s=np.load(f"{ART}/proba_{name}.npy")
    # SVM has native-sign labels; others use 0.5 threshold on calibrated proba
    lab_path=f"{ART}/labels_{name}.npy"
    yp=np.load(lab_path) if os.path.exists(lab_path) else (s>=0.5).astype(int)
    tt=json.load(open(f"{ART}/fit_{name}.json")).get("train_time_s",np.nan)
    cv=json.load(open(f"{ART}/fit_{name}.json")).get("cv_pr_auc",None)
    rows.append({"Model":name,
        "Accuracy":accuracy_score(yte,yp),
        "Precision":precision_score(yte,yp,zero_division=0),
        "Recall":recall_score(yte,yp,zero_division=0),
        "F1":f1_score(yte,yp,zero_division=0),
        "ROC_AUC":roc_auc_score(yte,s),
        "PR_AUC":average_precision_score(yte,s),
        "Recall@5%FPR":tpr_at_fpr(yte,s,0.05),
        "CV_PR_AUC":cv if cv is not None else np.nan,
        "TrainTime_s":tt})
    fpr,tpr,_=roc_curve(yte,s); roc_data[name]=(fpr,tpr)
    rec,prec,_=precision_recall_curve(yte,s); pr_data[name]=(prec,rec)
    cms[name]=confusion_matrix(yte,yp)

res=pd.DataFrame(rows).sort_values("PR_AUC",ascending=False).reset_index(drop=True)
res.to_csv(f"{TAB}/model_comparison.csv",index=False)
disp=res.copy()
for c in disp.columns:
    if c!="Model": disp[c]=disp[c].astype(float).round(4)
disp.to_json(f"{ART}/model_comparison.json",orient="records",indent=2)
print(disp.to_string(index=False))

order=res["Model"].tolist()
cmap=dict(zip(order,plt.cm.tab10(np.linspace(0,1,10))[:len(order)]))

# ROC
fig,ax=plt.subplots(figsize=(5.6,4.6))
for n in order:
    fpr,tpr=roc_data[n]
    ax.plot(fpr,tpr,lw=1.6,color=cmap[n],label=f"{n} (AUC={res.set_index('Model').loc[n,'ROC_AUC']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=.8); ax.axvline(0.05,color="grey",ls=":",lw=.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves (held-out test)"); ax.legend(fontsize=7,loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_roc.png"); plt.close(fig)

# PR
fig,ax=plt.subplots(figsize=(5.6,4.6))
for n in order:
    prec,rec=pr_data[n]
    ax.plot(rec,prec,lw=1.6,color=cmap[n],label=f"{n} (AP={res.set_index('Model').loc[n,'PR_AUC']:.3f})")
ax.axhline(yte.mean(),color="grey",ls="--",lw=.8,label=f"baseline={yte.mean():.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves (held-out test)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_pr.png"); plt.close(fig)

# Confusion matrices
nm=len(order); cols=4; rws=int(np.ceil(nm/cols))
fig,axes=plt.subplots(rws,cols,figsize=(3.0*cols,2.8*rws))
for ax,n in zip(axes.ravel(),order):
    cm=cms[n]; ax.imshow(cm,cmap="Blues")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,f"{v:,}",ha="center",va="center",
                color="white" if v>cm.max()/2 else "black",fontsize=8)
    ax.set_title(n,fontsize=9); ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Legit","Fraud"],fontsize=7); ax.set_yticklabels(["Legit","Fraud"],fontsize=7)
    ax.set_xlabel("Predicted",fontsize=7); ax.set_ylabel("Actual",fontsize=7)
for ax in axes.ravel()[nm:]: ax.axis("off")
fig.suptitle("Confusion matrices (held-out test; SVM at native threshold, others @0.5)",y=1.0,fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_confusion.png",bbox_inches="tight"); plt.close(fig)

# Metric bars
fig,ax=plt.subplots(figsize=(7.6,4.0))
metrics=["ROC_AUC","PR_AUC","Recall@5%FPR","F1"]; xp=np.arange(len(order)); w=0.2
for i,m in enumerate(metrics):
    ax.bar(xp+i*w,res.set_index("Model").loc[order,m].values,w,label=m)
ax.set_xticks(xp+1.5*w); ax.set_xticklabels(order,rotation=30,ha="right",fontsize=8)
ax.set_ylabel("Score"); ax.set_title("Model comparison across key metrics"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_metric_bars.png"); plt.close(fig)

print("\nBest by PR-AUC:",order[0])
print("EVAL DONE")
