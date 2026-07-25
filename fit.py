"""Stage B: fit ONE model (arg) with grid-search CV on cached arrays.
Saves ablation table, best estimator, test-set probabilities + train time."""
import sys, json, time, pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              StackingClassifier)
import pandas as pd

SEED = 42
ART="/home/claude/proj/artifacts"; TAB="/home/claude/proj/tables"; MOD="/home/claude/proj/models"
d = np.load(f"{ART}/arrays.npz")
Xtr,Xte,ytr,yte = d["Xtr"],d["Xte"],d["ytr"],d["yte"]
Xsm,ysm = d["Xsm"],d["ysm"]
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
SCORING="average_precision"
name = sys.argv[1]

def save_grid(name, gs, dt):
    res=pd.DataFrame(gs.cv_results_)
    keep=[c for c in res.columns if c.startswith("param_")]+["mean_test_score","std_test_score","rank_test_score"]
    tab=res[keep].sort_values("rank_test_score").reset_index(drop=True)
    tab.to_csv(f"{TAB}/ablation_{name}.csv", index=False)
    pickle.dump(gs.best_estimator_, open(f"{MOD}/best_{name}.pkl","wb"))
    s=gs.best_estimator_.predict_proba(Xte)[:,1]
    np.save(f"{ART}/proba_{name}.npy", s)
    json.dump({"name":name,"best_params":{k:str(v) for k,v in gs.best_params_.items()},
               "cv_pr_auc":float(gs.best_score_),"train_time_s":float(dt)},
              open(f"{ART}/fit_{name}.json","w"), indent=2)
    print(f"[{name}] CV PR-AUC={gs.best_score_:.4f} {gs.best_params_} time={dt:.1f}s")

def grid(est, params, X, Y, cv=cv5):
    t=time.time(); gs=GridSearchCV(est,params,scoring=SCORING,cv=cv,n_jobs=1,refit=True); gs.fit(X,Y)
    return gs, time.time()-t

if name=="LogReg":
    gs,dt=grid(LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED),
               {"C":[0.01,0.1,1.0,10.0]}, Xtr,ytr); save_grid(name,gs,dt)

elif name=="DecisionTree":
    gs,dt=grid(DecisionTreeClassifier(class_weight="balanced",random_state=SEED),
               {"max_depth":[3,5,10,None],"min_samples_leaf":[1,20]}, Xtr,ytr); save_grid(name,gs,dt)

elif name=="KNN":
    gs,dt=grid(KNeighborsClassifier(algorithm="brute",n_jobs=1),
               {"n_neighbors":[11,25,51],"weights":["uniform","distance"]}, Xsm,ysm); save_grid(name,gs,dt)

elif name=="SVM":
    # RBF SVM scales ~O(n^2); Platt 'probability=True' adds 5x CV cost -> we avoid it.
    # Grid on stratified ~12k subsample; rank via decision_function (monotone -> AUC/AP exact).
    rs=np.random.RandomState(SEED)
    i0=np.where(ytr==0)[0]; i1=np.where(ytr==1)[0]
    t0=rs.choice(i0,size=min(len(i0),11600),replace=False)
    si=np.concatenate([t0,i1]); rs.shuffle(si)
    gs,dt=grid(SVC(kernel="rbf",class_weight="balanced",random_state=SEED),
               {"C":[0.1,1.0,10.0],"gamma":["scale",0.1]}, Xtr[si],ytr[si],
               cv=StratifiedKFold(3,shuffle=True,random_state=SEED))
    # refit best on a 20k stratified subsample (documented tractability cap)
    r2=np.random.RandomState(SEED+1)
    j0=r2.choice(i0,size=min(len(i0),19500),replace=False)
    ji=np.concatenate([j0,i1]); r2.shuffle(ji)
    t=time.time()
    best=SVC(kernel="rbf",class_weight="balanced",random_state=SEED,
             **gs.best_params_).fit(Xtr[ji],ytr[ji])
    dt2=time.time()-t
    pickle.dump(best,open(f"{MOD}/best_SVM.pkl","wb"))
    raw=best.decision_function(Xte)
    s=(raw-raw.min())/(raw.max()-raw.min()+1e-12)   # min-max -> [0,1] for storage (monotone)
    np.save(f"{ART}/proba_SVM.npy",s.astype(np.float32))
    np.save(f"{ART}/labels_SVM.npy", best.predict(Xte).astype(int))  # native sign threshold
    res=pd.DataFrame(gs.cv_results_)
    keep=[c for c in res.columns if c.startswith("param_")]+["mean_test_score","std_test_score","rank_test_score"]
    res[keep].sort_values("rank_test_score").reset_index(drop=True).to_csv(f"{TAB}/ablation_SVM.csv",index=False)
    json.dump({"name":"SVM","best_params":{k:str(v) for k,v in gs.best_params_.items()},
               "cv_pr_auc":float(gs.best_score_),"train_time_s":float(dt+dt2),
               "note":"decision_function scores (uncalibrated); refit on 20k subsample"},
              open(f"{ART}/fit_SVM.json","w"),indent=2)
    print(f"[SVM] CV PR-AUC={gs.best_score_:.4f} {gs.best_params_} grid={dt:.1f}s refit={dt2:.1f}s")

elif name=="RandomForest":
    gs,dt=grid(RandomForestClassifier(n_estimators=150,class_weight="balanced",random_state=SEED,n_jobs=1),
               {"max_depth":[12,20],"min_samples_leaf":[5,20]}, Xtr,ytr)
    save_grid(name,gs,dt)

elif name=="HistGB":
    gs,dt=grid(HistGradientBoostingClassifier(class_weight="balanced",random_state=SEED,
               early_stopping=True,validation_fraction=0.15),
               {"learning_rate":[0.05,0.1],"max_depth":[None,6],"max_iter":[300,600]}, Xtr,ytr)
    save_grid(name,gs,dt)

elif name=="Stacking":
    rf=pickle.load(open(f"{MOD}/best_RandomForest.pkl","rb"))
    hgb=pickle.load(open(f"{MOD}/best_HistGB.pkl","rb"))
    lr=pickle.load(open(f"{MOD}/best_LogReg.pkl","rb"))
    t=time.time()
    st=StackingClassifier(estimators=[("rf",rf),("hgb",hgb),("lr",lr)],
        final_estimator=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED),
        cv=3,stack_method="predict_proba",n_jobs=1).fit(Xtr,ytr)
    dt=time.time()-t
    pickle.dump(st,open(f"{MOD}/best_Stacking.pkl","wb"))
    s=st.predict_proba(Xte)[:,1]; np.save(f"{ART}/proba_Stacking.npy",s)
    json.dump({"name":"Stacking","best_params":{"bases":"RF+HistGB+LogReg","meta":"LogReg","cv":3},
               "cv_pr_auc":None,"train_time_s":float(dt)}, open(f"{ART}/fit_Stacking.json","w"),indent=2)
    print(f"[Stacking] fit time={dt:.1f}s")
print("OK", name)
