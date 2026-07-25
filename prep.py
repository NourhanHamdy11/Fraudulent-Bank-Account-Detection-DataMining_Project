"""Stage A: load, sample, split, preprocess; cache arrays + preprocessor to disk."""
import json, pickle, gc
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

SEED = 42; np.random.seed(SEED)
PATH = "/mnt/user-data/uploads/Base.csv"
MOD = "/home/claude/proj/models"; ART = "/home/claude/proj/artifacts"
TARGET = "fraud_bool"; SAMPLE_N = 50000

df = pd.read_csv(PATH).drop(columns=["device_fraud_count"])
frac = SAMPLE_N/len(df)
parts = [g.sample(int(round(len(g)*frac)), random_state=SEED) for _, g in df.groupby(TARGET)]
df_s = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
del df, parts; gc.collect()
y = df_s[TARGET].values; X = df_s.drop(columns=[TARGET])

CAT_COLS = ["payment_type","employment_status","housing_status","source","device_os"]
BIN_COLS = ["email_is_free","phone_home_valid","phone_mobile_valid",
            "has_other_cards","foreign_request","keep_alive_session"]
SENTINEL = ["prev_address_months_count","current_address_months_count","bank_months_count",
            "session_length_in_minutes","device_distinct_emails_8w","credit_risk_score"]
NUM_COLS = [c for c in X.columns if c not in CAT_COLS+BIN_COLS]

def add_flags(d):
    d = d.copy()
    for c in SENTINEL:
        d.loc[d[c] == -1, c] = np.nan
    d["flag_prev_addr_missing"] = d["prev_address_months_count"].isna().astype(int)
    d["flag_bank_months_missing"] = d["bank_months_count"].isna().astype(int)
    return d
X = add_flags(X)
BIN_COLS += ["flag_prev_addr_missing","flag_bank_months_missing"]
NUM_COLS = [c for c in NUM_COLS if c in X.columns]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num_pipe, NUM_COLS),
                         ("cat", cat_pipe, CAT_COLS),
                         ("bin", "passthrough", BIN_COLS)], remainder="drop")
pre.fit(X_tr)
Xtr_t = pre.transform(X_tr).astype(np.float32)
Xte_t = pre.transform(X_te).astype(np.float32)
feat_names = (NUM_COLS +
              list(pre.named_transformers_["cat"]["onehot"].get_feature_names_out(CAT_COLS)) +
              BIN_COLS)
pickle.dump(pre, open(f"{MOD}/preprocessor.pkl","wb"))

# SMOTE (for KNN only)
def smote(Xa, ya, k=5, seed=SEED):
    rng = np.random.RandomState(seed); Xa = np.asarray(Xa, np.float32); ya = np.asarray(ya)
    n_maj=(ya==0).sum(); n_min=(ya==1).sum(); need=n_maj-n_min
    Xmin=Xa[ya==1]
    nn=NearestNeighbors(n_neighbors=k+1).fit(Xmin); _,idx=nn.kneighbors(Xmin)
    syn=np.empty((need,Xa.shape[1]),np.float32)
    for i in range(need):
        a=rng.randint(len(Xmin)); nb=idx[a,rng.randint(1,k+1)]; g=rng.rand()
        syn[i]=Xmin[a]+g*(Xmin[nb]-Xmin[a])
    Xr=np.vstack([Xa,syn]); yr=np.concatenate([ya,np.ones(need,int)])
    p=rng.permutation(len(yr)); return Xr[p],yr[p]
Xtr_sm, ytr_sm = smote(Xtr_t, y_tr)

np.savez_compressed(f"{ART}/arrays.npz",
    Xtr=Xtr_t, Xte=Xte_t, ytr=y_tr, yte=y_te, Xsm=Xtr_sm, ysm=ytr_sm)
json.dump({"feat_names":feat_names,"n_features":len(feat_names),
           "train_shape":list(Xtr_t.shape),"test_shape":list(Xte_t.shape),
           "train_fraud":int(y_tr.sum()),"test_fraud":int(y_te.sum()),
           "smote_before":{int(k):int(v) for k,v in zip(*np.unique(y_tr,return_counts=True))},
           "smote_after":{int(k):int(v) for k,v in zip(*np.unique(ytr_sm,return_counts=True))},
           "seed":SEED,"sample_n":SAMPLE_N}, open(f"{ART}/prep_meta.json","w"), indent=2)
print("PREP DONE", Xtr_t.shape, Xte_t.shape, "train_fraud", int(y_tr.sum()),
      "test_fraud", int(y_te.sum()), "smote_after", int(ytr_sm.sum()))
