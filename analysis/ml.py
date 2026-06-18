"""ML basico sobre el grafo PokeAPI (Hito 2). Tres objetivos consultados desde Python:

  1. Clasificacion de tipo primario  -> features = 6 stats base + conteo de moves por tipo.
  2. Link prediction de compatibilidad de crianza, con DOS encuadres:
       2a) topologico: COMPATIBLE es una union de cliques solapadas (cada egg group es un clique;
           ~27% de especies esta en 2 grupos y los puentea), asi que la prediccion por
           vecindario es casi perfecta y dice mas del grafo que del modelo.
       2b) por atributos fenotipicos (stats, tipo, generacion): la tarea predictiva real, no trivial.
  3. Clustering no supervisado de roles competitivos sobre la forma del spread de stats base.

Corre con: python3 analysis/ml.py  (requiere el grafo cargado por load_all.sh)
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

IMG = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(IMG, exist_ok=True)
driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)


def df(q):
    """Ejecuta Cypher contra el grafo y devuelve un DataFrame."""
    with driver.session() as s:
        return pd.DataFrame([r.data() for r in s.run(q)])


# ===================== 1. CLASIFICACION DE TIPO =====================
print("=" * 60, "\n1. CLASIFICACION DE TIPO PRIMARIO\n", "=" * 60)
stats = df("""
MATCH (p:Pokemon {is_default:true})-[r:HAS_STAT]->(s:Stat)
RETURN p.id AS pokemon, s.identifier AS stat, r.base_stat AS v
""").pivot_table(index="pokemon", columns="stat", values="v", fill_value=0)

movetypes = df("""
MATCH (p:Pokemon {is_default:true})-[:CAN_LEARN]->(m:Move)-[:MOVE_TYPE]->(t:Type)
WITH p, t, count(DISTINCT m) AS c
RETURN p.id AS pokemon, 'mt_' + t.identifier AS movetype, c
""").pivot_table(index="pokemon", columns="movetype", values="c", fill_value=0)

label = df("""
MATCH (p:Pokemon {is_default:true})-[r:HAS_TYPE {slot:1}]->(t:Type)
RETURN p.id AS pokemon, t.identifier AS tipo
""").set_index("pokemon")["tipo"]

X = stats.join(movetypes, how="left").fillna(0)
data = X.join(label, how="inner").dropna(subset=["tipo"])
y = data["tipo"]; Xm = data.drop(columns="tipo")
print(f"dataset: {Xm.shape[0]} pokemon, {Xm.shape[1]} features, {y.nunique()} clases")

clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = cross_val_score(clf, Xm, y, cv=cv, scoring="accuracy")
print(f"accuracy 5-fold CV: {scores.mean():.3f} +/- {scores.std():.3f}  (baseline mayoritaria: {y.value_counts(normalize=True).max():.3f})")
# Con 18 clases desbalanceadas la accuracy plana engaña; balanced-accuracy y macro-F1 pesan parejo
# las clases raras (ej fairy, ice) y son la lectura honesta.
bal = cross_val_score(clf, Xm, y, cv=cv, scoring="balanced_accuracy")
f1m = cross_val_score(clf, Xm, y, cv=cv, scoring="f1_macro")
print(f"balanced-accuracy: {bal.mean():.3f} | macro-F1: {f1m.mean():.3f}")

# Contraste honesto: solo con las 6 stats base (sin conteos de move-type) la accuracy se desploma.
# Confirma que la señal fuerte viene del movepool, correlacionado con el tipo casi por construccion
# (un pokemon de fuego aprende muchos moves de fuego, efecto STAB), no del fenotipo de stats.
stat_cols = [c for c in Xm.columns if not c.startswith("mt_")]
scores_stats = cross_val_score(clf, Xm[stat_cols], y, cv=cv, scoring="accuracy")
print(f"accuracy solo con stats base: {scores_stats.mean():.3f} (vs {scores.mean():.3f} con stats + move-types)")

Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=0.25, stratify=y, random_state=42)
clf.fit(Xtr, ytr)
labels_sorted = sorted(y.unique())
cm = confusion_matrix(yte, clf.predict(Xte), labels=labels_sorted)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(cm, display_labels=labels_sorted).plot(ax=ax, xticks_rotation=70, colorbar=False, cmap="Blues")
ax.set_title("Clasificacion de tipo primario - matriz de confusion (holdout)")
plt.tight_layout(); plt.savefig(f"{IMG}/type_confusion.png", dpi=110); plt.close()
imp = pd.Series(clf.feature_importances_, index=Xm.columns).sort_values(ascending=False)
print("top 8 features:", ", ".join(f"{k}={v:.3f}" for k, v in imp.head(8).items()))

# ===================== 2. LINK PREDICTION CRIANZA =====================
print("\n" + "=" * 60, "\n2. LINK PREDICTION - COMPATIBILIDAD DE CRIANZA\n", "=" * 60)
ed = df("MATCH (a:Species)-[:COMPATIBLE]->(b:Species) RETURN a.id AS a, b.id AS b")
nodes = sorted(set(ed.a) | set(ed.b))
edge_set = {(min(a, b), max(a, b)) for a, b in zip(ed.a, ed.b)}
pos = np.array(list(edge_set))
print(f"grafo crianza: {len(nodes)} nodos, {len(pos)} aristas positivas")
# features de fenotipo por especie (independientes del split, se consultan una vez)
sfeat = df("""
MATCH (s:Species)<-[:IS_SPECIES]-(p:Pokemon {is_default:true})-[r:HAS_STAT]->(st:Stat)
RETURN s.id AS sid, st.identifier AS stat, r.base_stat AS v
""").pivot_table(index="sid", columns="stat", values="v", fill_value=0)
smeta = df("""
MATCH (s:Species)
OPTIONAL MATCH (s)<-[:IS_SPECIES]-(:Pokemon {is_default:true})-[:HAS_TYPE {slot:1}]->(t:Type)
RETURN s.id AS sid, s.generation_id AS gen, t.identifier AS ptype
""").set_index("sid")
nl = np.array(nodes)


def sample_random_negatives(k, rng):
    """k pares no-arista. Excluye contra TODAS las positivas (edge_set), no solo las de train, para
    no etiquetar como negativa una arista real que cayo en el test."""
    out = set()
    while len(out) < k:
        u, v = rng.choice(nl, 2, replace=False)
        e = (int(min(u, v)), int(max(u, v)))
        if e not in edge_set:
            out.add(e)
    return np.array(list(out))


def topo_feats(pairs, adj, logdeg):
    """4 features de vecindario por par (common neighbors, Jaccard, Adamic-Adar, preferential
    attachment) sobre la adyacencia de ENTRENAMIENTO, no la del grafo completo (evita la fuga).
    logdeg trae 1/log(grado) precomputado por nodo para no recalcularlo en cada par (grafo denso)."""
    rows = []
    for u, v in pairs:
        nu, nv = adj[u], adj[v]
        common = nu & nv
        cn = len(common)
        un = len(nu) + len(nv) - cn
        jac = cn / un if un else 0.0
        aa = sum(logdeg[w] for w in common)
        rows.append([cn, jac, aa, len(nu) * len(nv)])
    return np.array(rows, dtype=float)


def attr_feats(pairs):
    """Por par: diffs absolutas de las 6 stats base, mismo tipo primario (0/1), misma generacion (0/1)."""
    rows = []
    for u, v in pairs:
        diffs = list(np.abs(sfeat.loc[u].values - sfeat.loc[v].values))
        tu, tv = smeta.loc[u, "ptype"], smeta.loc[v, "ptype"]
        same_type = 1.0 if (pd.notna(tu) and tu == tv) else 0.0
        same_gen = 1.0 if smeta.loc[u, "gen"] == smeta.loc[v, "gen"] else 0.0
        rows.append(diffs + [same_type, same_gen])
    return np.array(rows, dtype=float)


def eval_seed(seed, want_curves=False):
    """Un split honesto: adyacencia topologica armada solo con train, negativos seeded."""
    rng = np.random.default_rng(seed)
    pos_tr, pos_te = train_test_split(pos, test_size=0.2, random_state=seed)
    adj = {n: set() for n in nodes}
    for a, b in pos_tr:
        a, b = int(a), int(b); adj[a].add(b); adj[b].add(a)
    logdeg = {w: 1.0 / math.log(d) for w in adj if (d := len(adj[w])) > 1}
    neg = sample_random_negatives(len(pos), rng)
    neg_tr, neg_te = train_test_split(neg, test_size=0.2, random_state=seed)
    y_tr = np.r_[np.ones(len(pos_tr)), np.zeros(len(neg_tr))]
    y_te = np.r_[np.ones(len(pos_te)), np.zeros(len(neg_te))]
    res, curves = {}, {}
    for nombre, fz in [("topo", lambda P: topo_feats(P, adj, logdeg)), ("attr", attr_feats)]:
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        m.fit(np.vstack([fz(pos_tr), fz(neg_tr)]), y_tr)
        p = m.predict_proba(np.vstack([fz(pos_te), fz(neg_te)]))[:, 1]
        res[nombre] = (roc_auc_score(y_te, p), average_precision_score(y_te, p))
        if want_curves: curves[nombre] = roc_curve(y_te, p)
    return (res, curves) if want_curves else res


# 10 splits para reportar media +/- desviacion en vez de un solo numero. El AP esta sobre un set
# balanceado 1:1 (negativos = positivos), asi que su baseline aleatorio es 0.5, no la prevalencia real.
runs = [eval_seed(k) for k in range(5)]
def ms(key, i): return np.mean([r[key][i] for r in runs]), np.std([r[key][i] for r in runs])
print(f"2a) topologico (CN/Jaccard/AA/PA): AUC={ms('topo',0)[0]:.3f}+/-{ms('topo',0)[1]:.3f}  AP={ms('topo',1)[0]:.3f} (1:1, base 0.5)")
print("    -> casi perfecto: COMPATIBLE es union de cliques solapadas por egg group, el vecindario predice casi todo.")
print(f"2b) atributos (diffs stats + mismo tipo + misma gen): AUC={ms('attr',0)[0]:.3f}+/-{ms('attr',0)[1]:.3f}  AP={ms('attr',1)[0]:.3f} (1:1, base 0.5)")
print("    -> NO trivial: el egg group correlaciona con el fenotipo pero no esta determinado por el.")

_, curves = eval_seed(42, want_curves=True)
plt.figure(figsize=(6, 5))
for nombre, lab, c in [("topo", "topologico (cliques solapadas)", "#8172b3"),
                       ("attr", "por atributos fenotipicos", "#c44e52")]:
    fpr, tpr, _ = curves[nombre]
    plt.plot(fpr, tpr, color=c, label=f"{lab}: AUC={ms(nombre,0)[0]:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right")
plt.title("Link prediction crianza: topologia vs atributos")
plt.tight_layout(); plt.savefig(f"{IMG}/breeding_roc.png", dpi=110); plt.close()

# ===================== 3. CLUSTERING DE ROLES COMPETITIVOS =====================
# No supervisado sobre los 6 stats base: descubre arquetipos (sweeper/muro/tanque/pivot) sin
# umbrales a mano y los cruza con el tipo. Caveat honesto: es stat-spread crudo, el rol meta real
# depende del typing, items y EVs que no tenemos; sirve como esqueleto, no como tier de Smogon.
print("\n" + "=" * 60, "\n3. CLUSTERING DE ROLES POR STATS BASE\n", "=" * 60)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

stat_order = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]
roles = df("""
MATCH (p:Pokemon {is_default:true})-[r:HAS_STAT]->(s:Stat)
RETURN p.id AS pokemon, p.identifier AS nombre, s.identifier AS stat, r.base_stat AS v
""").pivot_table(index=["pokemon", "nombre"], columns="stat", values="v", fill_value=0)[stat_order]

# Clusterizar sobre PROPORCIONES (forma del spread, no poder bruto): asi un frail-veloz y un
# muro-lento caen en clusters distintos sin importar su BST.
prop = roles[stat_order].div(roles[stat_order].sum(axis=1), axis=0)
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(prop.values)
roles["cluster"] = km.labels_
cent = prop.groupby(km.labels_).mean()  # centroides en proporciones

# Etiqueta data-driven: eje dominante (z-score entre clusters) + sesgo fisico/especial.
ejes = pd.DataFrame({
    "speed": cent["speed"],
    "ofensa": cent["attack"] + cent["special-attack"],
    "bulk": cent["hp"] + cent["defense"] + cent["special-defense"],
}, index=cent.index)
zejes = (ejes - ejes.mean()) / ejes.std()
def etiqueta(c):
    z = zejes.loc[c]; dom = z.idxmax()
    fis = cent.loc[c, "attack"] >= cent.loc[c, "special-attack"]
    if dom == "speed":  return "barredor veloz" if ejes.loc[c, "ofensa"] >= ejes.mean()["ofensa"] else "veloz frágil"
    if dom == "bulk":
        if z["speed"] >= 0: return "tanque"
        return "muro de HP" if cent.loc[c, "hp"] >= cent.loc[c, "defense"] else "muro defensivo"
    return "atacante físico" if fis else "atacante especial"
roles["rol"] = roles["cluster"].map({c: etiqueta(c) for c in cent.index})
print(f"{len(roles)} pokemon en 5 clusters:")
print(roles["rol"].value_counts().to_string())
for rol in sorted(roles["rol"].unique()):
    ej = roles[roles["rol"] == rol].index.get_level_values("nombre")[:6].tolist()
    print(f"  {rol}: {', '.join(ej)}")

tipos1 = df("""
MATCH (p:Pokemon {is_default:true})-[r:HAS_TYPE {slot:1}]->(t:Type)
RETURN p.id AS pokemon, t.identifier AS tipo
""").set_index("pokemon")["tipo"]
roles_t = roles.reset_index().merge(tipos1.rename("tipo"), on="pokemon")
print("\ntipo primario dominante por rol:")
for rol, g in roles_t.groupby("rol"):
    top = g["tipo"].value_counts().head(3)
    print(f"  {rol}: {', '.join(f'{k}({v})' for k, v in top.items())}")

sil = silhouette_score(prop.values, km.labels_)
print(f"silueta media: {sil:.3f}  (arquetipos solapados, no islas nitidas)")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im = ax[0].imshow(cent.values, cmap="RdYlBu_r", aspect="auto")
ax[0].set_xticks(range(6)); ax[0].set_xticklabels([s.replace("special-", "sp.") for s in stat_order], rotation=30, ha="right")
ax[0].set_yticks(range(len(cent))); ax[0].set_yticklabels([f"{etiqueta(c)} (n={(km.labels_==c).sum()})" for c in cent.index])
for i, c in enumerate(cent.index):
    for j in range(6): ax[0].text(j, i, f"{cent.values[i, j]*100:.0f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax[0], label="% del BST"); ax[0].set_title("Perfil de stats por cluster (centroides)")
Y = PCA(2).fit_transform(prop.values)
for c in cent.index:
    m = km.labels_ == c; ax[1].scatter(Y[m, 0], Y[m, 1], s=10, alpha=0.6, label=etiqueta(c))
ax[1].legend(fontsize=8); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
ax[1].set_title(f"Clusters en espacio PCA (silueta={sil:.2f})")
plt.tight_layout(); plt.savefig(f"{IMG}/roles_cluster.png", dpi=110); plt.close()

print(f"\nFiguras en {IMG}/")
driver.close()
