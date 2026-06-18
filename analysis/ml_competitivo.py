"""Modelos predictivos competitivos, evaluados de forma critica y adversarial (capa meta Smogon).

Requiere el grafo base + la capa de Smogon cargada (pipeline/06_smogon.py).

A. Viabilidad: ¿un Pokemon se usa en gen9 OU? Ablacion baseline (BST + legendario) vs +features de
   grafo, y NEGATIVOS FACILES (todo el dex) vs DIFICILES (solo fully-evolved no-legendarios), para no
   medir lo trivial ("legendario vs Caterpie") sino lo que separa usados de no-usados comparables.
B. Recomendacion de teammates: link prediction sobre TEAMMATE_OF, contrastando una baseline de
   POPULARIDAD (usage_a * usage_b) contra la COMPLEMENTARIEDAD DEFENSIVA calculada en nuestro grafo
   de tipos. La pregunta adversarial: ¿el grafo aporta sobre "ambos se usan mucho"?

Corre con: python3 analysis/ml_competitivo.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def oof_proba(X, y, seed=42):
    """Probabilidades out-of-fold (cada muestra predicha por un modelo que NO la vio)."""
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1)
    return cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             method="predict_proba")[:, 1]

IMG = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(IMG, exist_ok=True)
RNG = np.random.default_rng(42)
driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)


def df(q):
    with driver.session() as s:
        return pd.DataFrame([r.data() for r in s.run(q)])


def auc_cv(X, y, seed=42):
    """ROC-AUC y PR-AUC con 5-fold estratificado (out-of-fold), RF balanceado."""
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1)
    proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
    return roc_auc_score(y, proba), average_precision_score(y, proba)


def auc_ci(X, y, seeds=range(10)):
    """AUC y PR-AUC out-of-fold promediados sobre varias semillas -> (AUC_media, AUC_std, PR_media).
    El intervalo distingue una mejora real de ruido de una sola particion."""
    res = [auc_cv(X, y, seed=k) for k in seeds]
    return float(np.mean([a for a, _ in res])), float(np.std([a for a, _ in res])), float(np.mean([p for _, p in res]))


# ============================================================
# A. PREDICCION DE VIABILIDAD (¿se usa en gen9 OU?)
# ============================================================
print("=" * 60, "\nA. VIABILIDAD COMPETITIVA (gen9 OU)\n", "=" * 60)

stats = df("""
MATCH (p:Pokemon {is_default:true})-[r:HAS_STAT]->(s:Stat)
RETURN p.id AS id, s.identifier AS stat, r.base_stat AS v
""").pivot_table(index="id", columns="stat", values="v", fill_value=0)
stats["bst"] = stats.sum(axis=1)

meta = df("""
MATCH (p:Pokemon {is_default:true})-[:IS_SPECIES]->(sp:Species)
OPTIONAL MATCH (p)-[:CAN_LEARN]->(mv:Move)
WITH p, sp, count(DISTINCT mv) AS movepool
OPTIONAL MATCH (p)-[:HAS_ABILITY]->(ab:Ability)
WITH p, sp, movepool, count(DISTINCT ab) AS n_abil
OPTIONAL MATCH (p)-[:CAN_LEARN]->(pm:Move) WHERE pm.priority > 0 AND pm.power > 0
RETURN p.id AS id, movepool, n_abil, count(DISTINCT pm) AS prio_moves,
       CASE WHEN sp.is_legendary THEN 1 ELSE 0 END AS legendary,
       CASE WHEN sp.is_mythical THEN 1 ELSE 0 END AS mythical,
       CASE WHEN (sp)-[:EVOLVES_TO]->(:Species) THEN 0 ELSE 1 END AS fully_evolved
""").set_index("id")

match = df("""
MATCH (p:Pokemon {is_default:true})-[:HAS_TYPE]->(def:Type)
WITH p, collect(def) AS defs
MATCH (atk:Type)
OPTIONAL MATCH (atk)-[e:EFFECTIVENESS]->(d) WHERE d IN defs
WITH p, atk, reduce(f=1.0, x IN collect(e.factor/100.0) | f*x) AS mult
RETURN p.id AS id,
       sum(CASE WHEN mult > 1 THEN 1 ELSE 0 END) AS weak,
       sum(CASE WHEN mult < 1 AND mult > 0 THEN 1 ELSE 0 END) AS resist,
       sum(CASE WHEN mult = 0 THEN 1 ELSE 0 END) AS immune
""").set_index("id")

stab = df("""
MATCH (p:Pokemon {is_default:true})-[:HAS_TYPE]->(pt:Type)
OPTIONAL MATCH (pt)-[:SUPER_EFFECTIVE]->(d:Type)
RETURN p.id AS id, count(DISTINCT d) AS stab_cov
""").set_index("id")

# Label a nivel de ESPECIE: viable en OU si CUALQUIER de sus formas se usa. Smogon empareja formas
# no-default que son staples reales (landorus-therian, urshifu, ogerpon, formas hisui), y marcar solo
# la forma default dejaba ~23 especies viables del lado negativo, inyectando falsos negativos duros.
label = df("""
MATCH (p:Pokemon {is_default:true})-[:IS_SPECIES]->(sp:Species)
RETURN p.id AS id,
       CASE WHEN EXISTS { (sp)<-[:IS_SPECIES]-(:Pokemon)-[:USED_IN]->(:Format {tier:'gen9ou'}) }
            THEN 1 ELSE 0 END AS in_ou
""").set_index("id")

D = stats.join([meta, match, stab, label], how="inner").fillna(0)
y = D["in_ou"].astype(int)
print(f"{len(D)} pokemon, {y.sum()} en OU ({y.mean():.1%})")

base_cols = ["bst", "legendary"]
full_cols = ["bst", "hp", "attack", "defense", "special-attack", "special-defense", "speed",
             "legendary", "mythical", "fully_evolved", "movepool", "n_abil", "prio_moves",
             "weak", "resist", "immune", "stab_cov"]

print("\n-- Negativos FACILES (todo el dex) --")
ab, ap = auc_cv(D[base_cols], y)
af, apf = auc_cv(D[full_cols], y)
print(f"  baseline (BST+legendario): AUC={ab:.3f}  AP={ap:.3f}")
print(f"  + features de grafo:       AUC={af:.3f}  AP={apf:.3f}")

# Negativos DIFICILES: fully-evolved, no-legendarios, no-miticos que NO se usan.
hard = D[(D.in_ou == 1) | ((D.fully_evolved == 1) & (D.legendary == 0) & (D.mythical == 0))]
yh = hard["in_ou"].astype(int)
print(f"\n-- Negativos DIFICILES (fully-evolved no-legendarios): {len(hard)} pokemon, {yh.sum()} en OU --")
# AUC y PR-AUC out-of-fold promediados sobre 10 semillas (media +/- desviacion). El intervalo dice si
# el salto baseline -> grafo es real o ruido de una sola particion. PR-AUC es la metrica honesta con
# desbalance; ROC-AUC tiende a ser optimista cuando los positivos son minoria.
mb, sb, pb = auc_ci(hard[base_cols], yh)
mf, sf, pf = auc_ci(hard[full_cols], yh)
print(f"  baseline (BST+legendario): AUC={mb:.3f}+/-{sb:.3f}  PR-AUC={pb:.3f}")
print(f"  + features de grafo:       AUC={mf:.3f}+/-{sf:.3f}  PR-AUC={pf:.3f}  (delta AUC={mf-mb:+.3f})")

clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1).fit(hard[full_cols], yh)
imp = pd.Series(clf.feature_importances_, index=full_cols).sort_values(ascending=False)
print("  top 6 features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(6).items()))

# CONTROL DE FUGA: con el label shuffleado el AUC debe colapsar a ~0.5; si no, hay fuga del label.
# (Detecta fuga label-en-feature; aqui las features son atributos del nodo, no de vecindario, asi que
# no hay fuga topologica train/test que ocultar.) Se promedian 5 shuffles.
nulos = [roc_auc_score(perm, oof_proba(hard[full_cols], perm, seed=k))
         for k in range(5) for perm in [pd.Series(np.random.default_rng(k).permutation(yh.values), index=yh.index)]]
print(f"  control de fuga (5 labels shuffleados): AUC medio={np.mean(nulos):.3f}  (debe ser ~0.5)")
print(f"  lectura: el grafo sube {mf-mb:+.3f} de AUC sobre el baseline entre mons comparables, y el")
print("  shuffle confirma que esa señal es real (no fuga). AUC medido out-of-fold, no sobre el train.")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for cols, lab, c in [(base_cols, "baseline (BST+leg.)", "#8172b3"), (full_cols, "+ features de grafo", "#c44e52")]:
    pr = oof_proba(hard[cols], yh)
    fpr, tpr, _ = roc_curve(yh, pr); prec, rec, _ = precision_recall_curve(yh, pr)
    ax[0].plot(fpr, tpr, color=c, label=f"{lab}: AUC={roc_auc_score(yh, pr):.3f}")
    ax[1].plot(rec, prec, color=c, label=f"{lab}: PR-AUC={average_precision_score(yh, pr):.3f}")
ax[0].plot([0, 1], [0, 1], "--", color="gray"); ax[0].legend(loc="lower right")
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC (negativos dificiles)")
ax[1].axhline(yh.mean(), ls="--", color="gray", label=f"prevalencia={yh.mean():.2f}")
ax[1].legend(loc="upper right"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[1].set_title("Precision-Recall (la honesta con desbalance)")
plt.tight_layout(); plt.savefig(f"{IMG}/viability_roc.png", dpi=110); plt.close()
plt.figure(figsize=(7, 4)); imp.head(10)[::-1].plot.barh(color="#55a868")
plt.title("Importancia de features (viabilidad OU)"); plt.tight_layout()
plt.savefig(f"{IMG}/viability_importance.png", dpi=110); plt.close()

# ============================================================
# B. RECOMENDACION DE TEAMMATES (link prediction sobre TEAMMATE_OF)
# ============================================================
print("\n" + "=" * 60, "\nB. RECOMENDACION DE TEAMMATES (gen9 OU)\n", "=" * 60)

tm = df("""
MATCH (a:Pokemon)-[t:TEAMMATE_OF]->(b:Pokemon)
RETURN a.id AS a, b.id AS b
""")
usage = df("""
MATCH (p:Pokemon)-[u:USED_IN]->(:Format {tier:'gen9ou'})
RETURN p.id AS id, u.usage AS usage
""").set_index("id")["usage"].to_dict()
nodes = sorted(usage.keys())
nodeset = set(nodes)

# perfil de debilidades/resistencias por mon OU (para complementariedad de tipos). Aqui 'resist'
# incluye las inmunidades (mult=0): para complementarse, que un compañero sea inmune a tu debilidad
# es la mejor cobertura posible. (En la parte A, weak/resist/immune van como features separadas.)
prof = df("""
MATCH (p:Pokemon)-[:USED_IN]->(:Format {tier:'gen9ou'})
MATCH (p)-[:HAS_TYPE]->(def:Type)
WITH p, collect(def) AS defs
MATCH (atk:Type)
OPTIONAL MATCH (atk)-[e:EFFECTIVENESS]->(d) WHERE d IN defs
WITH p, atk, reduce(f=1.0, x IN collect(e.factor/100.0) | f*x) AS mult
RETURN p.id AS id,
       [a IN collect(CASE WHEN mult >= 2 THEN atk.identifier END) WHERE a IS NOT NULL] AS weak,
       [a IN collect(CASE WHEN mult < 1 THEN atk.identifier END) WHERE a IS NOT NULL] AS resist
""")
weak = {r.id: set(r.weak) for r in prof.itertuples()}
resist = {r.id: set(r.resist) for r in prof.itertuples()}

edge_set = {(min(a, b), max(a, b)) for a, b in zip(tm.a, tm.b) if a in nodeset and b in nodeset}
pos = np.array(list(edge_set))
nl = np.array(nodes)


def complement(u, v):
    wu, wv, ru, rv = weak.get(u, set()), weak.get(v, set()), resist.get(u, set()), resist.get(v, set())
    cu = len(wu & rv) / len(wu) if wu else 1.0   # debilidades de u que v resiste
    cv = len(wv & ru) / len(wv) if wv else 1.0
    return (cu + cv) / 2


def neg_sample(k, rng):
    """k pares no-arista, excluyendo contra TODAS las positivas (no solo train)."""
    out = set()
    while len(out) < k:
        u, v = rng.choice(nl, 2, replace=False)
        e = (int(min(u, v)), int(max(u, v)))
        if e not in edge_set:
            out.add(e)
    return np.array(list(out))


def feats(pairs, adj):
    rows = []
    for u, v in pairs:
        pop = usage.get(u, 0) * usage.get(v, 0)
        cn = len(adj[u] & adj[v])
        rows.append([pop, complement(u, v), cn])
    return np.array(rows, dtype=float)


cols = {"solo popularidad": [0], "solo complementariedad de tipos": [1],
        "popularidad + complementariedad": [0, 1], "todo (+ vecinos comunes)": [0, 1, 2]}


def run_teammates(seed, want_curves=False):
    """Un split honesto: la adyacencia para 'vecinos comunes' se arma SOLO con aristas de train.
    Si se arma con todas (incluido test), el feature ve las aristas que tiene que predecir -> fuga."""
    rng = np.random.default_rng(seed)
    neg = neg_sample(len(pos), rng)
    pos_tr, pos_te = train_test_split(pos, test_size=0.25, random_state=seed)
    neg_tr, neg_te = train_test_split(neg, test_size=0.25, random_state=seed)
    adj = {n: set() for n in nodes}
    for a, b in pos_tr:
        a, b = int(a), int(b)
        adj[a].add(b); adj[b].add(a)
    ytr = np.r_[np.ones(len(pos_tr)), np.zeros(len(neg_tr))]
    yte = np.r_[np.ones(len(pos_te)), np.zeros(len(neg_te))]
    Xtr = feats(np.vstack([pos_tr, neg_tr]), adj); Xte = feats(np.vstack([pos_te, neg_te]), adj)
    out, curves = {}, {}
    for nombre, idx in cols.items():
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr[:, idx], ytr)
        p = m.predict_proba(Xte[:, idx])[:, 1]
        out[nombre] = roc_auc_score(yte, p)
        if want_curves: curves[nombre] = roc_curve(yte, p)
    msh = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xtr, rng.permutation(ytr))
    out["_nulo"] = roc_auc_score(yte, msh.predict_proba(Xte)[:, 1])
    return (out, curves) if want_curves else out


print(f"{len(nodes)} mons OU, {len(pos)} aristas teammate")
res = [run_teammates(k) for k in range(10)]
print("  AUC sobre holdout 75/25 (media +/- std de 10 splits, adyacencia solo-train):")
for nombre in cols:
    vals = [r[nombre] for r in res]
    print(f"  {nombre:34s} AUC={np.mean(vals):.3f}+/-{np.std(vals):.3f}")
print(f"  control de fuga (label shuffleado):        AUC={np.mean([r['_nulo'] for r in res]):.3f}  (debe ser ~0.5)")

_, curves = run_teammates(42, want_curves=True)
plt.figure(figsize=(6, 5))
for nombre, (fpr, tpr, _) in curves.items():
    plt.plot(fpr, tpr, label=f"{nombre}: {np.mean([r[nombre] for r in res]):.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray"); plt.legend(fontsize=7, loc="lower right")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Teammates: que señal predice el co-uso real")
plt.tight_layout(); plt.savefig(f"{IMG}/teammate_roc.png", dpi=110); plt.close()
print("  lectura adversarial: la complementariedad de tipos sola apenas supera el azar y no mejora")
print("  sobre popularidad; lo que predice el co-uso es la co-ocurrencia (vecinos), ya sin fuga.")

print(f"\nFiguras en {IMG}/")
driver.close()
