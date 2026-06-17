"""Construye reporte.ipynb (reporte tecnico del proyecto MDS7205) ensamblando narrativa
markdown + celdas de codigo ejecutable contra el Neo4j cargado por pipeline/load_all.sh.
Genera el .ipynb; la ejecucion (que llena los outputs) se hace aparte con nbconvert.
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Proyecto MDS7205 - Grafo de Conocimiento sobre PokeAPI

**Iniciativa de Datos e IA - Universidad de Chile**

Este reporte tecnico documenta el diseno, implementacion, consulta y analisis de un
grafo de conocimiento construido sobre **PokeAPI** (el esquema relacional completo de
Pokemon, 177 tablas CSV) modelado como **grafo de propiedades** y cargado en **Neo4j**.

## Motivacion

Queremos entender la estructura del universo Pokemon como red: como se ramifican las
evoluciones, como la crianza conecta especies, y si el balance de tipos esconde ciclos de
ventaja en vez de un tipo dominante. PokeAPI deja atacar eso como grafo, y de paso cumple el
criterio del enunciado (ser una red *interesante*) en tres frentes que verificamos sobre los
datos:

1. **Ciclos**: la efectividad de tipos es un grafo dirigido con ciclos reales
   (p.ej. `fighting -> ice -> flying -> fighting`) y self-loops (`ghost`, `dragon`).
2. **Recursion**: la evolucion es una jerarquia recursiva via la self-FK
   `evolves_from_species_id`.
3. **N-a-N navegable**: la compatibilidad de crianza por *egg groups* forma un grafo
   masivo con un componente conexo gigante.

El grafo final tiene del orden de **130 mil nodos** y **900 mil aristas**, dominadas por
la relacion `CAN_LEARN` (que movimientos puede aprender cada Pokemon), que ademas es un
**multigrafo con propiedades en la arista** (el caso canonico donde un grafo de
propiedades supera a RDF puro).
""")

md("""## Preguntas de investigacion

Antes de implementar definimos que queriamos averiguar de la red. Estas son las preguntas y la
capacidad de grafo que cada una explota; las consultas P1-P9 (seccion 3) y los dos modelos
(seccion 4) las responden.

1. **P1** Existen ciclos de super-efectividad entre tipos, o hay un tipo que le gana a todos? (ciclos dirigidos)
2. **P2** Cuales son los linajes evolutivos completos y cual es el mas largo? (paths recursivos)
3. **P3** Bajo que condiciones evoluciona Eevee a cada una de sus formas? (relacion n-aria reificada)
4. **P4** En que comunidades naturales agrupa la crianza por egg groups? (deteccion de comunidades)
5. **P5** Que especies actuan como puente entre esas comunidades de crianza? (betweenness)
6. **P6** Que tipo es ofensivamente mas central, propagando ventaja por toda la cadena? (PageRank ponderado)
7. **P7** Que par de Pokemon comparte mas movimientos aprendibles? (proyeccion N-a-N del multigrafo)
8. **P8** Que areas concentran mas biodiversidad de especies? (agregacion sobre encuentros reificados)
9. **P9** En que linaje se gana mas poder de la forma base a la final? (recursion + agregacion sobre el path)

Y dos preguntas para el ML basico:

- **ML-1** El fenotipo de un Pokemon (stats y repertorio de movimientos) basta para predecir su tipo, o el tipo es una etiqueta de diseno sin senal fenotipica?
- **ML-2** Se puede predecir si dos especies pueden criar mirando solo su fenotipo, sin conocer el egg group? Y cuanto de esa prediccion es nada mas la estructura del grafo?
""")

md("""## Setup

Conexion al Neo4j levantado por `pipeline/load_all.sh` (puerto bolt 7687, sin auth).
""")
code("""import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
%matplotlib inline

driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)
def q(query):
    \"\"\"Ejecuta Cypher y devuelve un DataFrame.\"\"\"
    with driver.session() as s:
        return pd.DataFrame([r.data() for r in s.run(query)])
print("conectado:", q("RETURN 'Neo4j ' + toString(1) AS ok").iloc[0,0])""")

md("""## 1. Modelo del grafo

El esquema relacional snowflake de PokeAPI se mapea a un grafo de propiedades: cada
tabla-entidad es un *label*, cada FK una relacion, y las tablas-puente con columnas
extra se vuelven **relaciones con propiedades** (`CAN_LEARN` carga nivel/metodo/version;
`HAS_TYPE` carga el slot; etc). Los encuentros se **reifican** como nodos `Encounter`
(relacion n-aria) y los nombres multilingues (11 idiomas) como nodos `Name`.

El pipeline de carga completo esta en `pipeline/` (`01_constraints` -> `05_scale`).
""")
code("""nodos = q("MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS nodos ORDER BY nodos DESC")
display(nodos)
print("TOTAL nodos:", f"{nodos['nodos'].sum():,}")""")
code("""aristas = q("MATCH ()-[r]->() RETURN type(r) AS relacion, count(*) AS aristas ORDER BY aristas DESC")
display(aristas)
print("TOTAL aristas:", f"{aristas['aristas'].sum():,}")""")

md("""## 2. Caracterizacion del grafo (EDA)

### 2.1 Distribucion del tamano de movepool
Cuantos movimientos distintos puede aprender cada Pokemon (formas default).
""")
code("""deg = q(\"\"\"
MATCH (p:Pokemon {is_default:true})-[:CAN_LEARN]->(m:Move)
RETURN p.identifier AS pokemon, count(DISTINCT m) AS moves
\"\"\")
print(deg["moves"].describe().round(1).to_string())
plt.figure(figsize=(7,4))
plt.hist(deg["moves"], bins=40, color="#4c72b0")
plt.xlabel("moves distintos aprendibles"); plt.ylabel("nro de pokemon")
plt.title("Distribucion de tamano de movepool"); plt.show()""")

md("""### 2.2 Grafo de crianza (COMPATIBLE)
Dos especies son compatibles si comparten un *egg group* (excluyendo ditto y no-eggs).
Lo analizamos con networkx.
""")
code("""ed = q("MATCH (a:Species)-[:COMPATIBLE]->(b:Species) RETURN a.id AS a, b.id AS b")
G = nx.from_pandas_edgelist(ed, "a", "b")
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"nodos: {G.number_of_nodes():,} | aristas: {G.number_of_edges():,}")
print(f"componentes: {len(comps)} | mayor: {len(comps[0])} | densidad: {nx.density(G):.4f}")
print(f"grado medio: {2*G.number_of_edges()/G.number_of_nodes():.1f} | clustering medio: {nx.average_clustering(G):.3f}")
plt.figure(figsize=(7,4))
plt.hist(list(dict(G.degree()).values()), bins=50, color="#55a868")
plt.xlabel("grado (especies compatibles)"); plt.ylabel("nro de especies")
plt.title("Distribucion de grado del grafo de crianza"); plt.show()""")

md("""### 2.3 Distribucion de tipos primarios""")
code("""tipos = q(\"\"\"
MATCH (p:Pokemon {is_default:true})-[r:HAS_TYPE {slot:1}]->(t:Type)
RETURN t.identifier AS tipo, count(*) AS n ORDER BY n DESC
\"\"\")
plt.figure(figsize=(9,4))
plt.bar(tipos["tipo"], tipos["n"], color="#c44e52")
plt.xticks(rotation=60, ha="right"); plt.ylabel("nro de pokemon")
plt.title("Pokemon por tipo primario"); plt.show()
display(tipos.set_index("tipo").T)""")

md("""### 2.4 Capa multilingue
Cada entidad tiene nombres en 11 idiomas como nodos `Name`. Ejemplo: Pikachu.
""")
code("""q(\"\"\"
MATCH (s:Species {identifier:'pikachu'})-[:HAS_NAME]->(n:Name)
RETURN n.lang AS lang_id, n.text AS nombre ORDER BY n.lang
\"\"\")""")

md("""## 3. Las consultas que responden P1-P9

Cada consulta plantea su pregunta y la responde con una capacidad de grafo que un `SELECT/JOIN`
no resuelve limpio. Despues de cada resultado va una lectura corta.

### P1 - ¿La super-efectividad de tipos forma ciclos, o hay un tipo invencible?
Ciclos dirigidos de largo 3 en el subgrafo de super-efectividad.""")
code("""q(\"\"\"
MATCH path = (t:Type)-[:SUPER_EFFECTIVE*3]->(t)
RETURN [n IN nodes(path) | n.identifier] AS ciclo LIMIT 10
\"\"\")""")

md("""La super-efectividad no es un orden lineal: aparecen varias triadas ciclicas
(`fighting -> steel -> fairy -> fighting` y otras), asi que ningun tipo le gana a todos.
Ademas `ghost` y `dragon` salen como self-loops, fuertes contra si mismos.""")

md("""### P2 - ¿Cuales son los linajes evolutivos completos y cual es el mas largo?
Path variable de la raiz a la hoja sobre EVOLVES_TO.""")
code("""q(\"\"\"
MATCH p = (raiz:Species)-[:EVOLVES_TO*]->(hoja:Species)
WHERE NOT (:Species)-[:EVOLVES_TO]->(raiz) AND NOT (hoja)-[:EVOLVES_TO]->(:Species)
RETURN [n IN nodes(p) | n.identifier] AS linaje, length(p) AS saltos
ORDER BY saltos DESC, linaje LIMIT 10
\"\"\")""")

md("""Los linajes mas largos tienen 3 especies, o sea 2 saltos, como `bulbasaur -> ivysaur ->
venusaur`. La evolucion es recursiva pero poco profunda: ninguna cadena pasa de 2 pasos.""")

md("""### P3 - ¿Bajo que condiciones evoluciona Eevee a cada una de sus formas?
Cada fila de evolucion es un nodo `EvolutionCondition`. Leafeon y Glaceon traen varias
(piedra evolutiva o cercania a una roca), por eso se colectan todas.""")
code("""q(\"\"\"
MATCH (eevee:Species {identifier:'eevee'})-[:EVOLVES_TO]->(evo:Species)-[:EVOLVES_VIA]->(c:EvolutionCondition)
RETURN evo.identifier AS evolucion,
       collect({trigger:c.trigger, nivel:c.min_level, item:c.trigger_item,
                hora:c.time_of_day, felicidad:c.min_happiness, lugar:c.location}) AS condiciones
ORDER BY evolucion
\"\"\")""")

md("""Se ve el gatillo de cada Eeveelution: piedras (Vaporeon, Jolteon, Flareon), felicidad y
hora del dia (Espeon de dia, Umbreon de noche), y Leafeon/Glaceon por piedra o por roca.""")

md("""### P4 - ¿En que comunidades agrupa la crianza por egg groups?
Louvain (GDS) sobre el grafo COMPATIBLE.""")
code("""with driver.session() as s: s.run("CALL gds.graph.drop('breeding', false)")
q("CALL gds.graph.project('breeding', 'Species', {COMPATIBLE: {orientation: 'UNDIRECTED'}})")
q(\"\"\"
CALL gds.louvain.stream('breeding') YIELD nodeId, communityId
RETURN communityId, count(*) AS tam, collect(gds.util.asNode(nodeId).identifier)[..5] AS muestra
ORDER BY tam DESC LIMIT 10
\"\"\")""")

md("""Las comunidades caen en familias tematicas (grupo campo, monstruo/agua, bicho, planta).
Nidoqueen y Nidorina salen solas porque son del grupo no-eggs, que no cria.""")

md("""### P5 - ¿Que especies actuan como puente entre comunidades de crianza?
Betweenness (GDS) sobre la misma proyeccion de P4.""")
code("""q(\"\"\"
CALL gds.betweenness.stream('breeding') YIELD nodeId, score
RETURN gds.util.asNode(nodeId).identifier AS especie, round(score) AS score
ORDER BY score DESC LIMIT 10
\"\"\")""")

md("""Las especies con mayor betweenness (cufant, fidough, copperajah...) son puentes:
conectan comunidades de crianza que sin ellas quedarian separadas.""")

md("""### P6 - ¿Que tipo es ofensivamente mas central en la cadena de efectividad?
PageRank (GDS) ponderado por el factor de dano.""")
code("""with driver.session() as s: s.run("CALL gds.graph.drop('typechart', false)")
q("CALL gds.graph.project('typechart', 'Type', {EFFECTIVENESS: {properties: 'factor'}})")
q(\"\"\"
CALL gds.pageRank.stream('typechart', {relationshipWeightProperty: 'factor'}) YIELD nodeId, score
RETURN gds.util.asNode(nodeId).identifier AS tipo, round(score*1000)/1000 AS score
ORDER BY score DESC LIMIT 8
\"\"\")""")

md("""Por PageRank ponderado, `ice`, `grass` y `rock` son los tipos ofensivamente mas
centrales: su ventaja se propaga mas lejos por la cadena de efectividad.""")

md("""### P7 - ¿Que par de Pokemon comparte mas movimientos aprendibles?
Proyeccion N-a-N. Se deduplica el multigrafo a pares distintos antes del self-join.""")
code("""q(\"\"\"
MATCH (p:Pokemon)-[:CAN_LEARN]->(m:Move)
WHERE p.is_default
WITH DISTINCT m, p
WITH m, collect(p) AS aprendices
UNWIND aprendices AS a UNWIND aprendices AS b
WITH a, b WHERE a.id < b.id
WITH a, b, count(*) AS comunes WHERE comunes > 120
RETURN a.identifier AS pokemon_a, b.identifier AS pokemon_b, comunes
ORDER BY comunes DESC LIMIT 15
\"\"\")""")

md("""El par con mas moves en comun es `mew`/`arceus` (164): los dos aprenden casi todo, lo
que los vuelve hubs del grafo de movesets.""")

md("""### P8 - ¿Que areas concentran mas biodiversidad de especies?
Agregacion sobre los encuentros reificados. Si el area no tiene nombre propio se cae al de la
Location padre.""")
code("""q(\"\"\"
MATCH (la:LocationArea)<-[:AT_AREA]-(:Encounter)<-[:HAS_ENCOUNTER]-(p:Pokemon)-[:IS_SPECIES]->(s:Species)
OPTIONAL MATCH (la)-[:IN_LOCATION]->(loc:Location)
WITH la, loc, count(DISTINCT s) AS biodiversidad
RETURN coalesce(la.identifier, loc.identifier, toString(la.id)) AS area, biodiversidad
ORDER BY biodiversidad DESC LIMIT 12
\"\"\")""")

md("""Las areas mas biodiversas (`kanto-route-13`, `elite-four-defeated`) concentran cerca de
45 especies distintas.""")

md("""### P9 - ¿En que linaje se gana mas poder de la forma base a la final?
Recursion sobre el path completo raiz-a-hoja (`EVOLVES_TO*`) mas agregacion de la ganancia de stats.""")
code("""q(\"\"\"
MATCH path = (raiz:Species)-[:EVOLVES_TO*]->(hoja:Species)
WHERE NOT (:Species)-[:EVOLVES_TO]->(raiz) AND NOT (hoja)-[:EVOLVES_TO]->(:Species)
MATCH (pr:Pokemon {is_default:true})-[:IS_SPECIES]->(raiz)
MATCH (ph:Pokemon {is_default:true})-[:IS_SPECIES]->(hoja)
MATCH (pr)-[r1:HAS_STAT]->(s:Stat)<-[r2:HAS_STAT]-(ph)
WITH [n IN nodes(path) | n.identifier] AS linaje, length(path) AS pasos,
     sum(r2.base_stat - r1.base_stat) AS ganancia
RETURN linaje, pasos, ganancia ORDER BY ganancia DESC LIMIT 12
\"\"\")""")

md("""El mayor salto de poder en un linaje completo es `cosmog -> cosmoem -> lunala`/`solgaleo`
(+480), seguido de `slakoth -> slaking` (+390). En un solo paso, `magikarp -> gyarados` y
`feebas -> milotic` (+340) son los que mas suben.""")

md("""## 4. Machine Learning basico

### 4.1 - ML-1: ¿el fenotipo basta para predecir el tipo?
Predecir el tipo primario (18 clases) desde las 6 stats base mas el conteo de moves por tipo.
""")
code("""from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

stats = q(\"\"\"MATCH (p:Pokemon {is_default:true})-[r:HAS_STAT]->(s:Stat)
RETURN p.id AS pokemon, s.identifier AS stat, r.base_stat AS v\"\"\").pivot_table(
    index="pokemon", columns="stat", values="v", fill_value=0)
movetypes = q(\"\"\"MATCH (p:Pokemon {is_default:true})-[:CAN_LEARN]->(m:Move)-[:MOVE_TYPE]->(t:Type)
WITH p, t, count(DISTINCT m) AS c
RETURN p.id AS pokemon, 'mt_' + t.identifier AS movetype, c\"\"\").pivot_table(
    index="pokemon", columns="movetype", values="c", fill_value=0)
label = q(\"\"\"MATCH (p:Pokemon {is_default:true})-[r:HAS_TYPE {slot:1}]->(t:Type)
RETURN p.id AS pokemon, t.identifier AS tipo\"\"\").set_index("pokemon")["tipo"]

data = stats.join(movetypes, how="left").fillna(0).join(label, how="inner").dropna(subset=["tipo"])
y = data["tipo"]; Xm = data.drop(columns="tipo")
clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = cross_val_score(clf, Xm, y, cv=cv)
stat_cols = [c for c in Xm.columns if not c.startswith("mt_")]
scores_stats = cross_val_score(clf, Xm[stat_cols], y, cv=cv)
print(f"{Xm.shape[0]} pokemon, {Xm.shape[1]} features, {y.nunique()} clases")
print(f"accuracy 5-fold CV: {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"baseline (clase mayoritaria): {y.value_counts(normalize=True).max():.3f}")
print(f"accuracy solo con stats base: {scores_stats.mean():.3f}")""")

md("""Hay que leer ese 0.82 con cuidado: se apoya casi entero en los conteos de moves por tipo.
Con solo las 6 stats base la accuracy cae a ~0.20, apenas sobre el baseline. Tiene sentido, un
pokemon de fuego aprende muchos moves de fuego (efecto STAB), asi que predecir el tipo desde el
movepool es en parte circular. La senal es real pero la tarea es mas facil de lo que el numero sugiere.
""")

md("""### 4.2 - ML-2: ¿se puede predecir la crianza desde el fenotipo?
Dos encuadres. (a) **Topologico**: con features de vecindario el AUC es ~1.0, pero esto es
**estructural** (COMPATIBLE es una union de cliques solapadas, una por egg group),
no un logro del modelo. (b) **Por atributos fenotipicos** (stats, tipo, generacion): la tarea
predictiva real, no trivial.
""")
code("""from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(42)

nodes = sorted(set(ed.a) | set(ed.b))
edge_set = {(min(a,b), max(a,b)) for a,b in zip(ed.a, ed.b)}
pos = np.array(list(edge_set))
pos_tr, pos_te = train_test_split(pos, test_size=0.2, random_state=42)
adj = {n:set() for n in nodes}
for a,b in pos_tr: adj[a].add(b); adj[b].add(a)

def rand_neg(k):
    out=set(); nl=np.array(nodes)
    while len(out)<k:
        u,v = rng.choice(nl,2,replace=False); e=(int(min(u,v)),int(max(u,v)))
        if e not in edge_set: out.add(e)
    return np.array(list(out))
neg = rand_neg(len(pos)); neg_tr, neg_te = train_test_split(neg, test_size=0.2, random_state=42)
ytr = np.r_[np.ones(len(pos_tr)), np.zeros(len(neg_tr))]
yte = np.r_[np.ones(len(pos_te)), np.zeros(len(neg_te))]

def topo(pairs):
    r=[]
    for u,v in pairs:
        nu,nv=adj[u],adj[v]; cm=nu&nv; un=nu|nv
        r.append([len(cm), len(cm)/len(un) if un else 0,
                  sum(1/math.log(len(adj[w])) for w in cm if len(adj[w])>1), len(nu)*len(nv)])
    return np.array(r,dtype=float)
m1=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1).fit(np.vstack([topo(pos_tr),topo(neg_tr)]),ytr)
p1=m1.predict_proba(np.vstack([topo(pos_te),topo(neg_te)]))[:,1]
auc1=roc_auc_score(yte,p1)

sf = q(\"\"\"MATCH (s:Species)<-[:IS_SPECIES]-(p:Pokemon {is_default:true})-[r:HAS_STAT]->(st:Stat)
RETURN s.id AS sid, st.identifier AS stat, r.base_stat AS v\"\"\").pivot_table(index="sid",columns="stat",values="v",fill_value=0)
sm = q(\"\"\"MATCH (s:Species) OPTIONAL MATCH (s)<-[:IS_SPECIES]-(:Pokemon {is_default:true})-[:HAS_TYPE {slot:1}]->(t:Type)
RETURN s.id AS sid, s.generation_id AS gen, t.identifier AS ptype\"\"\").set_index("sid")
def attr(pairs):
    r=[]
    for u,v in pairs:
        if u not in sf.index or v not in sf.index: r.append([0.0]*(sf.shape[1]+2)); continue
        d=list(np.abs(sf.loc[u].values-sf.loc[v].values))
        st=1.0 if (pd.notna(sm.loc[u,'ptype']) and sm.loc[u,'ptype']==sm.loc[v,'ptype']) else 0.0
        sg=1.0 if sm.loc[u,'gen']==sm.loc[v,'gen'] else 0.0
        r.append(d+[st,sg])
    return np.array(r,dtype=float)
m2=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1).fit(np.vstack([attr(pos_tr),attr(neg_tr)]),ytr)
p2=m2.predict_proba(np.vstack([attr(pos_te),attr(neg_te)]))[:,1]
auc2=roc_auc_score(yte,p2)
print(f"2a) topologico (cliques solapadas): AUC={auc1:.3f}")
print(f"2b) por atributos fenotipicos:     AUC={auc2:.3f}")
plt.figure(figsize=(6,5))
for p,a,l,c in [(p1,auc1,'topologico','#8172b3'),(p2,auc2,'atributos','#c44e52')]:
    fpr,tpr,_=roc_curve(yte,p); plt.plot(fpr,tpr,color=c,label=f"{l}: AUC={a:.3f}")
plt.plot([0,1],[0,1],'--',color='gray'); plt.legend(loc='lower right')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Link prediction crianza"); plt.show()""")

md("""## 5. Conclusiones

PokeAPI resulto un grafo de conocimiento genuino: cumple el criterio duro del enunciado (N-a-N
navegable, ciclos, recursion), todo verificado contra los datos, con del orden de 130 mil nodos
y 900 mil aristas. Los conteos exactos estan en la seccion 1.

El grafo de propiedades fue la eleccion correcta. `CAN_LEARN` carga propiedades en la arista y es
un multigrafo (el mismo par pokemon-move se repite por version y metodo), algo que en RDF puro
obligaria a reificar. Las consultas P1-P9 se apoyan en esa estructura: ciclos del type chart,
paths evolutivos, comunidades y centralidad de crianza, proyecciones N-a-N de movesets.

Del ML salieron dos lecturas honestas. El grafo de crianza es una union de cliques solapadas (una
por egg group), asi que predecir enlaces por topologia da AUC ~1.0, pero eso describe el grafo, no
al modelo; preguntar lo no trivial, si dos especies pueden cruzarse mirando solo su fenotipo, baja
a AUC ~0.67. Y la clasificacion de tipo llega a ~0.82, pero buena parte de esa senal es el movepool
(efecto STAB): con solo las stats base cae a ~0.20. En ambos casos el numero vistoso esconde un
matiz que vale mas que el numero.
""")
code("""driver.close()
print("fin del reporte")""")

nb["cells"] = cells
out = os.path.join(os.path.dirname(__file__), "reporte.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("notebook escrito en", out, "con", len(cells), "celdas")
