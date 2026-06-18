"""Exploracion y caracterizacion del grafo PokeAPI (Hito 2 del proyecto MDS7205).

Se conecta al Neo4j cargado por pipeline/load_all.sh y calcula metricas estructurales:
conteos, distribuciones de grado, componentes del grafo de crianza, distribucion de tipos.
Guarda figuras en analysis/img/. Corre con: python3 analysis/eda.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase

IMG = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(IMG, exist_ok=True)
driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)


def df(query):
    with driver.session() as s:
        return pd.DataFrame([r.data() for r in s.run(query)])


print("== Conteos globales ==")
nodos = df("MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY n DESC")
aristas = df("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC")
print(f"  nodos totales: {nodos['n'].sum():,} en {len(nodos)} labels")
print(f"  aristas totales: {aristas['n'].sum():,} en {len(aristas)} tipos")

print("\n== Distribucion de grado: moves aprendibles por pokemon (default) ==")
deg = df("""
MATCH (p:Pokemon {is_default:true})-[:CAN_LEARN]->(m:Move)
RETURN p.identifier AS pokemon, count(DISTINCT m) AS moves
""")
print(deg["moves"].describe().round(1).to_string())
plt.figure(figsize=(7, 4))
plt.hist(deg["moves"], bins=40, color="#4c72b0")
plt.xlabel("moves distintos aprendibles"); plt.ylabel("nro de pokemon")
plt.title("Distribucion de tamaño de movepool (formas default)")
plt.tight_layout(); plt.savefig(f"{IMG}/movepool_hist.png", dpi=110); plt.close()

print("\n== Grafo de crianza (COMPATIBLE) via networkx ==")
edges = df("MATCH (a:Species)-[:COMPATIBLE]->(b:Species) RETURN a.id AS a, b.id AS b")
G = nx.from_pandas_edgelist(edges, "a", "b")
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"  nodos: {G.number_of_nodes():,} | aristas: {G.number_of_edges():,}")
print(f"  componentes: {len(comps)} | mayor: {len(comps[0])} | densidad: {nx.density(G):.4f}")
print(f"  grado medio: {2*G.number_of_edges()/G.number_of_nodes():.1f} | clustering medio: {nx.average_clustering(G):.3f}")
degs = pd.Series(dict(G.degree()))
plt.figure(figsize=(7, 4))
plt.hist(degs, bins=50, color="#55a868")
plt.xlabel("grado (especies compatibles)"); plt.ylabel("nro de especies")
plt.title("Distribucion de grado del grafo de crianza")
plt.tight_layout(); plt.savefig(f"{IMG}/breeding_degree_hist.png", dpi=110); plt.close()

print("\n== Distribucion de tipos primarios (formas default) ==")
tipos = df("""
MATCH (p:Pokemon {is_default:true})-[r:HAS_TYPE {slot:1}]->(t:Type)
RETURN t.identifier AS tipo, count(*) AS n ORDER BY n DESC
""")
print(tipos.to_string(index=False))
plt.figure(figsize=(9, 4))
plt.bar(tipos["tipo"], tipos["n"], color="#c44e52")
plt.xticks(rotation=60, ha="right"); plt.ylabel("nro de pokemon")
plt.title("Pokemon por tipo primario")
plt.tight_layout(); plt.savefig(f"{IMG}/type_distribution.png", dpi=110); plt.close()

print("\n== Multilingue: nombres de pikachu por idioma (muestra de la capa Name) ==")
nombres = df("""
MATCH (s:Species {identifier:'pikachu'})-[:HAS_NAME]->(n:Name)
RETURN n.lang AS lang_id, n.text AS nombre ORDER BY n.lang
""")
print(nombres.to_string(index=False))

print(f"\nFiguras guardadas en {IMG}/")
driver.close()
