"""Capa de metajuego de Smogon sobre el grafo base (datos externos).

Descarga las usage stats mensuales de Smogon (formato chaos.json de Pokemon Showdown) y las
materializa como subgrafo reusando los nodos Pokemon/Move/Item/Ability que ya existen:

    (:Format)                              el metagame (gen9ou de un mes)
    (Pokemon)-[:USED_IN {usage,rank}]->(Format)
    (Pokemon)-[:RUNS_MOVE {pct}]->(Move)
    (Pokemon)-[:HOLDS_ITEM {pct}]->(Item)
    (Pokemon)-[:USES_ABILITY {pct}]->(Ability)
    (Pokemon)-[:TEAMMATE_OF {pct}]->(Pokemon)
    (Pokemon)-[:CHECKED_BY {score}]->(Pokemon)   de "Checks and Counters"

Esto habilita preguntas que el grafo base no puede: kit legal (CAN_LEARN) vs kit usado (RUNS_MOVE),
nucleos de equipo reales (TEAMMATE_OF), uso vs centralidad de tipo. El cuello de botella es la
normalizacion de nombres Smogon -> identifier de PokeAPI (formas therian/mega/ogerpon).

    python pipeline/06_smogon.py     (requiere el grafo cargado y red para bajar el JSON)
"""
import json
import re
import urllib.request
from neo4j import GraphDatabase

MONTH = "2026-05"
FORMAT = "gen9ou-1825"
URL = f"https://www.smogon.com/stats/{MONTH}/chaos/{FORMAT}.json"
FORMAT_ID = f"gen9ou-{MONTH}"

# Excepciones donde el slug directo no calza con el identifier de PokeAPI.
FORMAS = {
    "ogerpon-wellspring": "ogerpon-wellspring-mask",
    "ogerpon-hearthflame": "ogerpon-hearthflame-mask",
    "ogerpon-cornerstone": "ogerpon-cornerstone-mask",
    "urshifu": "urshifu-single-strike",
    "urshifu-rapid-strike": "urshifu-rapid-strike",
    "basculegion": "basculegion-male",
    "indeedee": "indeedee-male",
    "tauros-paldea-combat": "tauros-paldea-combat-breed",
    "tauros-paldea-blaze": "tauros-paldea-blaze-breed",
    "tauros-paldea-aqua": "tauros-paldea-aqua-breed",
    "maushold": "maushold-family-of-four",
    "dudunsparce": "dudunsparce-two-segment",
    "keldeo": "keldeo-ordinary",
    "zamazenta-crowned": "zamazenta-crowned",
}


def slug(name):
    s = name.lower().strip()
    s = s.replace(" ", "-").replace("'", "").replace(".", "").replace(":", "").replace("%", "")
    s = re.sub(r"-+", "-", s)
    return s


def poke_id(name, pokes):
    s = slug(name)
    if s in pokes:
        return s
    if s in FORMAS and FORMAS[s] in pokes:
        return FORMAS[s]
    # formas tipo "X-Mega", "X-Therian" que en PokeAPI van como "x-mega"/"x-therian"
    return s if s in pokes else None


def main():
    print(f">> bajando {URL}")
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (pokeapi-kg research)"}), timeout=60
    ).read()
    data = json.loads(raw)["data"]
    print(f"   {len(data)} pokemon en {FORMAT}")

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)
    with driver.session() as s:
        idlist = lambda label: [r["i"] for r in s.run(f"MATCH (n:{label}) WHERE n.identifier IS NOT NULL RETURN n.identifier AS i")]
        strip = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
        pokes = set(idlist("Pokemon"))
        # Smogon usa IDs sin separadores (headlongrush, heavydutyboots); PokeAPI usa guiones.
        movemap = {strip(i): i for i in idlist("Move")}
        itemmap = {strip(i): i for i in idlist("Item")}
        abilmap = {strip(i): i for i in idlist("Ability")}
        s.run("MERGE (f:Format {id:$id}) SET f.tier='gen9ou', f.month=$m", id=FORMAT_ID, m=MONTH)
        s.run("MATCH ()-[r:USED_IN|RUNS_MOVE|HOLDS_ITEM|USES_ABILITY|TEAMMATE_OF|CHECKED_BY]-() DELETE r")

        ranked = sorted(data.items(), key=lambda kv: -kv[1].get("usage", 0))
        matched = unmatched = 0
        for rank, (name, d) in enumerate(ranked, 1):
            pid = poke_id(name, pokes)
            if pid is None:
                unmatched += 1
                continue
            matched += 1
            total = sum(d.get("Items", {}).values()) or sum(d.get("Abilities", {}).values()) or 1.0

            def edges(field, rel, pmap, target_label):
                rows = []
                for k, w in d.get(field, {}).items():
                    tid = pmap.get(strip(k))
                    if tid and w / total >= 0.02:
                        rows.append({"t": tid, "pct": round(100 * w / total, 1)})
                if rows:
                    s.run(
                        f"MATCH (p:Pokemon {{identifier:$pid}}) UNWIND $rows AS r "
                        f"MATCH (x:{target_label} {{identifier:r.t}}) MERGE (p)-[e:{rel}]->(x) SET e.pct=r.pct",
                        pid=pid, rows=rows,
                    )

            s.run(
                "MATCH (p:Pokemon {identifier:$pid}) MATCH (f:Format {id:$fid}) "
                "MERGE (p)-[u:USED_IN]->(f) SET u.usage=$usage, u.rank=$rank",
                pid=pid, fid=FORMAT_ID, usage=round(100 * d.get("usage", 0), 3), rank=rank,
            )
            edges("Moves", "RUNS_MOVE", movemap, "Move")
            edges("Items", "HOLDS_ITEM", itemmap, "Item")
            edges("Abilities", "USES_ABILITY", abilmap, "Ability")

            # teammates (peso ya viene como desviacion sobre lo esperado; guardamos el crudo)
            tm = [{"t": poke_id(k, pokes), "w": round(v, 1)} for k, v in d.get("Teammates", {}).items()]
            tm = [r for r in tm if r["t"] and r["w"] > 0][:12]
            if tm:
                s.run(
                    "MATCH (p:Pokemon {identifier:$pid}) UNWIND $rows AS r "
                    "MATCH (q:Pokemon {identifier:r.t}) MERGE (p)-[e:TEAMMATE_OF]->(q) SET e.pct=r.w",
                    pid=pid, rows=tm,
                )
            # checks and counters: {opp: [n, score, stddev]} -> CHECKED_BY score
            cc = []
            for k, v in d.get("Checks and Counters", {}).items():
                tid = poke_id(k, pokes)
                if tid and isinstance(v, dict) and v.get("p", 0) > 0:
                    cc.append({"t": tid, "score": round(v["p"], 3)})
            cc = sorted(cc, key=lambda r: -r["score"])[:10]
            if cc:
                s.run(
                    "MATCH (p:Pokemon {identifier:$pid}) UNWIND $rows AS r "
                    "MATCH (q:Pokemon {identifier:r.t}) MERGE (p)-[e:CHECKED_BY]->(q) SET e.score=r.score",
                    pid=pid, rows=cc,
                )

        print(f">> emparejados {matched} / {matched + unmatched} pokemon (sin match: {unmatched})")
        for rel in ["USED_IN", "RUNS_MOVE", "HOLDS_ITEM", "USES_ABILITY", "TEAMMATE_OF", "CHECKED_BY"]:
            n = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n").single()["n"]
            print(f"   {rel}: {n}")
    driver.close()
    print(">> LISTO. Capa meta cargada. Consultas en pipeline/queries_competitivo.cypher (seccion meta).")


if __name__ == "__main__":
    main()
