// Capa de analisis competitivo (Smogon/VGC) sobre el grafo base. Estas consultas explotan el
// cuadro de tipos (EFFECTIVENESS/SUPER_EFFECTIVE), los stats base (HAS_STAT) y los movepools
// (CAN_LEARN, MOVE_TYPE) para preguntas de teambuilding que las 9 consultas estructurales no tocan.
// Limitacion honesta: medimos potencial por tipos y stats base, no el meta real (no hay efectos de
// habilidades/moves, ni EVs/naturalezas/items). "Aprendible" != "se usa en el meta".

// C1 - ¿Que Pokemon tienen el mejor tipado defensivo (mas resistencias e inmunidades)?
// El multiplicador frente a cada tipo atacante es el producto de los factores sobre sus 1-2 tipos.
MATCH (p:Pokemon)-[:HAS_TYPE]->(def:Type) WHERE p.is_default
WITH p, collect(def) AS defs
MATCH (atk:Type)
OPTIONAL MATCH (atk)-[e:EFFECTIVENESS]->(d) WHERE d IN defs
WITH p, atk, reduce(f=1.0, x IN collect(e.factor/100.0) | f*x) AS mult
RETURN p.identifier AS pokemon,
       count(CASE WHEN mult=0.0 THEN 1 END) AS inmunidades,
       count(CASE WHEN mult>0 AND mult<1 THEN 1 END) AS resistencias,
       count(CASE WHEN mult>1 THEN 1 END) AS debilidades
ORDER BY (inmunidades+resistencias) DESC, debilidades ASC LIMIT 15;

// C2 - ¿Que pares forman cores defensivos? (cada uno resiste todas las debilidades del otro)
MATCH (p:Pokemon)-[:HAS_TYPE]->(dt:Type) WHERE p.is_default
WITH p, collect(dt) AS defs
MATCH (atk:Type)
OPTIONAL MATCH (atk)-[e:EFFECTIVENESS]->(d) WHERE d IN defs
WITH p, atk, reduce(f=1.0, x IN collect(e.factor/100.0) | f*x) AS mult
WITH p, [a IN collect(CASE WHEN mult>=2.0 THEN atk.identifier END) WHERE a IS NOT NULL] AS weak,
        [a IN collect(CASE WHEN mult<1.0 THEN atk.identifier END) WHERE a IS NOT NULL] AS resist
WHERE size(weak) IN [2,3]
WITH collect({pk:p.identifier, weak:weak, resist:resist}) AS L
UNWIND range(0, size(L)-2) AS i UNWIND range(i+1, size(L)-1) AS j
WITH L[i] AS A, L[j] AS B
WHERE all(w IN A.weak WHERE w IN B.resist) AND all(w IN B.weak WHERE w IN A.resist)
RETURN A.pk AS poke_a, B.pk AS poke_b, A.weak AS debil_a, B.weak AS debil_b
ORDER BY poke_a, poke_b LIMIT 15;

// C3 - Vulnerabilidad a Stealth Rock por tier de multiplicador, y cuantos no tienen control propio.
// El control de hazards es defog / rapid-spin via CAN_LEARN. Los 4x pierden 50% de vida al entrar.
MATCH (rock:Type {identifier:'rock'})
MATCH (p:Pokemon)-[:HAS_TYPE]->(t:Type) WHERE p.is_default
MATCH (rock)-[e:EFFECTIVENESS]->(t)
WITH p, reduce(m=1.0, f IN collect(e.factor/100.0) | m*f) AS srMult
OPTIONAL MATCH (p)-[:CAN_LEARN]->(ctrl:Move) WHERE ctrl.identifier IN ['defog','rapid-spin']
WITH p, srMult, count(DISTINCT ctrl) AS control
RETURN srMult AS multiplicador_SR, count(p) AS pokemon,
       sum(CASE WHEN control=0 THEN 1 ELSE 0 END) AS sin_autocontrol
ORDER BY srMult DESC;

// C4 - ¿Que amenazas son mas dificiles de murar por tipos? Un counter resiste TODOS los STAB del
// threat (producto de factores < 1 para cada tipo). Menos counters = mas dificil de contrarrestar.
// Dragapult (ghost/dragon) sale con los menos; great-tusk (ground/fighting) con muchos.
UNWIND ['garchomp','tyranitar','dragonite','kingambit','great-tusk','dragapult'] AS name
MATCH (threat:Pokemon {identifier:name})-[:HAS_TYPE]->(tt:Type)
WITH name, collect(DISTINCT tt) AS T
MATCH (c:Pokemon)-[:HAS_TYPE]->(ct:Type) WHERE c.is_default
WITH name, T, c, collect(ct) AS cdefs
WHERE ALL(tk IN T WHERE
      reduce(f=1.0, d IN cdefs | f * coalesce([(tk)-[e:EFFECTIVENESS]->(d) | e.factor/100.0][0], 1.0)) < 1.0)
RETURN name AS amenaza, count(c) AS counters_por_tipos ORDER BY counters_por_tipos;

// C5 - ¿Que Pokemon son revenge-killers? (lentos, ataque alto, con prioridad STAB de dano)
// priority>=1 y power>0 (excluye silk-trap/burning-bulwark, prio alta pero sin dano).
MATCH (p:Pokemon)-[:CAN_LEARN]->(m:Move)-[:MOVE_TYPE]->(mt:Type)
WHERE m.priority >= 1 AND m.power > 0 AND p.is_default
MATCH (p)-[:HAS_TYPE]->(pt:Type) WHERE pt = mt
MATCH (p)-[hs:HAS_STAT]->(:Stat {identifier:'speed'})
MATCH (p)-[ha:HAS_STAT]->(:Stat {identifier:'attack'})
WITH p.identifier AS poke, hs.base_stat AS speed, ha.base_stat AS ataque, collect(DISTINCT m.identifier) AS prio_stab
WHERE speed <= 60 AND ataque >= 110
RETURN poke, speed, ataque, prio_stab ORDER BY ataque DESC LIMIT 15;

// C6 - ¿Cuantos Pokemon superan cada speed tier clave del meta? (solo velocidad base)
MATCH (p:Pokemon {is_default:true})-[r:HAS_STAT]->(:Stat {identifier:'speed'})
WITH r.base_stat AS speed
RETURN sum(CASE WHEN speed > 100 THEN 1 ELSE 0 END) AS sobre_base_100,
       sum(CASE WHEN speed > 110 THEN 1 ELSE 0 END) AS sobre_base_110,
       sum(CASE WHEN speed > 120 THEN 1 ELSE 0 END) AS sobre_base_120,
       sum(CASE WHEN speed > 130 THEN 1 ELSE 0 END) AS sobre_base_130,
       max(speed) AS tope;

// C7 - ¿Que Pokemon golpea super-efectivo a mas tipos usando solo su STAB? (el "EdgeQuake" y cia)
MATCH (p:Pokemon)-[:HAS_TYPE]->(pt:Type) WHERE p.is_default
MATCH (pt)-[:SUPER_EFFECTIVE]->(def:Type)
WITH p, collect(DISTINCT pt.identifier) AS tipos, count(DISTINCT def) AS cobertura_stab
RETURN p.identifier AS pokemon, tipos, cobertura_stab ORDER BY cobertura_stab DESC LIMIT 15;

// C8 - Calculadora de dano en Cypher (Lv100, 252 EV en el stat ofensivo, sin naturaleza).
// Ej: Charizard Flamethrower vs Venusaur. El rango (rolls 0.85-1.0) coincide con el calc de Showdown.
MATCH (atk:Pokemon {identifier:'charizard'})-[ha:HAS_STAT]->(:Stat {identifier:'special-attack'})
MATCH (def:Pokemon {identifier:'venusaur'})-[hd:HAS_STAT]->(:Stat {identifier:'special-defense'})
MATCH (m:Move {identifier:'flamethrower'})-[:MOVE_TYPE]->(mt:Type)
WITH atk, m, mt, ha, hd, CASE WHEN exists((atk)-[:HAS_TYPE]->(mt)) THEN 1.5 ELSE 1.0 END AS stab
MATCH (def:Pokemon {identifier:'venusaur'})-[:HAS_TYPE]->(dt:Type)
OPTIONAL MATCH (mt)-[e:EFFECTIVENESS]->(dt)
WITH m, ha, hd, stab, reduce(eff=1.0, f IN collect(coalesce(e.factor,100)/100.0) | eff*f) AS typeMult
WITH m, stab, typeMult,
     toInteger(floor((2.0*ha.base_stat + 31 + floor(252/4.0)) * 100/100.0)) + 5 AS A,
     toInteger(floor((2.0*hd.base_stat + 31 + floor(252/4.0)) * 100/100.0)) + 5 AS D
WITH m.power AS power, stab, typeMult,
     toInteger(floor(floor(floor(42.0*m.power*A/toFloat(D)) / 50))) + 2 AS baseDmg
RETURN power, stab, typeMult,
       toInteger(floor(baseDmg*stab*typeMult*0.85)) AS dmg_min,
       toInteger(floor(baseDmg*stab*typeMult*1.00)) AS dmg_max;

// C9 - Movepool legal por metodo en un version_group (legality). metodo: 1=level-up, 2=egg, 3=tutor, 4=TM.
MATCH (p:Pokemon {identifier:'garchomp'})-[r:CAN_LEARN]->(m:Move)
WHERE r.version_group = 25
RETURN r.method AS metodo, count(DISTINCT m) AS moves ORDER BY moves DESC;

// === Capa meta de Smogon (requiere haber corrido pipeline/06_smogon.py) ===

// M1 - Kit legal (aprendible) vs kit realmente usado en el meta, por Pokemon de OU.
MATCH (p:Pokemon)-[u:USED_IN]->(:Format {tier:'gen9ou'})
OPTIONAL MATCH (p)-[:CAN_LEARN]->(legal:Move)
WITH p, u.usage AS uso_pct, count(DISTINCT legal) AS aprendibles
OPTIONAL MATCH (p)-[:RUNS_MOVE]->(usado:Move)
RETURN p.identifier AS pokemon, uso_pct, aprendibles, count(DISTINCT usado) AS usados_en_meta
ORDER BY uso_pct DESC LIMIT 15;

// M2 - Nucleos de equipo: comunidades sobre el grafo real de companeros (TEAMMATE_OF) con Louvain.
CALL gds.graph.drop('teammates', false) YIELD graphName;
CALL gds.graph.project('teammates', 'Pokemon', {TEAMMATE_OF: {orientation:'UNDIRECTED'}});
CALL gds.louvain.stream('teammates') YIELD nodeId, communityId
WITH communityId, collect(gds.util.asNode(nodeId).identifier) AS mons
WHERE size(mons) > 2
RETURN communityId, size(mons) AS tam, mons[..8] AS muestra ORDER BY tam DESC LIMIT 8;

// M3 - Uso real cruzado con el tipo: que tipos dominan el meta de OU.
MATCH (p:Pokemon)-[u:USED_IN]->(:Format {tier:'gen9ou'})
MATCH (p)-[:HAS_TYPE]->(t:Type)
RETURN t.identifier AS tipo, count(DISTINCT p) AS mons_en_OU, round(avg(u.usage),2) AS uso_promedio
ORDER BY mons_en_OU DESC LIMIT 12;
