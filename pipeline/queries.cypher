// Las 9 consultas del proyecto. Cada una responde una pregunta sobre la red explotando una
// capacidad de grafo (ciclos, paths variables, comunidades, centralidad) que un SELECT/JOIN no
// resuelve limpio. P4, P5 y P6 usan GDS (gds.*); el resto es Cypher puro.
// Correr una por una en el Browser (http://localhost:7474) o con cypher-shell.

// P1 - ¿La super-efectividad de tipos forma ciclos, o hay un tipo que le gana a todos?
// Ciclos dirigidos de largo 3 en el subgrafo de super-efectividad.
MATCH path = (t:Type)-[:SUPER_EFFECTIVE*3]->(t)
RETURN [n IN nodes(path) | n.identifier] AS ciclo LIMIT 25;
// self-loops (tipos fuertes contra si mismos): MATCH (t:Type)-[:SUPER_EFFECTIVE]->(t) RETURN t.identifier;

// P2 - ¿Cuales son los linajes evolutivos completos y cual es el mas largo? (path recursivo)
MATCH p = (raiz:Species)-[:EVOLVES_TO*]->(hoja:Species)
WHERE NOT (:Species)-[:EVOLVES_TO]->(raiz) AND NOT (hoja)-[:EVOLVES_TO]->(:Species)
RETURN [n IN nodes(p) | n.identifier] AS linaje, length(p) AS saltos
ORDER BY saltos DESC, linaje LIMIT 15;

// P3 - ¿Bajo que condiciones evoluciona Eevee a cada una de sus formas? (relacion n-aria reificada)
// Eevee tiene 8 evoluciones; Leafeon y Glaceon traen mas de una condicion (piedra o roca),
// por eso se colectan todas.
MATCH (eevee:Species {identifier:'eevee'})-[:EVOLVES_TO]->(evo:Species)-[:EVOLVES_VIA]->(c:EvolutionCondition)
RETURN evo.identifier AS evolucion,
       collect({trigger:c.trigger, nivel:c.min_level, item:c.trigger_item,
                hora:c.time_of_day, felicidad:c.min_happiness, lugar:c.location}) AS condiciones
ORDER BY evolucion;

// P4 - ¿En que comunidades agrupa la crianza por egg groups? (Louvain, GDS)
// El drop idempotente deja re-correr sin reiniciar la sesion (el catalogo GDS es global).
CALL gds.graph.drop('breeding', false) YIELD graphName;
CALL gds.graph.project('breeding', 'Species', {COMPATIBLE: {orientation: 'UNDIRECTED'}});
CALL gds.louvain.stream('breeding') YIELD nodeId, communityId
RETURN communityId, count(*) AS tam,
       collect(gds.util.asNode(nodeId).identifier)[..6] AS muestra
ORDER BY tam DESC LIMIT 12;

// P5 - ¿Que especies actuan como puente entre comunidades de crianza? (betweenness, GDS)
// Reusa la proyeccion 'breeding' de P4: corre P4 antes.
CALL gds.betweenness.stream('breeding') YIELD nodeId, score
RETURN gds.util.asNode(nodeId).identifier AS especie, round(score) AS score
ORDER BY score DESC LIMIT 15;

// P6 - ¿Que tipo es ofensivamente mas central en la cadena de efectividad? (PageRank ponderado, GDS)
CALL gds.graph.drop('typechart', false) YIELD graphName;
CALL gds.graph.project('typechart', 'Type', {EFFECTIVENESS: {properties: 'factor'}});
CALL gds.pageRank.stream('typechart', {relationshipWeightProperty: 'factor'}) YIELD nodeId, score
RETURN gds.util.asNode(nodeId).identifier AS tipo, score ORDER BY score DESC LIMIT 10;

// P7 - ¿Que par de Pokemon comparte mas movimientos aprendibles? (proyeccion N-a-N)
// CAN_LEARN es multigrafo (el mismo par pokemon-move se repite por version/metodo), asi que se
// deduplica a pares distintos antes del self-join; si no, enumera E^2 y explota. El maximo real
// es ~164 (mew/arceus, porque Mew aprende casi todo).
MATCH (p:Pokemon)-[:CAN_LEARN]->(m:Move)
WHERE p.is_default
WITH DISTINCT m, p
WITH m, collect(p) AS aprendices
UNWIND aprendices AS a UNWIND aprendices AS b
WITH a, b WHERE a.id < b.id
WITH a, b, count(*) AS comunes WHERE comunes > 120
RETURN a.identifier AS pokemon_a, b.identifier AS pokemon_b, comunes
ORDER BY comunes DESC LIMIT 20;

// P8 - ¿Que areas concentran mas biodiversidad de especies? (agregacion sobre encuentros reificados)
// ~45% de las LocationArea no traen identifier propio, asi que se cae al nombre de la Location padre.
MATCH (la:LocationArea)<-[:AT_AREA]-(:Encounter)<-[:HAS_ENCOUNTER]-(p:Pokemon)-[:IS_SPECIES]->(s:Species)
OPTIONAL MATCH (la)-[:IN_LOCATION]->(loc:Location)
WITH la, loc, count(DISTINCT s) AS biodiversidad
RETURN coalesce(la.identifier, loc.identifier, toString(la.id)) AS area, biodiversidad
ORDER BY biodiversidad DESC LIMIT 15;

// P9 - ¿En que linaje se gana mas poder de la forma base a la final? (recursion + agregacion)
// Recorre el path completo raiz->hoja (EVOLVES_TO*) y suma la ganancia de stats de punta a punta.
MATCH path = (raiz:Species)-[:EVOLVES_TO*]->(hoja:Species)
WHERE NOT (:Species)-[:EVOLVES_TO]->(raiz) AND NOT (hoja)-[:EVOLVES_TO]->(:Species)
MATCH (pr:Pokemon {is_default:true})-[:IS_SPECIES]->(raiz)
MATCH (ph:Pokemon {is_default:true})-[:IS_SPECIES]->(hoja)
MATCH (pr)-[r1:HAS_STAT]->(s:Stat)<-[r2:HAS_STAT]-(ph)
WITH [n IN nodes(path) | n.identifier] AS linaje, length(path) AS pasos,
     sum(r2.base_stat - r1.base_stat) AS ganancia
RETURN linaje, pasos, ganancia ORDER BY ganancia DESC LIMIT 20;
