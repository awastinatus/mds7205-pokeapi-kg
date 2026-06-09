# Grafo de conocimiento sobre PokeAPI

Proyecto del curso MDS7205 (Grafos de Conocimiento, Universidad de Chile). Modela el dataset
completo de PokeAPI como un grafo de propiedades en Neo4j, lo consulta con GQL/Cypher y corre
un ML basico sobre el.

El grafo cargado tiene **131.879 nodos** y **902.000 aristas**. Lo que lo vuelve un grafo util
y no un CSV glorificado son tres cosas que chequeamos sobre los datos: el cuadro de tipos tiene
ciclos de verdad (`fighting -> steel -> fairy -> fighting`, y hasta tipos fuertes contra si
mismos como `ghost` y `dragon`); la evolucion es recursiva, porque la self-FK
`evolves_from_species_id` apunta al pre-evolutivo; y la crianza por egg groups termina conectando
873 especies en un solo componente de 71.232 aristas para navegar.

## Requisitos

- **Docker Desktop** (Windows, macOS o Linux). Tiene que estar abierto.
- **Python 3.10+** y **git**.

Funciona igual en los tres sistemas: el cargador es Python y la base corre en un contenedor.

## Puesta en marcha

```bash
git clone <este-repo>
cd Proyecto

pip install -r requirements.txt

python run.py          # descarga PokeAPI si falta, levanta Neo4j+GDS y carga el grafo
```

`run.py` deja Neo4j corriendo. La interfaz web queda en http://localhost:7474 (sin usuario ni
clave). La primera corrida clona PokeAPI (~48M) y tarda unos minutos en cargar las ~900k aristas.

> En Windows usa Docker Desktop con el backend WSL2. No hace falta instalar bash: todos los
> comandos son `python ...`.

## Uso

```bash
# Las 9 consultas del proyecto: pegarlas en el Browser (localhost:7474) o
docker exec -i neo4j-pokeapi cypher-shell -f /var/lib/neo4j/import/queries.cypher

python analysis/eda.py     # caracterizacion del grafo (figuras en analysis/img/)
python analysis/ml.py      # clasificacion de tipo + link prediction de crianza
```

Para regenerar el reporte tecnico:

```bash
python analysis/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace analysis/reporte.ipynb
jupyter nbconvert --to html analysis/reporte.ipynb
```

## Estructura

```
pipeline/
  01_constraints.cypher   constraints de unicidad por label
  02_nodes.cypher         12 labels de entidad desde los CSV
  03_relationships.cypher relaciones con propiedades (CAN_LEARN, HAS_TYPE, ...)
  04_derived.cypher       EVOLVES_TO, SUPER_EFFECTIVE, COMPATIBLE, condiciones reificadas
  05_scale.cypher         nombres multilingues y encuentros como nodos
  queries.cypher          las 9 consultas P1-P9
  verify.cypher           conteos y chequeo de las 3 estructuras
  load_all.sh             atajo equivalente a run.py para Linux
analysis/
  eda.py                  exploracion y caracterizacion
  ml.py                   los dos modelos
  build_notebook.py       arma el reporte.ipynb
  reporte.ipynb / .pdf    el reporte tecnico entregable
run.py                    cargador multiplataforma (Windows/macOS/Linux)
```

## Modelo del grafo

Cada tabla-entidad es un label (`Pokemon`, `Species`, `Move`, `Type`, `Item`, ...). Las tablas
puente con columnas extra se vuelven relaciones con propiedades: `CAN_LEARN` carga nivel, metodo
y version del aprendizaje, y es un multigrafo (el mismo par pokemon-move se repite por version).
Los encuentros y las condiciones de evolucion se reifican como nodos (`Encounter`,
`EvolutionCondition`) porque son relaciones n-arias.

## Notas honestas

Dos resultados del ML hay que leerlos con cuidado, y se explican en el reporte:

- El link prediction de crianza da AUC ~1.0 por topologia, pero eso describe el grafo (es una
  union de cliques solapadas, una por egg group), no al modelo. Preguntar lo no trivial,
  predecir crianza desde el fenotipo, baja a ~0.67.
- La clasificacion de tipo llega a ~0.82, pero buena parte es el movepool (efecto STAB): con
  solo las stats base cae a ~0.20.

## Datos

Los datos vienen de [PokeAPI](https://github.com/PokeAPI/pokeapi) (licencia BSD). El clon no se
versiona en este repo; `run.py` lo descarga.
