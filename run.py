"""Lanzador multiplataforma del grafo PokeAPI (Windows / macOS / Linux).

Hace lo mismo que pipeline/load_all.sh pero en Python, asi corre igual en las tres plataformas
con solo Docker Desktop + Python instalados. No usa bind-mounts (los copia con `docker cp`)
para evitar los problemas de rutas y permisos de Windows.

    python run.py

Levanta Neo4j 5 + GDS en Docker, carga los ~131k nodos / ~902k aristas y verifica.
UI: http://localhost:7474 (sin auth). Si falta el clon de PokeAPI, lo descarga solo.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline"
CSV = ROOT / "pokeapi" / "data" / "v2" / "csv"
NAME = "neo4j-pokeapi"

# Tablas que carga el pipeline (las demas del repo no se usan).
TABLES = [
    "pokemon", "pokemon_species", "moves", "abilities", "types", "items", "egg_groups",
    "generations", "regions", "locations", "location_areas", "stats", "pokemon_types",
    "pokemon_abilities", "pokemon_stats", "pokemon_moves", "encounters", "pokemon_egg_groups",
    "type_efficacy", "pokemon_evolution", "natures", "versions", "pokemon_forms",
    "pokemon_species_names", "move_names", "item_names", "ability_names", "type_names",
    "location_names", "region_names", "generation_names", "nature_names", "version_names",
    "pokemon_form_names",
]
SCRIPTS = ["01_constraints", "02_nodes", "03_relationships", "04_derived", "05_scale"]


def run(args, **kw):
    """Corre un comando y devuelve el CompletedProcess (sin shell, portable)."""
    return subprocess.run(args, **kw)


def docker(*args, capture=False, check=True):
    return run(["docker", *args], check=check,
               capture_output=capture, text=True)


def ensure_dataset():
    """Si falta el clon de PokeAPI, lo descarga (shallow). Requiere git en el PATH."""
    if CSV.is_dir():
        return
    print(">> no encuentro los CSV de PokeAPI, clonando el repo (shallow, ~48M)...")
    run(["git", "clone", "--depth", "1", "https://github.com/PokeAPI/pokeapi.git",
         str(ROOT / "pokeapi")], check=True)
    if not CSV.is_dir():
        sys.exit(f"ERROR: no aparecieron los CSV en {CSV}")


def wait_ready(timeout=180):
    print(">> esperando a que Neo4j acepte conexiones...")
    for i in range(timeout // 3):
        p = docker("exec", NAME, "cypher-shell", "RETURN 1;", capture=True, check=False)
        if p.returncode == 0:
            print(f"   listo (intento {i + 1})")
            return
        time.sleep(3)
    sys.exit("ERROR: Neo4j no respondio a tiempo")


def cypher_file(path_in_container):
    docker("exec", "-i", NAME, "cypher-shell", "--format", "plain", "-f", path_in_container)


def main():
    # docker disponible?
    if run(["docker", "--version"], capture_output=True, check=False).returncode != 0:
        sys.exit("ERROR: no encuentro 'docker'. Instala Docker Desktop y abrelo.")
    ensure_dataset()

    print(">> limpiando contenedor previo (si existe)")
    docker("rm", "-f", "-v", NAME, check=False, capture=True)

    print(">> levantando Neo4j 5 + GDS (puertos 7474 UI / 7687 bolt)")
    docker(
        "run", "-d", "--name", NAME,
        "-p", "7474:7474", "-p", "7687:7687",
        "-e", "NEO4J_AUTH=none",
        "-e", 'NEO4J_PLUGINS=["graph-data-science"]',
        "-e", "NEO4J_dbms_security_procedures_unrestricted=gds.*",
        "-e", "NEO4J_dbms_security_procedures_allowlist=gds.*",
        "-e", "NEO4J_server_memory_heap_max__size=2G",
        "-e", "NEO4J_server_memory_pagecache_size=1G",
        "neo4j:5", capture=True,
    )
    wait_ready()

    print(">> copiando CSV y scripts al contenedor")
    dst = f"{NAME}:/var/lib/neo4j/import/"
    # Se copia con cwd en la carpeta de origen y nombre relativo. En Windows una ruta
    # absoluta tipo C:\... confunde a `docker cp` (toma C: como nombre de contenedor).
    for t in TABLES:
        run(["docker", "cp", f"{t}.csv", dst], cwd=str(CSV), check=True, capture_output=True)
    for s in SCRIPTS + ["verify"]:
        run(["docker", "cp", f"{s}.cypher", dst], cwd=str(PIPE), check=True, capture_output=True)

    for s in SCRIPTS:
        print(f">> {s}.cypher")
        cypher_file(f"/var/lib/neo4j/import/{s}.cypher")

    print(">> verificacion")
    cypher_file("/var/lib/neo4j/import/verify.cypher")
    print(">> LISTO. UI http://localhost:7474 (sin auth). Consultas en pipeline/queries.cypher")


if __name__ == "__main__":
    main()
