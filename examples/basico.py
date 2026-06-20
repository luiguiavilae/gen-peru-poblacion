"""
Ejemplo básico: generar perfiles y exportarlos en los 3 formatos.
Requiere solo las dependencias core (sin [agents]).

Ejecutar desde la raíz del proyecto:
    python examples/basico.py
"""
from pathlib import Path

from gen_peru_poblacion import Config, PopulationGenerator

# 1. Configurar el generador
cfg = Config(
    segmento="mype",
    n=20,
    output_dir="./output_ejemplo",
)
gen = PopulationGenerator(cfg)

# 2. Generar los perfiles
print("Generando 20 perfiles MYPE...")
perfiles = gen.generate()
print(f"  → {len(perfiles)} perfiles generados")

# 3. Mostrar un perfil de muestra
p = perfiles[0]
print(f"\nEjemplo de perfil:")
print(f"  ID:          {p['perfil_id']}")
print(f"  Región:      {p['region']}")
print(f"  Rubro:       {p['rubro']}")
print(f"  Edad dueño:  {p['edad_dueño']}")
print(f"  Adopción:    {p['adopcion_digital']}")
print(f"  Ingreso:     S/ {p['ingreso_mensual_soles']:,}")
print(f"  Synthetic:   {p['synthetic']}")
print(f"  Fuentes:     {p['data_sources']}")

# 4. Exportar en los 3 formatos
out = Path("./output_ejemplo")
csv_path  = gen.export(perfiles, formato="csv",   path=out / "mype.csv")
json_path = gen.export(perfiles, formato="json",  path=out / "mype.json")
jsonl_path = gen.export(perfiles, formato="jsonl", path=out / "mype.jsonl")

print(f"\nArchivos generados:")
print(f"  CSV:   {csv_path}")
print(f"  JSON:  {json_path}")
print(f"  JSONL: {jsonl_path}")

# 5. Verificar KS similarity
from gen_peru_poblacion.verify import run_ks_check

print("\nVerificando KS similarity...")
report = run_ks_check(data_dir=out, fuente_dir="data/fuentes")
print(report)
