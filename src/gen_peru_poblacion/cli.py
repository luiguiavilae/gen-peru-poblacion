"""
CLI de gen-peru-poblacion.

Comandos:
  gen-peru-poblacion generate  → genera perfiles sintéticos
  gen-peru-poblacion verify    → verifica KS similarity contra fuentes
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="gen-peru-poblacion",
    help="Generador de poblaciones sintéticas de emprendedores peruanos (MYPE v1).",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True, style="bold red")

_FORMATOS = ["csv", "json", "jsonl"]
_SEGMENTOS = ["mype"]
_REGIONES = [
    "lima_metropolitana", "costa_norte", "costa_sur",
    "sierra_norte", "sierra_centro", "sierra_sur", "selva",
]


# ─────────────────────────────────────────────────────────────────────────────
# Comando: generate
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def generate(
    segmento: str = typer.Option(
        "mype",
        "--segmento", "-s",
        help=f"Segmento de población. Opciones: {_SEGMENTOS}",
        show_default=True,
    ),
    n: int = typer.Option(
        100,
        "--n",
        help="Número de perfiles a generar.",
        show_default=True,
        min=1,
        max=10_000,
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region", "-r",
        help=(
            f"Filtrar por región. Opciones: {', '.join(_REGIONES)}. "
            "Si no se especifica, se generan todas las regiones según distribución real."
        ),
    ),
    formato: str = typer.Option(
        "jsonl",
        "--formato", "-f",
        help=f"Formato de salida. Opciones: {_FORMATOS}",
        show_default=True,
    ),
    output: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Directorio de salida donde se guardará el archivo.",
        show_default=True,
    ),
    fuente_dir: Path = typer.Option(
        Path("data/fuentes"),
        "--fuente-dir",
        help="Directorio con los JSONs de distribuciones fuente.",
        show_default=True,
        hidden=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mostrar detalle de la ejecución."),
) -> None:
    """
    Genera perfiles sintéticos de emprendedores peruanos calibrados con datos INEI/PRODUCE/SBS.

    Ejemplos:

    \b
      gen-peru-poblacion generate --segmento mype --n 100 --formato csv
      gen-peru-poblacion generate --segmento mype --region lima_metropolitana --n 50 --formato jsonl
      gen-peru-poblacion generate --n 200 --formato json --output ./datos
    """
    # Validaciones tempranas con mensajes útiles
    if segmento not in _SEGMENTOS:
        err_console.print(f"Error: segmento '{segmento}' no válido. Opciones: {_SEGMENTOS}")
        raise typer.Exit(code=1)

    if formato not in _FORMATOS:
        err_console.print(f"Error: formato '{formato}' no válido. Opciones: {_FORMATOS}")
        raise typer.Exit(code=1)

    if region is not None and region not in _REGIONES:
        err_console.print(f"Error: región '{region}' no válida. Opciones: {_REGIONES}")
        raise typer.Exit(code=1)

    from gen_peru_poblacion.config import Config
    from gen_peru_poblacion.generator import PopulationGenerator

    cfg = Config(
        segmento=segmento,
        n=n,
        region=region,
        output_dir=str(output),
        formato=formato,
        fuente_dir=str(fuente_dir),
    )

    dest = output / f"{segmento}.{formato}"

    if verbose:
        console.print(Panel(
            f"[bold]Segmento:[/bold] {segmento}\n"
            f"[bold]N:[/bold] {n}\n"
            f"[bold]Región:[/bold] {region or 'todas'}\n"
            f"[bold]Formato:[/bold] {formato}\n"
            f"[bold]Salida:[/bold] {dest}",
            title="Configuración",
        ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Calibrando modelo SDV...", total=None)
            gen = PopulationGenerator(cfg)

            progress.update(task, description=f"Generando {n} perfiles...")
            perfiles = gen.generate()

            progress.update(task, description=f"Exportando a {formato.upper()}...")
            path = gen.export(perfiles, formato=formato, path=dest)

        console.print(f"[green]✓[/green] {len(perfiles)} perfiles generados → [bold]{path}[/bold]")

        if verbose:
            _print_sample(perfiles[:3])

    except FileNotFoundError as e:
        err_console.print(f"Error: archivo de fuente no encontrado — {e}")
        err_console.print("¿Estás ejecutando desde la raíz del proyecto?")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"Error inesperado: {e}")
        if verbose:
            import traceback
            console.print_exception()
        raise typer.Exit(code=1)


# ─────────────────────────────────────────────────────────────────────────────
# Comando: verify
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def verify(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir", "-d",
        help="Directorio con los perfiles generados (CSV/JSONL/JSON).",
    ),
    fuente_dir: Path = typer.Option(
        Path("data/fuentes"),
        "--fuente-dir",
        help="Directorio con los JSONs de distribuciones fuente.",
        show_default=True,
    ),
    segmento: str = typer.Option(
        "mype",
        "--segmento", "-s",
        help="Segmento a verificar.",
        show_default=True,
    ),
    threshold: float = typer.Option(
        0.70,
        "--threshold",
        help="Umbral de KS similarity para PASS/FAIL (0.0–1.0).",
        show_default=True,
        min=0.0,
        max=1.0,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mostrar tabla detallada por variable."),
) -> None:
    """
    Verifica la KS similarity entre perfiles generados y las distribuciones fuente INEI/PRODUCE/SBS.

    Ejemplos:

    \b
      gen-peru-poblacion verify --data-dir ./output
      gen-peru-poblacion verify --data-dir ./output --verbose
      gen-peru-poblacion verify --data-dir ./datos --threshold 0.80
    """
    if not data_dir.exists():
        err_console.print(f"Error: directorio '{data_dir}' no existe.")
        raise typer.Exit(code=1)

    from gen_peru_poblacion.verify import run_ks_check

    try:
        report = run_ks_check(
            data_dir=data_dir,
            fuente_dir=fuente_dir,
            segmento=segmento,
            threshold=threshold,
        )
    except FileNotFoundError as e:
        err_console.print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"Error inesperado: {e}")
        raise typer.Exit(code=1)

    # Score global
    status_color = "green" if report.passed else "red"
    status_label = "PASS ✓" if report.passed else "FAIL ✗"
    console.print(
        f"\n[bold]KS Similarity Report[/bold] — {report.n_generated} perfiles  "
        f"[{status_color}]{status_label}[/{status_color}]"
    )
    console.print(
        f"Score global: [{status_color}]{report.global_score:.3f}[/{status_color}]  "
        f"(umbral: {report.threshold:.2f})"
    )

    if verbose or not report.passed:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Variable", style="cyan", min_width=20)
        table.add_column("KS stat", justify="right")
        table.add_column("Similarity", justify="right")
        table.add_column("Estado", justify="center")

        for s in report.scores:
            estado = "[green]PASS[/green]" if s.passed else "[red]FAIL[/red]"
            table.add_row(
                s.variable,
                f"{s.ks_statistic:.4f}",
                f"{s.similarity:.4f}",
                estado,
            )
        console.print(table)

    raise typer.Exit(code=0 if report.passed else 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_sample(perfiles: list[dict]) -> None:
    """Imprime los primeros N perfiles como tabla compacta."""
    if not perfiles:
        return
    table = Table(title="Muestra de perfiles generados", show_lines=True)
    cols = ["perfil_id", "region", "rubro", "edad_dueño", "adopcion_digital", "ingreso_mensual_soles"]
    for col in cols:
        table.add_column(col, overflow="fold")
    for p in perfiles:
        table.add_row(*[str(p.get(c, ""))[:30] for c in cols])
    console.print(table)


if __name__ == "__main__":
    app()
