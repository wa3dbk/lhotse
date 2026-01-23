from typing import List, Optional

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.iberian_muls import (
    ALL_GENDERS,
    IBERIAN_MULS_LANGUAGES,
    download_iberian_muls,
    prepare_iberian_muls,
)
from lhotse.utils import Pathlike

__all__ = ["iberian_muls"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--languages",
    type=click.Choice(IBERIAN_MULS_LANGUAGES),
    multiple=True,
    default=list(IBERIAN_MULS_LANGUAGES),
    help="Which languages to prepare (default: all). "
    "Available: basque, catalan, galician.",
)
@click.option(
    "-g",
    "--genders",
    type=click.Choice(ALL_GENDERS),
    multiple=True,
    default=list(ALL_GENDERS),
    help="Which genders to prepare (default: all).",
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "lower"], case_sensitive=False),
    default="none",
    help="Type of text normalization to apply (default: none).",
)
def iberian_muls(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    languages: List[str],
    genders: List[str],
    normalize_text: str,
):
    """
    Iberian Multi-Speaker speech corpora manifest preparation.

    This dataset contains multi-speaker recordings for three Iberian languages,
    intended for TTS but also usable for ASR:

    \b
    - Basque (SLR76): ~14h total (female + male), 52 speakers
    - Catalan (SLR69): ~9.4h total (female + male), 36 speakers
    - Galician (SLR77): ~10.3h total (female + male), 44 speakers

    Total: ~33.5 hours from 132 speakers.

    CORPUS_DIR should point to the directory containing the extracted data
    (with subdirectories like ca_es_female, ca_es_male, eu_es_female, etc.).

    More info: https://www.openslr.org/resources.php (SLR69, 76, 77)
    """
    prepare_iberian_muls(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        languages=languages if languages else "all",
        genders=genders if genders else "all",
        normalize_text=normalize_text,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-l",
    "--languages",
    type=click.Choice(IBERIAN_MULS_LANGUAGES),
    multiple=True,
    default=list(IBERIAN_MULS_LANGUAGES),
    help="Which languages to download (default: all). "
    "Available: basque, catalan, galician.",
)
@click.option(
    "-g",
    "--genders",
    type=click.Choice(ALL_GENDERS),
    multiple=True,
    default=list(ALL_GENDERS),
    help="Which genders to download (default: all).",
)
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Force re-download even if files already exist.",
)
def iberian_muls(
    target_dir: Pathlike,
    languages: List[str],
    genders: List[str],
    force_download: bool,
):
    """
    Iberian Multi-Speaker dataset download from OpenSLR.

    Downloads multi-speaker speech corpora for Iberian languages:

    \b
    - Basque (SLR76): ~600MB total (female + male)
    - Catalan (SLR69): ~400MB total (female + male)
    - Galician (SLR77): ~450MB total (female + male)

    Total size: ~1.5GB when downloading all languages and genders.

    Reference: Kjartansson et al. (2020) "Open-Source High Quality Speech
    Datasets for Basque, Catalan and Galician" (SLTU-CCURL 2020)
    """
    download_iberian_muls(
        target_dir=target_dir,
        languages=languages if languages else "all",
        genders=genders if genders else "all",
        force_download=force_download,
    )
