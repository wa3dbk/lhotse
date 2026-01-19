from typing import List, Optional

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.indic_muls import (
    ALL_GENDERS,
    INDIC_MULS_LANGUAGES,
    download_indic_muls,
    prepare_indic_muls,
)
from lhotse.utils import Pathlike

__all__ = ["indic_muls"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--languages",
    type=click.Choice(INDIC_MULS_LANGUAGES),
    multiple=True,
    default=list(INDIC_MULS_LANGUAGES),
    help="Which languages to prepare (default: all). "
    "Available: gujarati, kannada, malayalam, marathi, tamil, telugu.",
)
@click.option(
    "-g",
    "--genders",
    type=click.Choice(ALL_GENDERS),
    multiple=True,
    default=list(ALL_GENDERS),
    help="Which genders to prepare (default: all). "
    "Note: Marathi only has female speakers.",
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "lower"], case_sensitive=False),
    default="none",
    help="Type of text normalization to apply (default: none). "
    "Note: 'lower' is not recommended for Indic scripts.",
)
def indic_muls(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    languages: List[str],
    genders: List[str],
    normalize_text: str,
):
    """
    Indic Multi-Speaker speech corpora manifest preparation.

    This dataset contains multi-speaker recordings for six Indian languages,
    intended for TTS but also usable for ASR:

    \b
    - Gujarati (SLR78): ~8h total (female + male)
    - Kannada (SLR79): ~8.5h total (female + male)
    - Malayalam (SLR63): ~5.5h total (female + male)
    - Marathi (SLR64): ~3h (female only)
    - Tamil (SLR65): ~7h total (female + male)
    - Telugu (SLR66): ~5.7h total (female + male)

    CORPUS_DIR should point to the directory containing the extracted data
    (with subdirectories like gu_in_female, gu_in_male, kn_in_female, etc.).

    More info: https://www.openslr.org/resources.php (SLR63-66, 78-79)
    """
    prepare_indic_muls(
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
    type=click.Choice(INDIC_MULS_LANGUAGES),
    multiple=True,
    default=list(INDIC_MULS_LANGUAGES),
    help="Which languages to download (default: all). "
    "Available: gujarati, kannada, malayalam, marathi, tamil, telugu.",
)
@click.option(
    "-g",
    "--genders",
    type=click.Choice(ALL_GENDERS),
    multiple=True,
    default=list(ALL_GENDERS),
    help="Which genders to download (default: all). "
    "Note: Marathi only has female speakers.",
)
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Force re-download even if files already exist.",
)
def indic_muls(
    target_dir: Pathlike,
    languages: List[str],
    genders: List[str],
    force_download: bool,
):
    """
    Indic Multi-Speaker dataset download from OpenSLR.

    Downloads multi-speaker speech corpora for Indian languages:

    \b
    - Gujarati (SLR78): ~350MB per gender
    - Kannada (SLR79): ~350MB per gender
    - Malayalam (SLR63): ~250MB per gender
    - Marathi (SLR64): ~250MB (female only)
    - Tamil (SLR65): ~300MB per gender
    - Telugu (SLR66): ~250MB per gender

    Total size: ~3.5GB when downloading all languages and genders.

    Reference: He et al. (2020) "Open-source Multi-speaker Speech Corpora
    for Building Gujarati, Kannada, Malayalam, Marathi, Tamil and Telugu
    Speech Synthesis Systems" (LREC 2020)
    """
    download_indic_muls(
        target_dir=target_dir,
        languages=languages if languages else "all",
        genders=genders if genders else "all",
        force_download=force_download,
    )
