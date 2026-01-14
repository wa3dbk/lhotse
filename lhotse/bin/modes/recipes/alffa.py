from typing import List, Optional

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.alffa import (
    ALFFA_LANGUAGES,
    LANG_SPLITS,
    download_alffa,
    prepare_alffa,
)
from lhotse.utils import Pathlike

__all__ = ["alffa"]

# Collect all possible splits across languages
ALL_SPLITS = tuple(sorted(set(split for splits in LANG_SPLITS.values() for split in splits)))


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--languages",
    type=click.Choice(ALFFA_LANGUAGES),
    multiple=True,
    default=list(ALFFA_LANGUAGES),
    help="Which ALFFA languages to prepare (by default all: amharic, swahili, wolof).",
)
@click.option(
    "-p",
    "--parts",
    type=click.Choice(ALL_SPLITS),
    multiple=True,
    default=list(ALL_SPLITS),
    help="Which dataset splits to prepare (by default all: train, dev, test). "
    "Note: 'dev' split is only available for Wolof.",
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "lower"], case_sensitive=False),
    default="none",
    help="Type of text normalization to apply (default: none). "
    "Use 'lower' to convert transcripts to lowercase.",
)
def alffa(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    languages: List[str],
    parts: List[str],
    normalize_text: str,
):
    """
    ALFFA (African Languages in the Field: speech Fundamentals and Automation)
    recording and supervision manifest preparation.

    ALFFA contains speech data for three African languages:

    \b
    - Amharic (~20h read speech): train, test splits
    - Swahili (~12h broadcast news): train, test splits
    - Wolof (~16h read speech): train, dev, test splits

    CORPUS_DIR should point to the directory containing the extracted ALFFA data
    (with subdirectories data_readspeech_am, data_broadcastnews_sw, data_readspeech_wo).

    More info: https://www.openslr.org/25/
    """
    prepare_alffa(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        languages=languages if languages else "all",
        dataset_parts=parts if parts else "all",
        normalize_text=normalize_text,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-l",
    "--languages",
    type=click.Choice(ALFFA_LANGUAGES),
    multiple=True,
    default=list(ALFFA_LANGUAGES),
    help="Which ALFFA languages to download (by default all: amharic, swahili, wolof).",
)
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Force re-download even if files already exist.",
)
def alffa(
    target_dir: Pathlike,
    languages: List[str],
    force_download: bool,
):
    """
    ALFFA dataset download (SLR25 from OpenSLR).

    Downloads speech data for African languages:

    \b
    - Amharic: ~1.0GB (read speech)
    - Swahili: ~1.2GB (broadcast news)
    - Wolof: ~1.7GB (read speech)

    Total size: ~4GB when downloading all languages.

    More info: https://www.openslr.org/25/
    """
    download_alffa(
        target_dir=target_dir,
        languages=languages if languages else "all",
        force_download=force_download,
    )
