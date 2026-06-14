from typing import Optional

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.armenian_crowdsourced import (
    download_armenian_crowdsourced,
    prepare_armenian_crowdsourced,
)
from lhotse.utils import Pathlike

__all__ = ["armenian_crowdsourced"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "lower"], case_sensitive=False),
    default="none",
    help="Type of text normalization to apply (default: none). "
    "Use 'lower' to convert transcripts to lowercase.",
)
def armenian_crowdsourced(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    normalize_text: str,
):
    """
    Armenian Crowdsourced Speech Data (SLR160) manifest preparation.

    Crowdsourced Armenian speech recordings with transcriptions.

    CORPUS_DIR should point to the extracted armenian_speech_crowdsourcing_data
    directory containing pitched.jsonl and the pitched/ audio subdirectory.

    More info: https://openslr.trmal.net/resources/160/
    """
    prepare_armenian_crowdsourced(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        normalize_text=normalize_text,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Force re-download even if files already exist.",
)
def armenian_crowdsourced(
    target_dir: Pathlike,
    force_download: bool,
):
    """
    Armenian Crowdsourced Speech dataset download (SLR160 from OpenSLR).

    Downloads crowdsourced Armenian speech data.

    More info: https://openslr.trmal.net/resources/160/
    """
    download_armenian_crowdsourced(
        target_dir=target_dir,
        force_download=force_download,
    )
