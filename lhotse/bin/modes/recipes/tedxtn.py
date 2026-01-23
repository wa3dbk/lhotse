from typing import List, Optional

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.tedxtn import (
    TEDXTN_LANGUAGES,
    TEDXTN_SPLITS,
    download_tedxtn,
    prepare_tedxtn,
)
from lhotse.utils import Pathlike

__all__ = ["tedxtn"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--langs",
    type=click.Choice(TEDXTN_LANGUAGES),
    multiple=True,
    default=["ta"],
    help="Language(s) to include. Use 'ta' for ASR (transcription only), "
    "or 'ta eng' for ST (speech translation with English translations). "
    "Default: ta (ASR mode).",
)
@click.option(
    "-p",
    "--parts",
    type=click.Choice(TEDXTN_SPLITS),
    multiple=True,
    default=list(TEDXTN_SPLITS),
    help="Which dataset splits to prepare (default: all).",
)
@click.option(
    "--normalize-text",
    is_flag=True,
    default=False,
    help="Apply text normalization to transcriptions (Arabic normalization, "
    "remove diacritics, punctuation, etc.).",
)
def tedxtn(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    langs: List[str],
    parts: List[str],
    normalize_text: bool,
):
    """
    TEDxTN corpus preparation for ASR and Speech Translation.

    TEDxTN is a three-way speech translation corpus containing TEDx talks
    in Tunisian Arabic with English translations. It features code-switching
    between Tunisian Arabic and other languages (French, English, MSA).

    \b
    Dataset statistics:
    - Train: ~X hours
    - Dev: ~X hours  
    - Test: ~X hours

    \b
    Modes:
    - ASR (default): Use --langs ta for transcription only
    - ST: Use --langs ta eng for speech translation with English translations

    \b
    CORPUS_DIR should point to the directory containing:
    - wav/ subdirectory with audio files (16kHz)
    - train.csv, dev.csv, test.csv (Arabic transcriptions)
    - train.en.csv, dev.en.csv, test.en.csv (English translations)

    Paper: https://aclanthology.org/2025.arabicnlp-main.22.pdf
    Dataset: https://huggingface.co/datasets/fbougares/TEDxTN
    """
    prepare_tedxtn(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        langs=list(langs) if langs else ["ta"],
        dataset_parts=list(parts) if parts else "all",
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
@click.option(
    "--no-audio",
    is_flag=True,
    default=False,
    help="Skip downloading audio files (only download CSV metadata).",
)
def tedxtn(
    target_dir: Pathlike,
    force_download: bool,
    no_audio: bool,
):
    """
    TEDxTN dataset download from HuggingFace.

    Downloads the TEDxTN speech translation corpus containing:
    - CSV files with transcriptions and translations
    - Audio files (WAV, 16kHz) from TEDxTN_URLS.txt

    \b
    The downloaded data will be organized as:
    target_dir/
    ├── wav/
    │   ├── Talk_Name_1.wav
    │   ├── Talk_Name_2.wav
    │   └── ...
    ├── train.csv
    ├── train.en.csv
    ├── dev.csv
    ├── dev.en.csv
    ├── test.csv
    ├── test.en.csv
    └── TEDxTN_URLS.txt

    Paper: https://aclanthology.org/2025.arabicnlp-main.22.pdf
    Dataset: https://huggingface.co/datasets/fbougares/TEDxTN
    """
    download_tedxtn(
        target_dir=target_dir,
        force_download=force_download,
        download_audio=not no_audio,
    )
