"""
Indic Multi-Speaker Speech Corpora (indic_muls)

This recipe supports the multi-speaker speech corpora for Indian languages
released by Google, covering:
- Gujarati (SLR78): ~8h (female + male)
- Kannada (SLR79): ~8.5h (female + male)
- Malayalam (SLR63): ~5.5h (female + male)
- Marathi (SLR64): ~3h (female only)
- Tamil (SLR65): ~7h (female + male)
- Telugu (SLR66): ~5.7h (female + male)

The corpora are primarily intended for text-to-speech (TTS) applications
but can also be used for ASR and other speech tasks. Audio is recorded
at 48kHz, 16-bit mono.

Dataset URLs:
- Gujarati: https://www.openslr.org/78/
- Kannada: https://www.openslr.org/79/
- Malayalam: https://www.openslr.org/63/
- Marathi: https://www.openslr.org/64/
- Tamil: https://www.openslr.org/65/
- Telugu: https://www.openslr.org/66/

Usage example:
    >>> from lhotse.recipes.indic_muls import download_indic_muls, prepare_indic_muls
    >>> # Download all languages
    >>> corpus_dirs = download_indic_muls(target_dir="./data", languages="all")
    >>> # Prepare manifests
    >>> manifests = prepare_indic_muls(corpus_dir="./data", output_dir="./manifests")
    >>> # Access specific language/gender
    >>> telugu_female = manifests["telugu"]["female"]
    >>> recordings = telugu_female["recordings"]
    >>> supervisions = telugu_female["supervisions"]

Reference:
    @inproceedings{he-etal-2020-open,
        title = "Open-source Multi-speaker Speech Corpora for Building {G}ujarati,
                 {K}annada, {M}alayalam, {M}arathi, {T}amil and {T}elugu Speech
                 Synthesis Systems",
        author = "He, Fei and Chu, Shan-Hui Cathy and Kjartansson, Oddur and
                  Rivera, Clara and Katanova, Anna and Gutkin, Alexander and
                  Demirs{\\c{s}}ahin, I{\\c{s}}in and Johny, Cibu and Jansche, Martin and
                  Sarin, Supheakmungkol and Pipatsrisawat, Knot",
        booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
        year = "2020",
        address = "Marseille, France",
        publisher = "European Language Resources Association",
        pages = "6494--6503",
    }
"""
import logging
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download

# Language configurations
INDIC_MULS_LANGUAGES = ("gujarati", "kannada", "malayalam", "marathi", "tamil", "telugu")

# Language to ISO 639-1 code mapping
LANG_TO_CODE = {
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
}

# Language to full language name (for SupervisionSegment)
LANG_TO_NAME = {
    "gujarati": "Gujarati",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "marathi": "Marathi",
    "tamil": "Tamil",
    "telugu": "Telugu",
}

# Language to SLR ID mapping
LANG_TO_SLR = {
    "gujarati": "78",
    "kannada": "79",
    "malayalam": "63",
    "marathi": "64",
    "tamil": "65",
    "telugu": "66",
}

# Available genders per language (Marathi only has female)
LANG_GENDERS = {
    "gujarati": ("female", "male"),
    "kannada": ("female", "male"),
    "malayalam": ("female", "male"),
    "marathi": ("female",),
    "tamil": ("female", "male"),
    "telugu": ("female", "male"),
}

# All possible genders
ALL_GENDERS = ("female", "male")

# Base URL for OpenSLR
BASE_URL = "https://openslr.trmal.net/resources"


def _get_split_name(lang_code: str, gender: str) -> str:
    """Get the split name (e.g., 'gu_in_female')."""
    gender_suffix = gender[0]  # 'f' or 'm'
    return f"{lang_code}_in_{gender}"


def _get_download_url(lang: str, gender: str) -> str:
    """Get the download URL for a specific language and gender."""
    lang_code = LANG_TO_CODE[lang]
    slr_id = LANG_TO_SLR[lang]
    split_name = _get_split_name(lang_code, gender)
    return f"{BASE_URL}/{slr_id}/{split_name}.zip"


def download_indic_muls(
    target_dir: Pathlike = ".",
    languages: Union[str, Sequence[str]] = "all",
    genders: Union[str, Sequence[str]] = "all",
    force_download: bool = False,
) -> Dict[str, Dict[str, Path]]:
    """
    Download and extract the Indic Multi-Speaker dataset.

    :param target_dir: Directory to store the downloaded data.
    :param languages: Which languages to download. Can be "all", a single language name,
        or a sequence of language names. Valid languages: gujarati, kannada, malayalam,
        marathi, tamil, telugu.
    :param genders: Which genders to download. Can be "all", "female", "male", or a
        sequence. Note: Marathi only has female speakers.
    :param force_download: If True, re-download even if files exist.
    :return: Nested dict mapping language -> gender -> extracted directory path.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if languages == "all":
        languages = list(INDIC_MULS_LANGUAGES)
    elif isinstance(languages, str):
        languages = [languages]
    languages = [lang.lower() for lang in languages]

    if genders == "all":
        genders = list(ALL_GENDERS)
    elif isinstance(genders, str):
        genders = [genders]
    genders = [g.lower() for g in genders]

    corpus_dirs = defaultdict(dict)

    for lang in tqdm(languages, desc="Downloading Indic Multi-Speaker languages"):
        if lang not in INDIC_MULS_LANGUAGES:
            logging.warning(
                f"Unknown language: {lang}. Valid options: {INDIC_MULS_LANGUAGES}"
            )
            continue

        lang_code = LANG_TO_CODE[lang]
        available_genders = LANG_GENDERS[lang]

        for gender in genders:
            if gender not in available_genders:
                if gender == "male" and lang == "marathi":
                    logging.debug(f"Marathi only has female speakers, skipping male.")
                else:
                    logging.debug(f"Gender '{gender}' not available for {lang}.")
                continue

            split_name = _get_split_name(lang_code, gender)
            split_dir = target_dir / split_name
            completed_detector = split_dir / ".completed"

            if completed_detector.is_file() and not force_download:
                logging.info(f"Skipping {split_name} because {completed_detector} exists.")
                corpus_dirs[lang][gender] = split_dir
                continue

            # Construct download URL
            url = _get_download_url(lang, gender)
            zip_path = target_dir / f"{split_name}.zip"

            # Download
            logging.info(f"Downloading {url}...")
            resumable_download(url, filename=zip_path, force_download=force_download)

            # Extract - the zip contains files directly, so we extract to a subdirectory
            shutil.rmtree(split_dir, ignore_errors=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Extracting {zip_path} to {split_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(split_dir)

            completed_detector.touch()
            corpus_dirs[lang][gender] = split_dir

    return dict(corpus_dirs)


def prepare_indic_muls(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    languages: Union[str, Sequence[str]] = "all",
    genders: Union[str, Sequence[str]] = "all",
    normalize_text: str = "none",
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Prepare manifests for the Indic Multi-Speaker dataset.

    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Path to the directory containing the extracted data.
        This should contain subdirectories like gu_in_female, gu_in_male, etc.
    :param output_dir: Path where the manifests should be written.
    :param languages: Which languages to prepare. Can be "all", a single language,
        or a sequence of languages.
    :param genders: Which genders to prepare. Can be "all", "female", "male",
        or a sequence. Note: Marathi only has female speakers.
    :param normalize_text: Text normalization mode. "none" keeps original text,
        "lower" converts to lowercase (not recommended for Indic scripts).
    :return: Nested dict: {language: {gender: {"recordings": ..., "supervisions": ...}}}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Parse languages
    if languages == "all":
        languages = list(INDIC_MULS_LANGUAGES)
    elif isinstance(languages, str):
        languages = [languages]
    languages = [lang.lower() for lang in languages]

    # Parse genders
    if genders == "all":
        genders = list(ALL_GENDERS)
    elif isinstance(genders, str):
        genders = [genders]
    genders = [g.lower() for g in genders]

    manifests = defaultdict(dict)

    for lang in tqdm(languages, desc="Processing Indic Multi-Speaker languages"):
        if lang not in INDIC_MULS_LANGUAGES:
            logging.warning(f"Unknown language: {lang}. Skipping.")
            continue

        lang_code = LANG_TO_CODE[lang]
        available_genders = LANG_GENDERS[lang]

        for gender in genders:
            if gender not in available_genders:
                logging.debug(f"Gender '{gender}' not available for {lang}. Skipping.")
                continue

            split_name = _get_split_name(lang_code, gender)
            part_id = f"{lang}_{gender}"

            # Check if manifests already exist
            if output_dir is not None and manifests_exist(
                part=part_id, output_dir=output_dir, prefix="indic_muls"
            ):
                logging.info(
                    f"Indic Multi-Speaker {lang} {gender} manifests already exist - skipping."
                )
                # Read cached manifests
                cached = read_manifests_if_cached(
                    dataset_parts=[part_id],
                    output_dir=output_dir,
                    prefix="indic_muls",
                )
                if cached:
                    manifests[lang][gender] = cached[part_id]
                continue

            # Find the split directory
            split_dir = corpus_dir / split_name
            if not split_dir.is_dir():
                logging.warning(f"Directory not found: {split_dir}. Skipping.")
                continue

            logging.info(f"Processing {lang} {gender}...")

            # Prepare manifests
            recordings, supervisions = _prepare_split(
                split_dir, lang, gender, normalize_text
            )

            if len(recordings) == 0:
                logging.warning(f"No recordings found for {lang} {gender}. Skipping.")
                continue

            # Fix and validate manifests
            recordings, supervisions = fix_manifests(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)

            # Save manifests
            if output_dir is not None:
                recordings.to_file(
                    output_dir / f"indic_muls_recordings_{part_id}.jsonl.gz"
                )
                supervisions.to_file(
                    output_dir / f"indic_muls_supervisions_{part_id}.jsonl.gz"
                )

            manifests[lang][gender] = {
                "recordings": recordings,
                "supervisions": supervisions,
            }

    return dict(manifests)


def _parse_line_index(line_index_path: Path) -> Dict[str, str]:
    """
    Parse the line_index.tsv file.

    Format: segid<TAB>transcription
    Example: tef_01033_00351357063	ఈ వివాదం సెప్టెంబర్...
    """
    texts = {}
    with open(line_index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) >= 2:
                utt_id, text = parts
                texts[utt_id] = text
            elif len(parts) == 1:
                # No transcription
                texts[parts[0]] = ""
    return texts


def _extract_speaker_id(utt_id: str) -> str:
    """
    Extract speaker ID from utterance ID.

    Utterance ID format: {lang_gender}_{speaker_id}_{hash}
    Example: tef_01033_00351357063 -> tef_01033
    """
    parts = utt_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return utt_id


def _prepare_split(
    split_dir: Path,
    lang: str,
    gender: str,
    normalize_text: str,
) -> Tuple[RecordingSet, SupervisionSet]:
    """Prepare a single split (language + gender combination)."""
    recordings = []
    supervisions = []

    # Find line_index.tsv file
    line_index_path = split_dir / "line_index.tsv"
    if not line_index_path.exists():
        logging.warning(f"line_index.tsv not found in {split_dir}")
        return RecordingSet.from_recordings([]), SupervisionSet.from_segments([])

    # Parse transcriptions
    texts = _parse_line_index(line_index_path)
    logging.info(f"Found {len(texts)} transcriptions in {line_index_path}")

    # Process wav files
    wav_files = list(split_dir.glob("*.wav"))
    logging.info(f"Found {len(wav_files)} wav files in {split_dir}")

    for wav_path in tqdm(sorted(wav_files), desc=f"Processing {lang} {gender}", leave=False):
        utt_id = wav_path.stem

        if utt_id not in texts:
            logging.debug(f"No transcription for {utt_id}, skipping.")
            continue

        try:
            recording = Recording.from_file(wav_path, recording_id=utt_id)
        except Exception as e:
            logging.warning(f"Failed to read {wav_path}: {e}")
            continue

        text = texts[utt_id]
        if normalize_text == "lower":
            text = text.lower()

        speaker_id = _extract_speaker_id(utt_id)

        supervision = SupervisionSegment(
            id=utt_id,
            recording_id=utt_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            text=text,
            language=LANG_TO_NAME[lang],
            speaker=speaker_id,
            gender=gender[0],  # 'f' or 'm'
            custom={"gender_full": gender},
        )

        recordings.append(recording)
        supervisions.append(supervision)

    return RecordingSet.from_recordings(recordings), SupervisionSet.from_segments(
        supervisions
    )
