"""
TEDxTN: A Three-way Speech Translation Corpus for Code-Switched Tunisian Arabic-English.

This dataset contains TEDx talks in Tunisian Arabic with English translations,
featuring code-switching between Tunisian Arabic and other languages (French, English, MSA).

Paper: "TEDxTN: A Three-way Speech Translation Corpus for Code-Switched Tunisian Arabic-English"
       https://aclanthology.org/2025.arabicnlp-main.22.pdf

Dataset URL: https://huggingface.co/datasets/fbougares/TEDxTN

The dataset contains train, dev, and test splits with:
- Tunisian Arabic transcriptions (*.csv)
- English translations (*.en.csv)

Audio files should be downloaded separately from the URLs listed at:
https://huggingface.co/datasets/fbougares/TEDxTN/raw/main/TEDxTN_URLS.txt

Usage example:
    >>> from lhotse.recipes.tedxtn import download_tedxtn, prepare_tedxtn
    >>> # Download dataset (metadata + audio)
    >>> download_tedxtn(target_dir="./data/tedxtn")
    >>> # Prepare manifests for ASR
    >>> manifests = prepare_tedxtn(corpus_dir="./data/tedxtn", output_dir="./manifests", langs=["ta"])
    >>> # Prepare manifests for ST (with English translations)
    >>> manifests = prepare_tedxtn(corpus_dir="./data/tedxtn", output_dir="./manifests", langs=["ta", "eng"])
    >>> # Access specific split
    >>> train_recs = manifests["train"]["recordings"]
    >>> train_sups = manifests["train"]["supervisions"]

References:
    @inproceedings{bougares-etal-2025-tedxtn,
        title = "{TED}x{TN}: A Three-way Speech Translation Corpus for Code-Switched {T}unisian {A}rabic-{E}nglish",
        author = "Bougares, Fethi and Khalfallah, Salma and Gaido, Marco and Negri, Matteo",
        booktitle = "Proceedings of ArabicNLP 2025",
        year = "2025",
        url = "https://aclanthology.org/2025.arabicnlp-main.22",
    }
"""

import csv
import logging
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download

# Dataset configuration
TEDXTN_SPLITS = ("train", "dev", "test")

# HuggingFace dataset URLs
HF_BASE_URL = "https://huggingface.co/datasets/fbougares/TEDxTN/raw/main"
CSV_URLS = {
    "train": f"{HF_BASE_URL}/train.csv",
    "train_en": f"{HF_BASE_URL}/train.en.csv",
    "dev": f"{HF_BASE_URL}/dev.csv",
    "dev_en": f"{HF_BASE_URL}/dev.en.csv",
    "test": f"{HF_BASE_URL}/test.csv",
    "test_en": f"{HF_BASE_URL}/test.en.csv",
    "urls": f"{HF_BASE_URL}/TEDxTN_URLS.txt",
}

# Language codes
TEDXTN_LANGUAGES = ("ta", "eng")
LANG_TO_NAME = {
    "ta": "Tunisian Arabic",
    "eng": "English",
}


def download_tedxtn(
    target_dir: Pathlike,
    force_download: bool = False,
    download_audio: bool = True,
) -> Path:
    """
    Download the TEDxTN dataset.

    :param target_dir: Directory to store the downloaded data.
    :param force_download: If True, re-download even if files exist.
    :param download_audio: If True, also download audio files from TEDxTN_URLS.txt.
        Audio files will be placed in target_dir/wav/.
    :return: Path to the corpus directory.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = target_dir / ".completed"

    if completed_detector.is_file() and not force_download:
        logging.info(f"Skipping download because {completed_detector} exists.")
        return target_dir

    # Download CSV files
    logging.info("Downloading TEDxTN metadata files...")
    for name, url in tqdm(CSV_URLS.items(), desc="Downloading metadata"):
        if name == "urls":
            filename = "TEDxTN_URLS.txt"
        elif "_en" in name:
            split = name.replace("_en", "")
            filename = f"{split}.en.csv"
        else:
            filename = f"{name}.csv"

        target_path = target_dir / filename
        if target_path.exists() and not force_download:
            logging.debug(f"Skipping {filename} (already exists)")
            continue

        logging.info(f"Downloading {url}...")
        resumable_download(url, filename=target_path, force_download=force_download)

    # Download audio files if requested
    if download_audio:
        wav_dir = target_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        urls_file = target_dir / "TEDxTN_URLS.txt"
        if urls_file.exists():
            logging.info("Downloading TEDxTN audio files...")
            with open(urls_file, "r", encoding="utf-8") as f:
                audio_urls = [line.strip() for line in f if line.strip()]

            for url in tqdm(audio_urls, desc="Downloading audio"):
                # Extract filename from URL
                filename = url.split("/")[-1]
                target_path = wav_dir / filename

                if target_path.exists() and not force_download:
                    logging.debug(f"Skipping {filename} (already exists)")
                    continue

                try:
                    resumable_download(
                        url, filename=target_path, force_download=force_download
                    )
                except Exception as e:
                    logging.warning(f"Failed to download {url}: {e}")
        else:
            logging.warning(
                f"URLs file not found: {urls_file}. "
                "Please download audio files manually."
            )

    completed_detector.touch()
    return target_dir


def prepare_tedxtn(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    langs: Optional[List[str]] = None,
    dataset_parts: Union[str, Sequence[str]] = "all",
    normalize_text: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the TEDxTN dataset.

    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Path to the TEDxTN dataset directory containing:
        - wav/ subdirectory with audio files
        - train.csv, dev.csv, test.csv (Arabic transcriptions)
        - train.en.csv, dev.en.csv, test.en.csv (English translations, optional)
    :param output_dir: Directory where the manifests should be written. Can be omitted
        to avoid writing.
    :param langs: List of language abbreviations. Use ["ta"] for ASR-only (transcription),
        or ["ta", "eng"] for ST (speech translation with English translations).
        Defaults to ["ta"] for ASR-only mode.
    :param dataset_parts: Which splits to prepare. Can be "all", a single split name,
        or a sequence of split names. Valid: "train", "dev", "test".
    :param normalize_text: If True, apply text normalization to transcriptions.
    :return: Dict mapping split names to {"recordings": ..., "supervisions": ...}.
    """
    import soundfile as sf

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    # Default to ASR-only mode if langs not specified
    if langs is None:
        langs = ["ta"]

    # Determine if we need translations (ST mode)
    include_translations = len(langs) >= 2
    target_lang = langs[1] if include_translations else None
    source_lang = langs[0]

    # Parse dataset parts
    if dataset_parts == "all":
        dataset_parts = list(TEDXTN_SPLITS)
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached manifests
        manifest_prefix = f"tedxtn_{'_'.join(langs)}"
        if all(
            manifests_exist(
                part=part,
                output_dir=output_dir,
                prefix=manifest_prefix,
            )
            for part in dataset_parts
        ):
            logging.info("Found cached manifests, reading...")
            return read_manifests_if_cached(
                dataset_parts=dataset_parts,
                output_dir=output_dir,
                prefix=manifest_prefix,
            )

    audio_dir = corpus_dir / "wav"
    if not audio_dir.exists():
        raise ValueError(
            f"Audio directory not found: {audio_dir}. "
            f"Please ensure audio files are downloaded to {audio_dir}"
        )

    manifests = {}

    for split in tqdm(dataset_parts, desc="Processing TEDxTN splits"):
        if split not in TEDXTN_SPLITS:
            logging.warning(f"Unknown split: {split}. Skipping.")
            continue

        logging.info(f"Processing {split} split...")

        # Load Arabic transcriptions
        transcripts_path = corpus_dir / f"{split}.csv"
        if not transcripts_path.exists():
            logging.warning(
                f"Transcripts file not found: {transcripts_path}, skipping {split}"
            )
            continue

        transcripts = _load_csv(transcripts_path, is_translation=False)

        # Load English translations if needed
        translations_dict = {}
        if include_translations:
            translations_path = corpus_dir / f"{split}.en.csv"
            if translations_path.exists():
                translations = _load_csv(translations_path, is_translation=True)
                # Build lookup dictionary keyed by (audio_filename, start_time)
                for entry in translations:
                    key = (entry["audio_filename"], entry["start_time"])
                    translations_dict[key] = entry["text"]
            else:
                logging.warning(
                    f"Translation file not found: {translations_path}. "
                    f"Supervisions will not include translations for {split}."
                )

        # Group transcripts by audio file to create recordings
        audio_files = {}
        for entry in transcripts:
            audio_filename = entry["audio_filename"]
            if audio_filename not in audio_files:
                audio_files[audio_filename] = []
            audio_files[audio_filename].append(entry)

        recordings = []
        supervisions = []

        for audio_filename, segments in tqdm(
            audio_files.items(), desc=f"Processing {split}", leave=False
        ):
            # Find audio file (try .wav extension)
            audio_path = audio_dir / f"{audio_filename}.wav"
            if not audio_path.exists():
                # Try without extension in case filename already has it
                audio_path = audio_dir / audio_filename
                if not audio_path.exists():
                    logging.warning(
                        f"Audio file not found: {audio_filename}, skipping..."
                    )
                    continue

            # Create recording
            try:
                audio_sf = sf.SoundFile(str(audio_path))
                recording = Recording(
                    id=audio_filename,
                    sources=[
                        AudioSource(
                            type="file",
                            channels=[0],
                            source=str(audio_path),
                        )
                    ],
                    sampling_rate=audio_sf.samplerate,
                    num_samples=audio_sf.frames,
                    duration=audio_sf.frames / audio_sf.samplerate,
                )
                recordings.append(recording)
            except Exception as e:
                logging.warning(
                    f"Error reading audio file {audio_path}: {e}, skipping..."
                )
                continue

            # Create supervisions for each segment
            for entry in segments:
                start = entry["start_time"]
                end = entry["end_time"]
                text = entry["text"]

                if normalize_text:
                    text = text_cleaning(text)
                    if text.strip() == "":
                        logging.warning(
                            f"Skipping {audio_filename} {start}-{end} with empty cleaned transcript"
                        )
                        continue

                # Generate unique utterance ID
                utt_id = f"{audio_filename}_{int(1000 * start):08d}"

                # Build supervision ID based on available languages
                if include_translations and target_lang:
                    sup_id = f"{source_lang}_{target_lang}_{utt_id}"
                else:
                    sup_id = f"{source_lang}_{utt_id}"

                # Build custom dict with translations if available
                custom_dict = None
                if include_translations and target_lang:
                    key = (audio_filename, start)
                    text_tgt = translations_dict.get(key)
                    if text_tgt is not None:
                        if normalize_text:
                            text_tgt = _normalize_english(text_tgt)
                        custom_dict = {"translated_text": {target_lang: text_tgt}}
                    else:
                        logging.debug(
                            f"No translation found for {utt_id} at start={start}"
                        )

                # Use audio filename as speaker ID (one speaker per talk)
                speaker_id = audio_filename

                supervisions.append(
                    SupervisionSegment(
                        id=sup_id,
                        recording_id=audio_filename,
                        start=start,
                        duration=round(end - start, ndigits=8),
                        channel=0,
                        text=text,
                        language=LANG_TO_NAME.get(source_lang, source_lang),
                        speaker=speaker_id,
                        custom=custom_dict,
                    )
                )

        # Deduplicate and create manifest sets
        supervisions = _deduplicate_supervisions(supervisions)
        supervisions = SupervisionSet.from_segments(supervisions)
        recordings = RecordingSet.from_recordings(recordings)

        if len(recordings) > 0:
            recordings, supervisions = fix_manifests(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)

        manifests[split] = {"recordings": recordings, "supervisions": supervisions}

        logging.info(
            f"{split}: {len(recordings)} recordings, {len(supervisions)} supervisions"
        )

        # Write manifests if output_dir is specified
        if output_dir is not None:
            manifest_prefix = f"tedxtn_{'_'.join(langs)}"
            recordings.to_file(
                output_dir / f"{manifest_prefix}_recordings_{split}.jsonl.gz"
            )
            supervisions.to_file(
                output_dir / f"{manifest_prefix}_supervisions_{split}.jsonl.gz"
            )

    return manifests


def _load_csv(path: Path, is_translation: bool = False) -> List[Dict]:
    """
    Load a CSV file with TEDxTN format.

    :param path: Path to the CSV file.
    :param is_translation: If True, expect 'translation' column; else 'transcription'.
    :return: List of dicts with keys: audio_filename, start_time, end_time, text.
    """
    entries = []
    text_column = "translation" if is_translation else "transcription"

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip header rows that might be duplicated in the data
            if row.get("audio_filename") == "audio_filename":
                continue

            try:
                entry = {
                    "audio_filename": row["audio_filename"],
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "text": row.get(text_column, ""),
                }
                entries.append(entry)
            except (KeyError, ValueError) as e:
                logging.warning(f"Error parsing row in {path}: {row}, error: {e}")
                continue

    return entries


def _deduplicate_supervisions(
    supervisions: Iterable[SupervisionSegment],
) -> List[SupervisionSegment]:
    """Remove duplicate supervisions, keeping the first occurrence."""
    from cytoolz import groupby

    duplicates = groupby((lambda s: s.id), sorted(supervisions, key=lambda s: s.id))
    filtered = []
    for k, v in duplicates.items():
        if len(v) > 1:
            logging.warning(
                f"Found {len(v)} supervisions with conflicting IDs ({v[0].id}) "
                f"- keeping only the first one."
            )
        filtered.append(v[0])
    return filtered


# =============================================================================
# Text normalization functions (adapted from IWSLT22-TA)
# =============================================================================

_preNormalize = " \u0629\u0649\u0623\u0625\u0622"
_postNormalize = " \u0647\u064a\u0627\u0627\u0627"
_toNormalize = {ord(b): a for a, b in zip(_postNormalize, _preNormalize)}


def _normalize_text_(s: str) -> str:
    return s.translate(_toNormalize)


def _normalize_arabic(text: str) -> str:
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub(r"(أ){2,}", "ا", text)
    text = re.sub(r"(ا){2,}", "ا", text)
    text = re.sub(r"(آ){2,}", "ا", text)
    text = re.sub(r"(ص){2,}", "ص", text)
    text = re.sub(r"(و){2,}", "و", text)
    return text


def _remove_diacritics(text: str) -> str:
    # https://unicode-table.com/en/blocks/arabic/
    return re.sub(r"[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+", "", text)


def _remove_punctuations(text: str) -> str:
    """Remove all punctuations except verbatim markers."""
    arabic_punctuations = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
    english_punctuations = string.punctuation
    all_punctuations = set(arabic_punctuations + english_punctuations)

    for p in all_punctuations:
        if p in text:
            text = text.replace(p, " ")
    return text


def _remove_extra_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.\s+", ".", text)
    return text.strip()


def _east_to_west_num(text: str) -> str:
    eastern_to_western = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "٪": "%",
        "_": " ",
        "ڤ": "ف",
        "|": " ",
    }
    trans_string = str.maketrans(eastern_to_western)
    return text.translate(trans_string)


def text_cleaning(text: str) -> str:
    """Apply full text normalization pipeline for Arabic/Tunisian text."""
    text = _remove_punctuations(text)
    text = _east_to_west_num(text)
    text = _remove_diacritics(text)
    text = _remove_extra_space(text)
    text = _normalize_arabic(text)
    text = _normalize_text_(text)
    return text


def _normalize_english(text: str) -> str:
    """Basic normalization for English translations."""
    # Remove extra punctuation but keep basic structure
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = _remove_extra_space(text)
    return text.lower()
