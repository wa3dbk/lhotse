"""
ALFFA (African Languages in the Field: speech Fundamentals and Automation)

This recipe supports the SLR25 dataset from OpenSLR containing speech data for:
- Amharic (data_readspeech_am): ~20 hours of read speech
- Swahili (data_broadcastnews_sw): ~12 hours of broadcast news
- Wolof (data_readspeech_wo): ~16 hours of read speech

Dataset URL: https://www.openslr.org/25/

The data was collected as part of the ALFFA project (http://alffa.imag.fr/).

Usage example:
    >>> from lhotse.recipes.alffa import download_alffa, prepare_alffa
    >>> # Download all languages
    >>> corpus_dirs = download_alffa(target_dir="./data", languages="all")
    >>> # Prepare manifests
    >>> manifests = prepare_alffa(corpus_dir="./data", output_dir="./manifests")
    >>> # Access specific language/split
    >>> amharic_train = manifests["amharic"]["train"]
    >>> recordings = amharic_train["recordings"]
    >>> supervisions = amharic_train["supervisions"]

References:
    @article{gauthier2016collect,
        Author = {Gauthier, Elodie and Besacier, Laurent and Voisin, Sylvie and Melese, Michael and Elingui, Uriel Pascal},
        Journal = {LREC},
        Title = {Collecting Resources in Sub-Saharan African Languages for Automatic Speech Recognition: a Case Study of Wolof},
        Year = {2016}
    }

    @InProceedings{Abate2005,
        Author = {Solomon Teferra Abate and Wolfgang Menzel and Bairu Tafila},
        booktitle = {INTERSPEECH-2005},
        Title = {An Amharic Speech Corpus for Large Vocabulary Continuous Speech Recognition},
        Year = {2005}
    }

    @InProceedings{gelas:hal-00954048,
        author = {Gelas, Hadrien and Besacier, Laurent and Pellegrino, Francois},
        title = {{D}evelopments of {S}wahili resources for an automatic speech recognition system},
        booktitle = {{SLTU} - {W}orkshop on {S}poken {L}anguage {T}echnologies for {U}nder-{R}esourced {L}anguages},
        year = {2012},
        address = {Cape-Town, Afrique Du Sud},
    }
"""
import logging
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

# Language configurations
ALFFA_LANGUAGES = ("amharic", "swahili", "wolof")

# Language to directory name mapping
LANG_TO_DIR = {
    "amharic": "data_readspeech_am",
    "swahili": "data_broadcastnews_sw",
    "wolof": "data_readspeech_wo",
}

# Language to ISO 639-1/639-3 code mapping
LANG_TO_CODE = {
    "amharic": "am",
    "swahili": "sw",
    "wolof": "wo",
}

# Language to full language name (for SupervisionSegment)
LANG_TO_NAME = {
    "amharic": "Amharic",
    "swahili": "Swahili",
    "wolof": "Wolof",
}

# Available splits per language
LANG_SPLITS = {
    "amharic": ("train", "test"),
    "swahili": ("train", "test"),
    "wolof": ("train", "dev", "test"),
}

# Download URLs
BASE_URL = "https://openslr.trmal.net/resources/25"
DOWNLOAD_URLS = {
    "amharic": f"{BASE_URL}/data_readspeech_am.tar.bz2",
    "swahili": f"{BASE_URL}/data_broadcastnews_sw.tar.bz2",
    "wolof": f"{BASE_URL}/data_readspeech_wo.tar.bz2",
}


def download_alffa(
    target_dir: Pathlike = ".",
    languages: Union[str, Sequence[str]] = "all",
    force_download: bool = False,
    base_url: str = BASE_URL,
) -> Dict[str, Path]:
    """
    Download and extract the ALFFA dataset.

    :param target_dir: Directory to store the downloaded data.
    :param languages: Which languages to download. Can be "all", a single language name,
        or a sequence of language names. Valid languages: "amharic", "swahili", "wolof".
    :param force_download: If True, re-download even if files exist.
    :param base_url: Base URL for downloading (can be changed to use mirrors).
    :return: Dict mapping language names to their extracted corpus directories.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if languages == "all":
        languages = list(ALFFA_LANGUAGES)
    elif isinstance(languages, str):
        languages = [languages]

    corpus_dirs = {}

    for lang in tqdm(languages, desc="Downloading ALFFA languages"):
        lang = lang.lower()
        if lang not in ALFFA_LANGUAGES:
            logging.warning(
                f"Unknown language: {lang}. Valid options: {ALFFA_LANGUAGES}"
            )
            continue

        logging.info(f"Processing ALFFA {lang.capitalize()} dataset...")

        dir_name = LANG_TO_DIR[lang]
        corpus_dir = target_dir / dir_name
        completed_detector = corpus_dir / ".completed"

        if completed_detector.is_file() and not force_download:
            logging.info(f"Skipping {lang} because {completed_detector} exists.")
            corpus_dirs[lang] = corpus_dir
            continue

        # Construct download URL
        tar_name = f"{dir_name}.tar.bz2"
        tar_path = target_dir / tar_name
        url = f"{base_url}/{tar_name}"

        # Download
        logging.info(f"Downloading {url}...")
        resumable_download(url, filename=tar_path, force_download=force_download)

        # Extract
        shutil.rmtree(corpus_dir, ignore_errors=True)
        logging.info(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            safe_extract(tar, path=target_dir)

        completed_detector.touch()
        corpus_dirs[lang] = corpus_dir

    return corpus_dirs


def prepare_alffa(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    languages: Union[str, Sequence[str]] = "all",
    dataset_parts: Union[str, Sequence[str]] = "all",
    normalize_text: str = "none",
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Prepare manifests for the ALFFA dataset.

    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Path to the directory containing the extracted ALFFA data.
        This should be the parent directory containing data_readspeech_am,
        data_broadcastnews_sw, and/or data_readspeech_wo subdirectories.
    :param output_dir: Path where the manifests should be written.
    :param languages: Which languages to prepare. Can be "all", a single language,
        or a sequence of languages. Valid: "amharic", "swahili", "wolof".
    :param dataset_parts: Which splits to prepare. Can be "all", a single split name,
        or a sequence of split names. Valid: "train", "dev", "test".
        Note: "dev" is only available for Wolof.
    :param normalize_text: Text normalization mode. "none" keeps original text,
        "lower" converts to lowercase.
    :return: Nested dict: {language: {split: {"recordings": ..., "supervisions": ...}}}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Parse languages
    if languages == "all":
        languages = list(ALFFA_LANGUAGES)
    elif isinstance(languages, str):
        languages = [languages]
    languages = [lang.lower() for lang in languages]

    # Parse dataset parts
    if dataset_parts == "all":
        all_parts = set()
        for lang in languages:
            all_parts.update(LANG_SPLITS.get(lang, ()))
        dataset_parts = list(all_parts)
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    manifests = defaultdict(dict)

    for lang in tqdm(languages, desc="Processing ALFFA languages"):
        if lang not in ALFFA_LANGUAGES:
            logging.warning(f"Unknown language: {lang}. Skipping.")
            continue

        lang_dir = corpus_dir / LANG_TO_DIR[lang]
        if not lang_dir.is_dir():
            # Try if corpus_dir is the language directory itself
            if corpus_dir.name == LANG_TO_DIR[lang]:
                lang_dir = corpus_dir
            else:
                logging.warning(
                    f"Directory not found for {lang}: {lang_dir}. Skipping."
                )
                continue

        logging.info(f"Preparing ALFFA {lang.capitalize()} manifests...")

        available_splits = LANG_SPLITS[lang]
        for split in dataset_parts:
            if split not in available_splits:
                logging.debug(f"Split '{split}' not available for {lang}. Skipping.")
                continue

            # Check if manifests already exist
            if output_dir is not None and manifests_exist(
                part=f"{lang}_{split}", output_dir=output_dir, prefix="alffa"
            ):
                logging.info(
                    f"ALFFA {lang} {split} manifests already exist - skipping preparation."
                )
                # Read cached manifests
                cached = read_manifests_if_cached(
                    dataset_parts=[f"{lang}_{split}"],
                    output_dir=output_dir,
                    prefix="alffa",
                )
                if cached:
                    manifests[lang][split] = cached[f"{lang}_{split}"]
                continue

            logging.info(f"Processing {lang} {split} split...")

            # Prepare manifests based on language-specific format
            if lang == "amharic":
                recordings, supervisions = _prepare_amharic(
                    lang_dir, split, normalize_text
                )
            elif lang == "swahili":
                recordings, supervisions = _prepare_swahili(
                    lang_dir, split, normalize_text
                )
            elif lang == "wolof":
                recordings, supervisions = _prepare_wolof(
                    lang_dir, split, normalize_text
                )
            else:
                continue

            if len(recordings) == 0:
                logging.warning(f"No recordings found for {lang} {split}. Skipping.")
                continue

            # Fix and validate manifests
            recordings, supervisions = fix_manifests(recordings, supervisions)
            validate_recordings_and_supervisions(recordings, supervisions)

            # Save manifests
            if output_dir is not None:
                recordings.to_file(
                    output_dir / f"alffa_recordings_{lang}_{split}.jsonl.gz"
                )
                supervisions.to_file(
                    output_dir / f"alffa_supervisions_{lang}_{split}.jsonl.gz"
                )

            manifests[lang][split] = {
                "recordings": recordings,
                "supervisions": supervisions,
            }

    return dict(manifests)


def _prepare_amharic(
    lang_dir: Path, split: str, normalize_text: str
) -> Tuple[RecordingSet, SupervisionSet]:
    """Prepare Amharic data from Kaldi-style files."""
    split_dir = lang_dir / "data" / split

    recordings = []
    supervisions = []

    # Read text file for transcriptions
    text_path = split_dir / "text"
    if not text_path.exists():
        logging.warning(f"Text file not found: {text_path}")
        return RecordingSet.from_recordings([]), SupervisionSet.from_segments([])

    texts = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                utt_id, text = parts
                texts[utt_id] = text
            elif len(parts) == 1:
                texts[parts[0]] = ""

    # Read utt2spk for speaker mapping
    utt2spk = {}
    utt2spk_path = split_dir / "utt2spk"
    if utt2spk_path.exists():
        with open(utt2spk_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt2spk[parts[0]] = parts[1]

    # Find wav files
    wav_dir = split_dir / "wav"
    if wav_dir.is_dir():
        for wav_path in sorted(wav_dir.glob("*.wav")):
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

            speaker = utt2spk.get(utt_id, utt_id)

            supervision = SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                text=text,
                language=LANG_TO_NAME["amharic"],
                speaker=speaker,
            )

            recordings.append(recording)
            supervisions.append(supervision)

    return RecordingSet.from_recordings(recordings), SupervisionSet.from_segments(
        supervisions
    )


def _prepare_swahili(
    lang_dir: Path, split: str, normalize_text: str
) -> Tuple[RecordingSet, SupervisionSet]:
    """Prepare Swahili data from Kaldi-style files."""
    split_dir = lang_dir / "data" / split

    recordings = []
    supervisions = []

    # Read text file for transcriptions
    text_path = split_dir / "text"
    if not text_path.exists():
        logging.warning(f"Text file not found: {text_path}")
        return RecordingSet.from_recordings([]), SupervisionSet.from_segments([])

    texts = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                utt_id, text = parts
                texts[utt_id] = text
            elif len(parts) == 1:
                texts[parts[0]] = ""

    # Read utt2spk for speaker mapping
    utt2spk = {}
    utt2spk_path = split_dir / "utt2spk"
    if utt2spk_path.exists():
        with open(utt2spk_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt2spk[parts[0]] = parts[1]

    # Find wav files - Swahili has speaker subdirectories
    # Structure: wav/SPEAKER_ID/*.wav or wav5/SPEAKER_ID/*.wav
    wav_dirs = []
    for wav_dir_name in ["wav", "wav5"]:
        wav_dir = split_dir / wav_dir_name
        if wav_dir.is_dir():
            wav_dirs.append(wav_dir)

    # Also check if wavs are directly in split_dir/wav/SPEAKER_ID/
    for wav_dir in wav_dirs:
        # Check for speaker subdirectories
        for speaker_dir in wav_dir.iterdir():
            if speaker_dir.is_dir() and not speaker_dir.name.startswith("."):
                for wav_path in sorted(speaker_dir.glob("*.wav")):
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

                    speaker = utt2spk.get(utt_id, speaker_dir.name)

                    supervision = SupervisionSegment(
                        id=utt_id,
                        recording_id=utt_id,
                        start=0.0,
                        duration=recording.duration,
                        channel=0,
                        text=text,
                        language=LANG_TO_NAME["swahili"],
                        speaker=speaker,
                    )

                    recordings.append(recording)
                    supervisions.append(supervision)

        # Also check for wav files directly in wav_dir
        for wav_path in sorted(wav_dir.glob("*.wav")):
            utt_id = wav_path.stem

            if utt_id not in texts:
                continue

            try:
                recording = Recording.from_file(wav_path, recording_id=utt_id)
            except Exception as e:
                logging.warning(f"Failed to read {wav_path}: {e}")
                continue

            text = texts[utt_id]
            if normalize_text == "lower":
                text = text.lower()

            speaker = utt2spk.get(utt_id, utt_id)

            supervision = SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                text=text,
                language=LANG_TO_NAME["swahili"],
                speaker=speaker,
            )

            recordings.append(recording)
            supervisions.append(supervision)

    return RecordingSet.from_recordings(recordings), SupervisionSet.from_segments(
        supervisions
    )


def _prepare_wolof(
    lang_dir: Path, split: str, normalize_text: str
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Prepare Wolof data.

    Wolof has a different structure - text files contain utterance IDs like:
    WOL_03_lect_0001  seen suuf dafa nangu lu ñu fa ji mu sax

    The audio files should be in wav/ directory with matching names.
    """
    split_dir = lang_dir / "data" / split

    recordings = []
    supervisions = []

    # Read text file for transcriptions
    text_path = split_dir / "text"
    if not text_path.exists():
        logging.warning(f"Text file not found: {text_path}")
        return RecordingSet.from_recordings([]), SupervisionSet.from_segments([])

    texts = {}
    utt_to_wav = {} 
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                utt_id, text = parts
                texts[utt_id] = text

                # Build a mapping of utterance IDs to wav paths
                subdir = utt_id.split("_")[1] # extract sub-directory WOL_03_lect_0001 -> 03
                wavfile = split_dir / subdir / f"{utt_id}.wav"

                if not Path(wavfile).is_file():
                    logging.debug(f"No audio file found for {utt_id}, skipping (wavfile={wavfile}).")
                else:
                    utt_to_wav[utt_id] = wavfile

            elif len(parts) == 1:
                texts[parts[0]] = ""



    # Extract speaker ID from utterance ID (e.g., WOL_03_lect_0001 -> WOL_03)
    def extract_speaker(utt_id: str) -> str:
        parts = utt_id.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return utt_id

    for utt_id, text in texts.items():
        wav_path = utt_to_wav.get(utt_id)
        if wav_path is None:
            logging.debug(f"No audio file found for {utt_id}, skipping.")
            continue

        try:
            recording = Recording.from_file(wav_path, recording_id=utt_id)
        except Exception as e:
            logging.warning(f"Failed to read {wav_path}: {e}")
            continue

        if normalize_text == "lower":
            text = text.lower()

        speaker = extract_speaker(utt_id)

        supervision = SupervisionSegment(
            id=utt_id,
            recording_id=utt_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            text=text,
            language=LANG_TO_NAME["wolof"],
            speaker=speaker,
        )

        recordings.append(recording)
        supervisions.append(supervision)

    return RecordingSet.from_recordings(recordings), SupervisionSet.from_segments(
        supervisions
    )

