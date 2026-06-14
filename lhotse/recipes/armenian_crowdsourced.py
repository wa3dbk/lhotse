"""
Armenian Crowdsourced Speech Data (OpenSLR SLR160)

This recipe supports the SLR160 dataset from OpenSLR containing crowdsourced
Armenian speech data.

Dataset URL: https://openslr.trmal.net/resources/160/armenian_speech_crowdsourcing_data.tar.gz

The dataset contains a single archive with:
- pitched.jsonl: transcript file with text, audio_filepath, and duration fields
- pitched/: directory containing WAV audio files

Usage example:
    >>> from lhotse.recipes.armenian_crowdsourced import download_armenian_crowdsourced, prepare_armenian_crowdsourced
    >>> corpus_dir = download_armenian_crowdsourced(target_dir="./data")
    >>> manifests = prepare_armenian_crowdsourced(corpus_dir=corpus_dir, output_dir="./manifests")
    >>> recordings = manifests["all"]["recordings"]
    >>> supervisions = manifests["all"]["supervisions"]
"""
import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

DOWNLOAD_URL = "https://openslr.trmal.net/resources/160/armenian_speech_crowdsourcing_data.tar.gz"

CORPUS_DIR_NAME = "armenian_speech_crowdsourcing_data"


def download_armenian_crowdsourced(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and extract the Armenian Crowdsourced Speech dataset (SLR160).

    :param target_dir: Directory to store the downloaded data.
    :param force_download: If True, re-download even if files exist.
    :return: Path to the extracted corpus directory.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = target_dir / CORPUS_DIR_NAME
    completed_detector = corpus_dir / ".completed"

    if completed_detector.is_file() and not force_download:
        logging.info(
            f"Skipping download because {completed_detector} exists."
        )
        return corpus_dir

    tar_name = "armenian_speech_crowdsourcing_data.tar.gz"
    tar_path = target_dir / tar_name

    logging.info(f"Downloading {DOWNLOAD_URL}...")
    resumable_download(DOWNLOAD_URL, filename=tar_path, force_download=force_download)

    shutil.rmtree(corpus_dir, ignore_errors=True)
    logging.info(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        safe_extract(tar, path=target_dir)

    completed_detector.touch()
    return corpus_dir


def prepare_armenian_crowdsourced(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_text: str = "none",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the Armenian Crowdsourced Speech dataset (SLR160).

    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Path to the extracted armenian_speech_crowdsourcing_data directory.
    :param output_dir: Path where the manifests should be written.
    :param normalize_text: Text normalization mode. "none" keeps original text,
        "lower" converts to lowercase.
    :return: Dict: {"all": {"recordings": ..., "supervisions": ...}}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check if manifests already exist
    if output_dir is not None and manifests_exist(
        part="all", output_dir=output_dir, prefix="armenian-crowdsourced"
    ):
        logging.info(
            "Armenian Crowdsourced manifests already exist - skipping preparation."
        )
        cached = read_manifests_if_cached(
            dataset_parts=["all"],
            output_dir=output_dir,
            prefix="armenian-crowdsourced",
        )
        if cached:
            return dict(cached)

    # Read the JSONL transcript file
    jsonl_path = corpus_dir / "pitched.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {jsonl_path}. "
            f"Expected pitched.jsonl in {corpus_dir}."
        )

    recordings = []
    supervisions = []

    logging.info("Preparing Armenian Crowdsourced Speech manifests...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing utterances"):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            text = entry["text"]
            audio_filepath = entry["audio_filepath"]
            duration = entry.get("duration")

            # Extract filename: remove /data/pitched/ prefix
            # e.g. /data/pitched/eu.xxx.wav -> eu.xxx.wav
            filename = audio_filepath.replace("/data/pitched/", "")

            # Reconstruct the actual path
            wav_path = corpus_dir / "pitched" / filename

            if not wav_path.exists():
                logging.warning(f"Audio file not found: {wav_path}, skipping.")
                continue

            utt_id = wav_path.stem  # e.g. eu.6a770d68-46a2-4288-95d5-fe7bff083609

            try:
                recording = Recording.from_file(wav_path, recording_id=utt_id)
            except Exception as e:
                logging.warning(f"Failed to read {wav_path}: {e}")
                continue

            if normalize_text == "lower":
                text = text.lower()

            supervision = SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                text=text,
                language="Armenian",
            )

            recordings.append(recording)
            supervisions.append(supervision)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    # Fix and validate manifests
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    # Save manifests
    if output_dir is not None:
        recording_set.to_file(
            output_dir / "armenian-crowdsourced_recordings_all.jsonl.gz"
        )
        supervision_set.to_file(
            output_dir / "armenian-crowdsourced_supervisions_all.jsonl.gz"
        )

    return {"all": {"recordings": recording_set, "supervisions": supervision_set}}
