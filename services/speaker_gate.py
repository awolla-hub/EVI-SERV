"""Speaker verification gate — «слушать только владельца».

A lightweight ONNX speaker-embedding model (sherpa-onnx, CAM++/ERes2Net class, ~30 MB, CPU) turns a
speech snippet into a voice fingerprint vector. Enrollment stores the OWNER's mean fingerprint; at
runtime every recognized phrase's audio is embedded and cosine-compared — phrases from OTHER voices
are dropped before they can open a turn or interrupt the assistant. This is the radical fix for
«если другие люди говорят — перебивка работает бесконечно» and it swallows most ambient noise too
(noise never matches the enrolled fingerprint).

Fail-open design: if sherpa-onnx / the model / the profile is missing, the gate is transparent and
the pipeline behaves exactly as before.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from loguru import logger

MODEL_PATH = os.environ.get(
    "SPEAKER_MODEL", "/opt/pyatnitsa/models/speaker_embedding.onnx"
)
PROFILE_PATH = os.environ.get(
    "VOICE_PROFILE", "/opt/pyatnitsa/voice_profile.json"
)


class SpeakerVerifier:
    """Embeds 16 kHz float32 mono snippets and scores them against the enrolled profile."""

    def __init__(self) -> None:
        self._extractor = None
        self._profile: Optional[np.ndarray] = None
        self._load()

    # -- setup ---------------------------------------------------------------

    def _load(self) -> None:
        try:
            import sherpa_onnx

            if not os.path.exists(MODEL_PATH):
                logger.warning("Speaker model missing at {} — gate disabled", MODEL_PATH)
                return
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=MODEL_PATH, num_threads=1
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            logger.info("Speaker verifier ready (dim={})", self._extractor.dim)
        except Exception:  # noqa: BLE001 - fail open
            logger.exception("sherpa-onnx unavailable — speaker gate disabled")
            self._extractor = None
        self._profile = self._load_profile()

    def _load_profile(self) -> Optional[np.ndarray]:
        try:
            if os.path.exists(PROFILE_PATH):
                with open(PROFILE_PATH) as f:
                    vec = np.asarray(json.load(f)["embedding"], dtype=np.float32)
                n = np.linalg.norm(vec)
                if n > 0:
                    logger.info("Voice profile loaded ({} dims)", len(vec))
                    return vec / n
        except Exception:  # noqa: BLE001
            logger.exception("Voice profile unreadable")
        return None

    # -- state ---------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._extractor is not None

    @property
    def enrolled(self) -> bool:
        return self._profile is not None and self.available

    # -- core ----------------------------------------------------------------

    def embed(self, audio16k: np.ndarray) -> Optional[np.ndarray]:
        """Fingerprint a float32 mono 16 kHz snippet (None on failure/too short)."""
        if self._extractor is None or audio16k.size < 8000:  # <0.5 s → unreliable
            return None
        try:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=16000, waveform=audio16k)
            stream.input_finished()
            vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
            n = np.linalg.norm(vec)
            return vec / n if n > 0 else None
        except Exception:  # noqa: BLE001
            logger.exception("Speaker embed failed")
            return None

    def score(self, audio16k: np.ndarray) -> Optional[float]:
        """Cosine similarity of the snippet vs the enrolled profile (None if unavailable)."""
        if self._profile is None:
            return None
        emb = self.embed(audio16k)
        if emb is None:
            return None
        return float(np.dot(emb, self._profile))

    def enroll(self, snippets: list[np.ndarray]) -> bool:
        """Store the mean fingerprint of the owner's speech snippets as the profile."""
        embs = [e for e in (self.embed(s) for s in snippets) if e is not None]
        if not embs:
            logger.warning("Enrollment failed: no usable snippets")
            return False
        mean = np.mean(np.stack(embs), axis=0)
        n = np.linalg.norm(mean)
        if n == 0:
            return False
        mean = mean / n
        try:
            # Atomic: an interrupted write used to leave a truncated JSON that _load_profile could
            # not parse, silently disabling the gate (fail-open) on the next start.
            tmp = PROFILE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"embedding": mean.tolist()}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, PROFILE_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Could not save voice profile")
            return False
        self._profile = mean
        logger.info("Voice profile ENROLLED from {} snippets", len(embs))
        return True
