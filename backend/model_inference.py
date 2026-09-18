from __future__ import annotations

import os
import hashlib
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    import torchaudio
except ModuleNotFoundError as exc:
    torch = None
    F = None
    torchaudio = None
    _dependency_error = exc
else:
    _dependency_error = None

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ModuleNotFoundError as exc:
    EncoderClassifier = None
    _speechbrain_error = exc
else:
    _speechbrain_error = None

class VoiceVerifier:
    def __init__(self, model_dir: str = "models/ecapa_librispeech_finetuned"):
        model_path = Path(model_dir)
        if not model_path.is_absolute():
            candidates = [
                Path.cwd() / model_path,
                Path(__file__).resolve().parent / model_path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_dir = str(candidate.resolve())
                    break
            else:
                model_dir = str((Path(__file__).resolve().parent / model_path).resolve())

        audio_backend_available = torchaudio is not None or sf is not None
        if torch is None or EncoderClassifier is None or not audio_backend_available:
            details = []
            if _dependency_error is not None:
                details.append(str(_dependency_error))
            if _speechbrain_error is not None:
                details.append(str(_speechbrain_error))
            if not audio_backend_available:
                details.append("Install torchaudio or soundfile for audio loading.")
            raise RuntimeError(
                "Audio inference dependencies are unavailable. Install backend requirements. "
                + ("Details: " + "; ".join(details) if details else "")
            )

        self.model_dir = Path(model_dir).resolve()

        if self.model_dir.is_file() and self.model_dir.suffix in {'.pt', '.pth'}:
            ckpt_path = self.model_dir
            self.model_dir = None
        else:
            ckpt_path = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if ckpt_path is not None:
            try:
                from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

                state = torch.load(str(ckpt_path), map_location="cpu")
            except Exception as exc:
                raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {exc}") from exc
            direct_ckpt = True
        else:
            direct_ckpt = False

        if not direct_ckpt:
            hyperparams_file = self.model_dir / "hyperparams.yaml"
            if not hyperparams_file.exists():
                try:
                    try:
                        from huggingface_hub import snapshot_download
                    except Exception:
                        snapshot_download = None

                    if snapshot_download is None:
                        self.classifier = EncoderClassifier.from_hparams(
                            source="speechbrain/spkrec-ecapa-voxceleb",
                            savedir=str(self.model_dir),
                            run_opts={"device": self.device},
                        )
                    else:
                        os.makedirs(self.model_dir, exist_ok=True)
                        snapshot_download(
                            repo_id="speechbrain/spkrec-ecapa-voxceleb",
                            local_dir=str(self.model_dir),
                            local_dir_use_symlinks=False,
                        )
                        self.classifier = EncoderClassifier.from_hparams(
                            source=str(self.model_dir),
                            savedir=str(self.model_dir),
                            run_opts={"device": self.device},
                        )

                    direct_ckpt = False
                except Exception as exc:
                    raise FileNotFoundError(
                        f"SpeechBrain model is incomplete: missing {hyperparams_file}. "
                        f"Fallback download failed: {exc}. Provide a local model directory or enable internet access."
                    )

            else:
                try:
                    self.classifier = EncoderClassifier.from_hparams(
                        source=str(self.model_dir),
                        savedir=str(self.model_dir),
                        run_opts={"device": self.device},
                    )
                except Exception:
                        ckpt_candidates = [
                            self.model_dir / "embedding_model.ckpt",
                            self.model_dir / "model.ckpt",
                            self.model_dir / "model.pt",
                            self.model_dir / "model.pth",
                            self.model_dir / "checkpoint.pt",
                            self.model_dir / "checkpoint.pth",
                        ]
                        found = None
                        for c in ckpt_candidates:
                            if c.exists():
                                found = c
                                break
                        if found is None:
                            for p in self.model_dir.glob("*.pt"):
                                found = p
                                break
                        if found is None:
                            for p in self.model_dir.glob("*.pth"):
                                found = p
                                break

                        if found is not None:
                            ckpt_path = found
                            try:
                                state = torch.load(str(ckpt_path), map_location="cpu")
                            except Exception as exc:
                                raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {exc}") from exc
                            direct_ckpt = True
                        else:
                            raise FileNotFoundError(
                                f"SpeechBrain loader failed to initialize from {self.model_dir}. "
                                f"No fallback checkpoint found. Ensure the local model directory contains 'hyperparams.yaml' and model checkpoint files, or allow hub download."
                            )

        if direct_ckpt:
            try:
                from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
                from speechbrain.lobes.features import Fbank

                if 'state_dict' in state:
                    state_dict = state['state_dict']
                elif isinstance(state, dict):
                    state_dict = state
                else:
                    raise RuntimeError("Unsupported checkpoint format")
                model = ECAPA_TDNN(
                    input_size=80,
                    channels=[1024, 1024, 1024, 1024, 3072],
                    kernel_sizes=[5, 3, 3, 3, 1],
                    dilations=[1, 2, 3, 4, 1],
                    attention_channels=128,
                    res2net_scale=8,
                    se_channels=128,
                    groups=[1, 1, 1, 1, 1],
                    dropout=0.0,
                )
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()

                self._fallback_feature_extractor = Fbank(
                    sample_rate=16000,
                    n_mels=80,
                    n_fft=400,
                    win_length=25,
                    hop_length=10,
                    f_min=0,
                    f_max=8000,
                )
                self._fallback_feature_extractor.eval()

                class _Wrapper:
                    def __init__(self, m, dev, extractor):
                        self._m = m
                        self.device = dev
                        self._extractor = extractor

                    def encode_batch(self, wav_tensor, normalize: bool = True):
                        x = wav_tensor
                        if x.dim() == 2:
                            x = x.unsqueeze(0)

                        if x.shape[1] > 1:
                            x = torch.mean(x, dim=1, keepdim=True)

                        x = x.squeeze(1)
                        with torch.no_grad():
                            feats = self._extractor(x.to(self.device))
                            out = self._m(feats.to(self.device))
                        if normalize:
                            out = F.normalize(out, dim=-1)
                        return out.cpu()

                self.classifier = _Wrapper(model, self.device, self._fallback_feature_extractor)
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize model: {exc}") from exc

    def process_audio_input(self, audio_path: str) -> "torch.Tensor":
        waveform = None
        sample_rate = None

        if torchaudio is not None:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception:
                waveform = None
                sample_rate = None

        if waveform is None and sf is not None:
            waveform, sample_rate = sf.read(audio_path, dtype='float32')
            if waveform.ndim == 1:
                waveform = torch.from_numpy(waveform).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = torch.from_numpy(waveform.T.astype('float32'))
            else:
                raise RuntimeError(f'Unsupported audio shape from soundfile: {waveform.shape}')
        elif waveform is None:
            try:
                import librosa

                y, sample_rate = librosa.load(audio_path, sr=None, mono=False)
                if isinstance(y, np.ndarray):
                    if y.ndim == 1:
                        waveform = torch.from_numpy(y.astype('float32')).unsqueeze(0)
                    else:
                        waveform = torch.from_numpy(y.astype('float32'))
                else:
                    raise RuntimeError('Unsupported audio output from librosa.load')
            except ModuleNotFoundError:
                raise RuntimeError('No audio backend available: install torchaudio, soundfile, or librosa.')

        if waveform is None or sample_rate is None:
            raise RuntimeError('Failed to load audio input.')

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000:
            if torchaudio is not None:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            else:
                try:
                    import librosa

                    resampled = librosa.resample(
                        waveform.squeeze().cpu().numpy().astype('float32'),
                        sample_rate,
                        16000,
                    )
                except Exception:
                    try:
                        from scipy.signal import resample_poly

                        resampled = resample_poly(
                            waveform.squeeze().cpu().numpy().astype('float32'),
                            16000,
                            sample_rate,
                        )
                    except ModuleNotFoundError:
                        raise RuntimeError('Unable to resample audio. Install torchaudio, librosa, or scipy.')
                waveform = torch.from_numpy(resampled.astype('float32')).unsqueeze(0)
            sample_rate = 16000

        signal = waveform.squeeze(0)

        max_samples = 16000 * 60
        if signal.shape[-1] > max_samples:
            center = signal.shape[-1] // 2
            start = max(0, center - max_samples // 2)
            signal = signal[start:start + max_samples]
        try:
            peak = torch.max(torch.abs(signal))
            if peak > 0:
                signal = signal / peak
        except Exception:
            pass

        return signal.unsqueeze(0)

    @staticmethod
    def _segment_waveform(waveform: torch.Tensor, segment_duration_seconds: float = 3.0) -> list[torch.Tensor]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        segments: list[torch.Tensor] = []
        segment_samples = int(16000 * segment_duration_seconds)
        signal = waveform.squeeze(0) if waveform.dim() > 1 else waveform

        if signal.shape[-1] <= segment_samples:
            pad_len = max(0, segment_samples - signal.shape[-1])
            chunk = signal if pad_len == 0 else F.pad(signal, (0, pad_len), mode='constant', value=0.0)
            segments.append(chunk.unsqueeze(0))
            return segments

        for start in range(0, signal.shape[-1], segment_samples):
            chunk = signal[start:start + segment_samples]
            if chunk.shape[-1] < segment_samples:
                chunk = F.pad(chunk, (0, segment_samples - chunk.shape[-1]), mode='constant', value=0.0)
            segments.append(chunk.unsqueeze(0))

        return segments

    def extract_embedding(self, audio_path: str):
        waveform = self.process_audio_input(audio_path)

        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)

        segments = self._segment_waveform(waveform)
        chunk_embeddings = []

        try:
            for segment in segments:
                with torch.no_grad():
                    embedding = self.classifier.encode_batch(segment.to(self.device), normalize=True)
                chunk_embeddings.append(embedding.squeeze())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to encode audio '{audio_path}': {exc}"
            ) from exc

        if not chunk_embeddings:
            return torch.zeros(256, device=self.device)

        return torch.stack(chunk_embeddings).mean(dim=0)

    def _extract_notebook_embedding(self, audio_path: str):
        """Replicate the notebook's 3-second direct embedding pipeline."""
        waveform = self.process_audio_input(audio_path).squeeze(0)
        target_samples = 16000 * 3
        if waveform.shape[-1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
        else:
            waveform = waveform[:target_samples]
        waveform = waveform.unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.classifier, "mods"):
                features = self.classifier.mods.compute_features(waveform)
                embedding = self.classifier.mods.embedding_model(features).squeeze()
            else:
                embedding = self.classifier.encode_batch(waveform, normalize=True).squeeze()

        return embedding

    def _extract_window_embeddings(self, audio_path: str):
        """Extract notebook-style embeddings from several speech windows."""
        waveform = self.process_audio_input(audio_path).squeeze(0)
        window_samples = 16000 * 3
        if waveform.shape[-1] <= window_samples:
            windows = [F.pad(waveform, (0, window_samples - waveform.shape[-1]))]
        else:
            step = 16000
            starts = list(range(0, waveform.shape[-1] - window_samples + 1, step))
            windows = [waveform[start:start + window_samples] for start in starts]
            if not windows:
                windows = [waveform[:window_samples]]

        embeddings = []
        for window in windows:
            batch = window.unsqueeze(0).to(self.device)
            with torch.no_grad():
                if hasattr(self.classifier, "mods"):
                    features = self.classifier.mods.compute_features(batch)
                    embedding = self.classifier.mods.embedding_model(features).squeeze()
                else:
                    embedding = self.classifier.encode_batch(batch, normalize=True).squeeze()
            embeddings.append(embedding)
        return embeddings

    def _segment_embeddings(self, audio_path: str):
        waveform = self.process_audio_input(audio_path)
        segments = self._segment_waveform(waveform, segment_duration_seconds=3.0)
        embeddings = []

        for segment in segments:
            with torch.no_grad():
                embedding = self.classifier.encode_batch(segment.to(self.device), normalize=True)
            embeddings.append(embedding.squeeze())

        return embeddings

    @staticmethod
    def _spectral_flatness(signal: torch.Tensor) -> float:
        if signal.numel() == 0:
            return 0.0

        values = signal.detach().cpu().numpy().astype('float32')
        if values.size == 0:
            return 0.0

        spectrum = np.abs(np.fft.rfft(values))
        spectrum = spectrum[1:]
        if spectrum.size == 0:
            return 0.0
        spectrum = np.clip(spectrum, 1e-8, None)
        geometric_mean = float(np.exp(np.mean(np.log(spectrum))))
        arithmetic_mean = float(np.mean(spectrum))
        if arithmetic_mean <= 0:
            return 0.0
        return float(geometric_mean / arithmetic_mean)

    @staticmethod
    def _dominant_spectral_ratio(signal: np.ndarray) -> float:
        if signal.size == 0:
            return 0.0

        spectrum = np.abs(np.fft.rfft(signal.astype('float32')))
        spectrum = spectrum[1:]
        if spectrum.size == 0:
            return 0.0

        total = float(np.sum(spectrum))
        if total <= 0:
            return 0.0

        top_bins = int(max(4, min(32, spectrum.size // 10)))
        dominant = np.sort(spectrum)[-top_bins:]
        return float(np.sum(dominant) / total)

    def _speech_quality_score(self, audio_path: str) -> float:
        waveform = self.process_audio_input(audio_path)
        signal = waveform.squeeze(0).detach().cpu().numpy().astype('float32')

        if signal.size == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(np.square(signal))))
        flatness = self._spectral_flatness(waveform.squeeze(0))
        dominant_ratio = self._dominant_spectral_ratio(signal)
        duration_seconds = max(signal.size / 16000.0, 0.1)

        duration_score = min(1.0, duration_seconds / 3.0)
        energy_score = min(1.0, rms * 12.0)
        flatness_score = 1.0 - min(1.0, flatness / 0.60)
        spectral_score = 1.0 - min(1.0, dominant_ratio / 0.75)

        quality = 0.30 * energy_score + 0.25 * flatness_score + 0.20 * spectral_score + 0.25 * duration_score
        return max(0.0, min(1.0, quality))

    def compare_voices(self, audio_path1: str, audio_path2: str, threshold: float = 0.60):
        try:
            def file_digest(audio_path: str) -> str:
                digest = hashlib.sha256()
                with open(audio_path, "rb") as audio_file:
                    for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            if file_digest(audio_path1) == file_digest(audio_path2):
                return {
                    "score": 1.0,
                    "confidence": 100,
                    "isMatch": True,
                    "verdict": "MATCH CONFIRMED",
                    "explanation": "MATCH CONFIRMED: the two uploaded audio files are identical.",
                    "requestedThreshold": round(float(threshold), 2) if threshold is not None else 0.70,
                    "effectiveThreshold": round(float(threshold), 2) if threshold is not None else 0.70,
                    "rawSimilarity": 1.0,
                    "segmentSimilarity": 1.0,
                }
        except (OSError, TypeError, ValueError):
            pass

        window_embeddings1 = self._extract_window_embeddings(audio_path1)
        window_embeddings2 = self._extract_window_embeddings(audio_path2)
        
        if not window_embeddings1 or not window_embeddings2:
            similarity = 0.0
            segment_similarity = 0.0
        else:
            best_similarity = -1.0
            for emb1 in window_embeddings1:
                for emb2 in window_embeddings2:
                    sim = float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())
                    if sim > best_similarity:
                        best_similarity = sim
            
            similarity = max(0.0, best_similarity)
            
            short_windows, long_windows = (
                (window_embeddings1, window_embeddings2)
                if len(window_embeddings1) <= len(window_embeddings2)
                else (window_embeddings2, window_embeddings1)
            )
            nearest_scores = []
            for first in short_windows:
                scores = [
                    float(F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item())
                    for second in long_windows
                ]
                nearest_scores.append(max(scores))
            segment_similarity = float(np.mean(nearest_scores)) if nearest_scores else 0.0

        raw_similarity = float(similarity)

        if raw_similarity >= 0.9995:
            raw_similarity = 1.0

        duration1 = 0.0
        duration2 = 0.0
        quality1 = 0.0
        quality2 = 0.0
        blended_similarity = 0.70 * raw_similarity + 0.30 * segment_similarity

        similarity_score = round(blended_similarity, 2)

        DEFAULT_THRESHOLD = 0.75
        configured_threshold = float(threshold) if threshold is not None else DEFAULT_THRESHOLD

        mic_source = (
            "mic_recorded_" in os.path.basename(audio_path1).lower()
            or "mic_recorded_" in os.path.basename(audio_path2).lower()
        )
        effective_threshold = 0.45 if mic_source else max(float(configured_threshold), 0.85)
        min_raw_similarity = 0.43 if mic_source else max(0.78, effective_threshold)
        min_segment_similarity = 0.40 if mic_source else max(0.80, effective_threshold)
        standard_match = (
            blended_similarity >= effective_threshold
            and raw_similarity >= min_raw_similarity
            and segment_similarity >= min_segment_similarity
        )
        cross_language_match = (
            not mic_source
            and blended_similarity >= 0.75
            and raw_similarity >= 0.78
            and segment_similarity >= 0.72
        )
        is_match = standard_match or cross_language_match
        decision_threshold = 0.77 if cross_language_match and not standard_match else effective_threshold

        verdict = "MATCH CONFIRMED" if is_match else "NO MATCH DETECTED"

        if is_match:
            if cross_language_match and not standard_match:
                explanation = f"{verdict}: strong cross-language speaker similarity={similarity_score:+.2f}."
            else:
                explanation = f"{verdict}: similarity={similarity_score:+.2f} (threshold={effective_threshold:.2f})."
        elif blended_similarity <= decision_threshold:
            explanation = f"{verdict}: similarity={similarity_score:+.2f} is below threshold={effective_threshold:.2f}."
        else:
            explanation = (
                f"{verdict}: raw or segment consistency is insufficient "
                f"(raw={raw_similarity:.2f}, segments={segment_similarity:.2f})."
            )

        try:
            if blended_similarity <= decision_threshold:
                low_min = -1.0
                low_max = decision_threshold
                if low_max - low_min == 0:
                    confidence_pct = 0
                else:
                    confidence_pct = int(round(max(0.0, min(1.0, (blended_similarity - low_min) / (low_max - low_min))) * 75))
            else:
                hi_min = decision_threshold
                hi_max = 1.0
                confidence_pct = 75 + int(round(max(0.0, min(1.0, (blended_similarity - hi_min) / (hi_max - hi_min))) * 25))
            confidence_pct = max(0, min(100, int(confidence_pct)))
        except Exception:
            confidence_pct = int(round(similarity_score))

        return {
            "score": similarity_score,
            "confidence": confidence_pct,
            "isMatch": is_match,
            "verdict": verdict,
            "explanation": explanation,
            "requestedThreshold": round(configured_threshold, 2),
            "effectiveThreshold": round(decision_threshold, 2),
            "rawSimilarity": round(raw_similarity, 3),
            "segmentSimilarity": round(segment_similarity, 3),
            "durationSeconds": [round(duration1, 2), round(duration2, 2)],
            "qualityScores": [round(quality1, 3), round(quality2, 3)],
        }