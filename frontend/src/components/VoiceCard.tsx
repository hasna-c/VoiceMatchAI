import React, { useEffect, useRef, useState } from 'react';
import { FileAudio, Mic, Upload, CheckCircle2, Trash2, Play, Pause } from 'lucide-react';
import { VoiceSample } from '../types';

interface VoiceCardProps {
  label: string;
  labelColor?: string;
  sample: VoiceSample | null;
  onUploadSample: (file: File) => void;
  onSaveRecord: (sample: VoiceSample) => void;
  onRemoveSample: () => void;
}

export const VoiceCard: React.FC<VoiceCardProps> = ({
  label,
  labelColor = 'text-[#00f5ff]',
  sample,
  onUploadSample,
  onSaveRecord,
  onRemoveSample,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const saveOnStopRef = useRef(true);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const formatTime = (timeInSec: number) => {
    const mins = Math.floor(timeInSec / 60);
    const secs = Math.floor(timeInSec % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const cleanupRecording = () => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setIsPlaying(false);
      setCurrentTime(0);
      cleanupRecording();
    };
  }, [sample]);

  // Load audio metadata when a sample is set so total duration displays immediately
  useEffect(() => {
    setTotalDuration(0);
    if (!sample?.fileUrl) return;
    // If we have a mounted audio element, set its src so it will fire loadedmetadata
    if (audioRef.current) {
      audioRef.current.src = sample.fileUrl;
    }
    // No cleanup necessary here; the audio element lifecycle is tied to component
  }, [sample?.fileUrl]);

  const prepareUploadedFile = async (file: File) => {
    try {
      const wavBlob = await convertBlobToWav(file);
      return new File([wavBlob], `${file.name.replace(/\.[^/.]+$/, '')}.wav`, { type: 'audio/wav' });
    } catch (error) {
      console.error('Uploaded audio conversion failed:', error);
      return file;
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    onUploadSample(await prepareUploadedFile(file));
    e.target.value = '';
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) onUploadSample(await prepareUploadedFile(file));
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const togglePlay = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (!sample?.fileUrl) return;
    // Use the mounted audio element for reliable metadata and playback updates
    // If the hidden audio element isn't currently attached, create one so
    // playback works even when React's refs were cleared by lifecycle timing
    // or when the element isn't reliably present in the DOM.
    if (!audioRef.current) {
      const a = new Audio(sample.fileUrl);
      a.style.display = 'none';
      a.addEventListener('timeupdate', () => {
        setCurrentTime(a.currentTime);
      });
      a.addEventListener('ended', () => {
        setIsPlaying(false);
        setCurrentTime(0);
      });
      a.addEventListener('loadedmetadata', () => {
        setTotalDuration(a.duration || 0);
      });
      audioRef.current = a;
    }

    // Ensure src is set to current sample
    if (audioRef.current.src !== sample.fileUrl) audioRef.current.src = sample.fileUrl;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch((err) => {
        console.error('Playback failed:', err);
        // Provide a lightweight user-visible hint
        alert('Playback failed. Check that your browser allows audio playback for this page.');
      });
    }
  };

  const mergeChannels = (audioBuffer: AudioBuffer) => {
    const channelCount = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const output = new Float32Array(length);

    if (channelCount === 1) {
      output.set(audioBuffer.getChannelData(0));
      return output;
    }

    for (let channel = 0; channel < channelCount; channel += 1) {
      const channelData = audioBuffer.getChannelData(channel);
      for (let i = 0; i < length; i += 1) {
        output[i] += channelData[i] / channelCount;
      }
    }

    return output;
  };

  const normalizeFloat32 = (samples: Float32Array) => {
    let peak = 0;
    for (let i = 0; i < samples.length; i += 1) {
      peak = Math.max(peak, Math.abs(samples[i]));
    }
    if (peak <= 0) return samples;
    const scale = 0.99 / peak;
    const normalized = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      normalized[i] = samples[i] * scale;
    }
    return normalized;
  };

  const removeDcOffset = (samples: Float32Array) => {
    const mean = samples.reduce((acc, value) => acc + value, 0) / Math.max(1, samples.length);
    const filtered = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      filtered[i] = samples[i] - mean;
    }
    return filtered;
  };

  const applySoftPreemphasis = (samples: Float32Array, coeff = 0.97) => {
    const filtered = new Float32Array(samples.length);
    for (let i = 1; i < samples.length; i += 1) {
      filtered[i] = samples[i] - coeff * samples[i - 1];
    }
    filtered[0] = samples[0];
    return filtered;
  };

  const trimSilence = (samples: Float32Array, silenceThreshold = 0.02) => {
    let start = 0;
    let end = samples.length;

    while (start < samples.length && Math.abs(samples[start]) < silenceThreshold) {
      start += 1;
    }

    while (end > start && Math.abs(samples[end - 1]) < silenceThreshold) {
      end -= 1;
    }

    if (end <= start) return new Float32Array(0);
    return samples.slice(start, end);
  };

  const isUsableVoiceRecording = (samples: Float32Array) => {
    if (!samples.length) return false;
    const trimmed = trimSilence(samples);
    if (trimmed.length < 16000 * 1.0) return false;

    let energy = 0;
    let clipped = 0;
    let peak = 0;
    for (let i = 0; i < trimmed.length; i += 1) {
      const value = Math.abs(trimmed[i]);
      energy += value * value;
      peak = Math.max(peak, value);
      if (value > 0.95) clipped += 1;
    }

    const rms = Math.sqrt(energy / trimmed.length);
    const clippingRatio = clipped / trimmed.length;
    return rms > 0.012 && peak > 0.04 && clippingRatio < 0.05;
  };

  const resampleTo16kHz = (samples: Float32Array, originalSampleRate: number) => {
    if (originalSampleRate === 16000 || !Number.isFinite(originalSampleRate) || originalSampleRate <= 0) {
      return samples;
    }

    const targetSampleRate = 16000;
    const seconds = samples.length / originalSampleRate;
    const targetLength = Math.max(1, Math.ceil(seconds * targetSampleRate));
    const resampled = new Float32Array(targetLength);

    for (let i = 0; i < targetLength; i += 1) {
      const time = i / targetSampleRate;
      const sourceIndex = time * originalSampleRate;
      const index = Math.floor(sourceIndex);
      const nextIndex = Math.min(samples.length - 1, index + 1);
      const fraction = sourceIndex - index;
      const current = samples[index] ?? 0;
      const next = samples[nextIndex] ?? current;
      resampled[i] = current + (next - current) * fraction;
    }

    return normalizeFloat32(resampled);
  };

  const floatTo16BitPCM = (output: Float32Array) => {
    const buffer = new ArrayBuffer(output.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < output.length; i += 1) {
      let s = Math.max(-1, Math.min(1, output[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return view;
  };

  const encodeWAV = (samples: Float32Array, sampleRate: number) => {
    const input = floatTo16BitPCM(samples);
    const buffer = new ArrayBuffer(44 + input.byteLength);
    const view = new DataView(buffer);

    const writeString = (str: string, offset: number) => {
      for (let i = 0; i < str.length; i += 1) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString('RIFF', 0);
    view.setUint32(4, 36 + input.byteLength, true);
    writeString('WAVE', 8);
    writeString('fmt ', 12);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString('data', 36);
    view.setUint32(40, input.byteLength, true);
    const wavBytes = new Uint8Array(buffer);
    wavBytes.set(new Uint8Array(input.buffer), 44);
    return new Blob([wavBytes], { type: 'audio/wav' });
  };

  const convertBlobToWav = async (blob: Blob) => {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new AudioContext();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    const monoSamples = normalizeFloat32(mergeChannels(decoded));
    const offlineCtx = new OfflineAudioContext(1, Math.ceil(monoSamples.length * (16000 / decoded.sampleRate)), 16000);
    const buffer = offlineCtx.createBuffer(1, monoSamples.length, decoded.sampleRate);
    buffer.copyToChannel(monoSamples, 0);
    const source = offlineCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(offlineCtx.destination);
    source.start();
    const rendered = await offlineCtx.startRendering();
    const renderedSamples = rendered.getChannelData(0);
    const normalized = normalizeFloat32(renderedSamples);
    return encodeWAV(normalized, 16000);
  };

  const saveRecording = async (blob: Blob) => {
    const wavBlob = await convertBlobToWav(blob);
    const url = URL.createObjectURL(wavBlob);
    // Create audio element to read accurate duration metadata from the blob
    const a = new Audio(url);
    const finalize = (durSeconds: number) => {
      const newSample: VoiceSample = {
        id: `mic-${Date.now()}`,
        name: `mic_recorded_${label.replace(/\s+/g, '_').toLowerCase()}.wav`,
        duration: Math.max(1, Math.round(durSeconds || recordingTime || 0)),
        size: `${(wavBlob.size / 1024 / 1024).toFixed(1)} MB`,
        fileUrl: url,
      };
      onSaveRecord(newSample);
      a.removeEventListener('loadedmetadata', onLoaded);
      clearTimeout(fb);
    };

    const onLoaded = () => finalize(a.duration || recordingTime || 0);
    a.addEventListener('loadedmetadata', onLoaded);

    // Fallback: if metadata doesn't arrive, finalize after 2s
    const fb = window.setTimeout(() => finalize(recordingTime || 1), 2000);
  };

  const recordWithAudioContext = async (stream: MediaStream, onStop: (wavBlob: Blob, duration: number) => void) => {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const audioContext = new AudioContextClass();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const audioData: Float32Array[] = [];
    let totalSamples = 0;

    processor.onaudioprocess = (event) => {
      const samples = event.inputBuffer.getChannelData(0);
      audioData.push(new Float32Array(samples));
      totalSamples += samples.length;
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    return {
      stop: () => {
        source.disconnect();
        processor.disconnect();

        const combinedAudio = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of audioData) {
          combinedAudio.set(chunk, offset);
          offset += chunk.length;
        }

        const sampleRate = audioContext.sampleRate;
        const resampled = resampleTo16kHz(combinedAudio, sampleRate);
        const dcRemoved = removeDcOffset(resampled);
        const cleaned = trimSilence(dcRemoved, 0.01);

        if (!isUsableVoiceRecording(cleaned)) {
          audioContext.close();
          onStop(new Blob([createWavHeader(16000, 1), floatTo16BitPCM(new Float32Array([0]))], { type: 'audio/wav' }), 0.1);
          return;
        }

        const pcm16bit = floatTo16BitPCM(cleaned);
        const wavBlob = new Blob(
          [createWavHeader(16000, cleaned.length), pcm16bit],
          { type: 'audio/wav' }
        );
        const duration = cleaned.length / 16000;

        audioContext.close();
        onStop(wavBlob, duration);
      },
    };
  };

  const createWavHeader = (sampleRate: number, numSamples: number) => {
    const buffer = new ArrayBuffer(44);
    const view = new DataView(buffer);
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample;

    const writeString = (str: string, offset: number) => {
      for (let i = 0; i < str.length; i += 1) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    // RIFF header
    writeString('RIFF', 0);
    view.setUint32(4, 36 + numSamples * bytesPerSample, true);
    writeString('WAVE', 8);
    writeString('fmt ', 12);
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true); // avg bytes per sec
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // bits per sample
    writeString('data', 36);
    view.setUint32(40, numSamples * bytesPerSample, true);

    return new Uint8Array(buffer);
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Microphone access is not supported by your browser. Open this app in a modern browser like Chrome, Edge, or Firefox on localhost or HTTPS.');
      return;
    }

    const isSecure = window.location.protocol === 'https:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    if (!isSecure) {
      alert('Microphone access requires a secure origin. Open the app using http://localhost or https:// so the browser can grant microphone permission.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      saveOnStopRef.current = true;
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

      visualizeMic(stream);

      const capture = await recordWithAudioContext(stream, async (wavBlob, duration) => {
        try {
          if (saveOnStopRef.current) {
            const url = URL.createObjectURL(wavBlob);
            const a = new Audio(url);
            const finalized = (durSeconds: number) => {
              const safeDuration = Math.max(1, Math.round(durSeconds || duration || recordingTime || 0));
              if (wavBlob.size < 12000 || safeDuration < 2) {
                alert('Your microphone recording is too short or too quiet. Please record a clear voice sample for at least 2 seconds.');
                return;
              }

              const newSample: VoiceSample = {
                id: `mic-${Date.now()}`,
                name: `mic_recorded_${label.replace(/\s+/g, '_').toLowerCase()}.wav`,
                duration: safeDuration,
                size: `${(wavBlob.size / 1024 / 1024).toFixed(1)} MB`,
                fileUrl: url,
              };
              onSaveRecord(newSample);
            };
            a.addEventListener('loadedmetadata', () => finalized(a.duration || duration || recordingTime || 0), { once: true });
            window.setTimeout(() => finalized(duration || recordingTime || 1), 2000);
          }
        } catch (error) {
          console.error('Microphone audio conversion failed:', error);
          alert('Could not process the microphone recording. Please try again.');
        } finally {
          setIsRecording(false);
          setRecordingTime(0);
          cleanupRecording();
        }
      });

      mediaRecorderRef.current = capture as unknown as MediaRecorder;
    } catch (error: unknown) {
      console.error('Microphone access error:', error);
      const message =
        error instanceof DOMException &&
        (error.name === 'NotAllowedError' ||
          error.name === 'PermissionDeniedError' ||
          error.name === 'NotFoundError' ||
          error.name === 'SecurityError')
          ? 'Microphone access was denied or blocked by browser security settings. Please allow microphone access and use HTTPS or localhost.'
          : 'Unable to access microphone. Please allow microphone access and try again.';
      alert(message);
    }
  };

  const stopRecording = (shouldSave = true) => {
    const recorder = mediaRecorderRef.current as { stop?: () => void; state?: string } | null;
    if (!recorder || typeof recorder.stop !== 'function') return;

    saveOnStopRef.current = shouldSave;
    setIsRecording(false);
    try {
      recorder.stop();
    } catch (error) {
      console.error('Could not stop microphone recorder:', error);
      mediaRecorderRef.current = null;
      cleanupRecording();
      setRecordingTime(0);
    }
  };

  const visualizeMic = (stream: MediaStream) => {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AudioContextClass();
    audioCtxRef.current = ctx;

    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 64;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!canvasRef.current) return;
      const canvas = canvasRef.current;
      const canvasCtx = canvas.getContext('2d');
      if (!canvasCtx) return;

      analyser.getByteFrequencyData(dataArray);
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i += 1) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        canvasCtx.fillStyle = '#00f5ff';
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }

      animationFrameRef.current = requestAnimationFrame(draw);
    };

    draw();
  };

  const total = totalDuration || sample?.duration || 0;
  const progress = total ? Math.min((currentTime / total) * 100, 100) : 0;
  const progressWidth = `${progress}%`;

  const seekPlayback = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextTime = Number(event.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = nextTime;
      setCurrentTime(nextTime);
    }
  };

  return (
    <div className="rounded-xl p-4 border border-white/8 bg-transparent flex flex-col gap-4">
      {/* Hidden audio element used for playback and metadata */}
      <audio
        ref={audioRef}
        style={{ display: 'none' }}
        onTimeUpdate={() => {
          if (audioRef.current) setCurrentTime(audioRef.current.currentTime);
        }}
        onEnded={() => {
          setIsPlaying(false);
          setCurrentTime(0);
        }}
        onLoadedMetadata={() => {
          if (audioRef.current) setTotalDuration(audioRef.current.duration || 0);
        }}
      />

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
        className="hidden"
        onChange={handleFileChange}
      />

      <div className="flex items-center justify-between">
        <span className={`font-mono text-xs font-bold uppercase tracking-widest ${labelColor}`}>
          {label}
        </span>
        <FileAudio className="w-5 h-5 text-zinc-400/80" />
      </div>

      {sample ? (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="text-sm font-medium truncate">{sample.name}</div>
              <div className="text-xs text-zinc-400">{formatTime(sample.duration)} • {sample.size}</div>
            </div>
            <div className="flex items-center gap-2">
              {sample.fileUrl && (
                <button
                  type="button"
                  onClick={togglePlay}
                  className="p-2 rounded-full bg-white/5 text-white"
                  title={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
              )}
              <button
                type="button"
                onClick={() => { onRemoveSample(); setIsPlaying(false); setCurrentTime(0); }}
                className="p-2 rounded-full text-zinc-400 hover:text-rose-400"
                title="Remove"
              >
                <Trash2 className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="ml-2 px-3 py-1 rounded-md border border-white/8 text-sm"
              >
                Replace
              </button>
            </div>
          </div>

          {sample.fileUrl && (
            <div className="flex flex-col gap-1">
              <input
                type="range"
                min="0"
                max={total || 0}
                step="0.01"
                value={Math.min(currentTime, total || 0)}
                onChange={seekPlayback}
                disabled={!total}
                aria-label={`${label} playback position`}
                className="w-full accent-cyan-400 cursor-pointer disabled:cursor-default"
              />
              <div className="flex justify-between text-[11px] text-zinc-500 font-mono tabular-nums">
                <span>{formatTime(currentTime)}</span>
                <span>/{formatTime(Math.max(0, total - currentTime))} remaining</span>
              </div>
            </div>
          )}
          </div>
      ) : isRecording ? (
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-rose-500 flex items-center justify-center animate-pulse">
              <Mic className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="text-sm font-medium">Recording</div>
              <div className="text-xs text-zinc-400">{formatTime(recordingTime)}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" onPointerDown={(e) => e.stopPropagation()} onClick={() => stopRecording(true)} className="relative z-10 px-3 py-1 rounded-md border">Stop</button>
            <button type="button" onPointerDown={(e) => e.stopPropagation()} onClick={() => stopRecording(false)} className="relative z-10 px-3 py-1 rounded-md">Cancel</button>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3" onClick={() => fileInputRef.current?.click()}>
            <div className="w-10 h-10 rounded-md bg-white/5 flex items-center justify-center">
              <Upload className="w-5 h-5 text-zinc-300" />
            </div>
            <div>
              <div className="text-sm font-medium">Upload or drag audio</div>
              <div className="text-xs text-zinc-400">WAV, MP3, M4A, OGG, FLAC</div>
            </div>
          </div>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); startRecording(); }}
            className="flex items-center justify-center transition-all duration-200 hover:scale-110 active:scale-95 p-1"
            title="Record"
          >
            <Mic className="w-5 h-5 text-white/60 hover:text-white" />
          </button>
        </div>
      )}
    </div>
  );
};
