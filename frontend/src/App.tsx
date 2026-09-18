import React, { useState } from 'react';
import { Header } from './Header';
import { VoiceCard } from './components/VoiceCard';
import { SimilarityResult } from './components/SimilarityResult';
// Removed threshold UI controls; threshold is fixed server-side
import { VoiceSample, VerificationResult } from './types';

export default function App() {
  const [voice1, setVoice1] = useState<VoiceSample | null>(null);
  const [voice2, setVoice2] = useState<VoiceSample | null>(null);
  // No client-controlled threshold; the backend enforces the production threshold.
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerificationResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const getVerifyUrl = () => {
    const apiUrl = (import.meta as any).env?.VITE_API_URL;
    if (apiUrl && apiUrl.trim().length > 0) {
      return `${String(apiUrl).replace(/\/+$/, '')}/api/verify`;
    }

    // In dev (Vite) use the proxy path so `/api` forwards to backend.
    // In production/preview (no dev proxy) call the backend directly.
    // `import.meta.env.DEV` is true when running `vite` dev server.
    if ((import.meta as any).env?.DEV) {
      return '/api/verify';
    }

    // Default to localhost backend for preview/static builds
    return 'http://127.0.0.1:8000/api/verify';
  };

  const runAnalysis = async () => {
    if (!voice1 || !voice2) return;

    setErrorMessage(null);
    setIsAnalyzing(true);

    try {
      const [blob1Res, blob2Res] = await Promise.all([
        fetch(voice1.fileUrl),
        fetch(voice2.fileUrl),
      ]);

      if (!blob1Res.ok || !blob2Res.ok) {
        throw new Error('Failed to load local audio files for upload.');
      }

      const blob1 = await blob1Res.blob();
      const blob2 = await blob2Res.blob();

      const formData = new FormData();
      formData.append('voice1', blob1, voice1.name);
      formData.append('voice2', blob2, voice2.name);

      const response = await fetch(getVerifyUrl(), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server error ${response.status}: ${text}`);
      }

      const data = await response.json();

      setVerificationResult(data);
      setErrorMessage(null);
    } catch (error) {
      console.error('Error verifying voice similarity:', error);
      const message = error instanceof Error ? error.message : 'Unknown backend error';
      setErrorMessage(message);
      setVerificationResult(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleUploadSample = (slot: 'VOICE 1' | 'VOICE 2', file: File) => {
    const url = URL.createObjectURL(file);

    // Create an audio element to read metadata (duration) before saving sample
    const audio = new Audio(url);
    const save = (durationSeconds: number) => {
      const newSample: VoiceSample = {
        id: `uploaded-${Date.now()}`,
        name: file.name,
        duration: Math.max(1, Math.round(durationSeconds)),
        size: `${(file.size / 1024 / 1024).toFixed(1)} MB`,
        fileUrl: url,
      };

      if (slot === 'VOICE 1') setVoice1(newSample);
      else setVoice2(newSample);
    };

    const onLoaded = () => {
      const dur = audio.duration || 0;
      save(dur);
      audio.removeEventListener('loadedmetadata', onLoaded);
    };

    audio.addEventListener('loadedmetadata', onLoaded);

    // Fallback timeout: if metadata doesn't load, still save sample after short delay
    setTimeout(() => {
      if (!audio.duration || isNaN(audio.duration)) {
        save(0);
      }
    }, 2000);
  };

  const handleResetAll = () => {
    setVoice1(null);
    setVoice2(null);
    setVerificationResult(null);
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen bg-[#131315] text-[#e5e1e4] font-sans py-8 px-4 sm:px-6 md:px-12 max-w-2xl mx-auto flex flex-col justify-between">
      <div>
        <Header onResetAll={handleResetAll} />

        <div className="grid grid-cols-1 gap-5 my-5">
          <VoiceCard
            label="Voice 1"
 pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp           labelColor="text-[#00f5ff]"
            sample={voice1}
            onUploadSample={(file) => handleUploadSample('VOICE 1', file)}
            onSaveRecord={(sample) => setVoice1(sample)}
            onRemoveSample={() => setVoice1(null)}
          />

          <VoiceCard
            label="Voice 2"
            labelColor="text-[#e5b5ff]"
            sample={voice2}
            onUploadSample={(file) => handleUploadSample('VOICE 2', file)}
            onSaveRecord={(sample) => setVoice2(sample)}
            onRemoveSample={() => setVoice2(null)}
          />
        </div>

        {/* Threshold controls removed for streamlined UX. */}

        {errorMessage && (
          <div className="mb-5 rounded-2xl border border-rose-400/50 bg-rose-500/10 p-4 text-rose-200 text-sm font-medium">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <strong>Error:</strong> {errorMessage}
              </div>
              <button
                type="button"
                onClick={runAnalysis}
                disabled={isAnalyzing}
                className="rounded-xl bg-white/10 border border-white/20 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-white transition hover:bg-white/15 disabled:opacity-50"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        <SimilarityResult
          result={verificationResult}
          sample1={voice1}
          sample2={voice2}
          onCompare={runAnalysis}
          isAnalyzing={isAnalyzing}
        />
      </div>

    </div>
  );
}