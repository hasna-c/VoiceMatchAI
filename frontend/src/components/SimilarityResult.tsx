import React from 'react';
import { Sliders, Cpu } from 'lucide-react';
import { VerificationResult, VoiceSample } from '../types';

interface SimilarityResultProps {
  result: VerificationResult | null;
  sample1: VoiceSample | null;
  sample2: VoiceSample | null;
  onCompare: () => void;
  isAnalyzing: boolean;
}

export const SimilarityResult: React.FC<SimilarityResultProps> = ({
  result,
  sample1,
  sample2,
  onCompare,
  isAnalyzing,
}) => {
  const isButtonDisabled = isAnalyzing || !sample1 || !sample2;
  const hasResult = Boolean(result);

  const scoreValue = result?.score ?? 0;
  const scoreNumber = result ? Number(result.score) : null;
  const scoreDisplay = scoreNumber !== null
    ? (scoreNumber >= 0 ? `+${scoreNumber.toFixed(2)}` : scoreNumber.toFixed(2))
    : '--';
  // Prefer backend decision when available
  const passed = result ? (typeof result.isMatch === 'boolean' ? result.isMatch : false) : false;

  const confidencePct = result?.confidence ?? Math.round(((scoreNumber ?? 0) + 1) / 2 * 100);
  const scoreNormalized = (scoreValue + 1) / 2;
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference * (1 - scoreNormalized);
  const safeThreshold = 0.75;
  const micOnlyMode = Boolean(sample1?.name?.startsWith('mic_recorded_') && sample2?.name?.startsWith('mic_recorded_'));
  const decisionLabel = passed ? 'Match detected' : 'No match detected';
  const decisionTone = passed ? 'text-cyan-400' : 'text-amber-400';

  return (
    <div className="w-full my-6 flex flex-col items-center gap-6">
      <div className="relative w-full flex justify-center py-2">
        <div className="absolute inset-0 max-w-sm mx-auto bg-gradient-to-r from-[#00f5ff]/30 via-[#a100f0]/20 to-[#00f5ff]/30 blur-2xl opacity-75 pointer-events-none rounded-full" />

        <button
          onClick={onCompare}
          disabled={isButtonDisabled}
          className="relative group w-full max-w-sm sm:max-w-md py-4 px-8 rounded-2xl glow-cyan-btn font-mono font-bold text-sm sm:text-base tracking-wider flex items-center justify-center gap-3 transition-all duration-300 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          {isAnalyzing ? (
            <>
              <Cpu className="w-5 h-5 animate-spin text-[#002021]" />
              <span>COMPUTING EMBEDDINGS...</span>
            </>
          ) : (
            <>
              <Sliders className="w-5 h-5 text-[#002021] group-hover:rotate-90 transition-transform duration-300" />
              <span>COMPARE VOICES</span>
            </>
          )}
        </button>
      </div>

      <div className="w-full glass-card rounded-2xl p-8 border border-white/10 transition-all">
        <span className="font-mono text-xs text-zinc-400 uppercase tracking-widest block mb-8 text-center">
          SIMILARITY ANALYSIS
        </span>
        <div className="mb-6 text-center font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
          Decision threshold: <strong className="text-cyan-300">{(result?.effectiveThreshold ?? safeThreshold).toFixed(2)}</strong>
        </div>

        {!hasResult ? (
          <div className="rounded-3xl border border-white/10 bg-[#0c0c0f] p-8 text-center text-sm text-zinc-400">
            Upload or record audio for both voices and click Compare Voices to see results.
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex justify-center">
              <div className={`inline-flex items-center gap-2 bg-[#00f5ff]/10 border border-[#00f5ff]/40 px-4 py-2 rounded-xl ${decisionTone} font-mono font-bold text-xs sm:text-sm tracking-wider uppercase shadow-[0_0_15px_rgba(0,245,255,0.15)]`}>
                <span className={`w-2 h-2 rounded-full ${passed ? 'bg-[#00f5ff] animate-ping' : 'bg-amber-400'}`} />
                <span className={passed ? 'w-2 h-2 rounded-full bg-[#00f5ff] -ml-4' : 'w-2 h-2 rounded-full bg-amber-400 -ml-4'} />
                <span>{decisionLabel}</span>
              </div>
            </div>

            <div className="flex justify-center py-4">
              <div className="relative w-48 h-48">
                <svg className="w-full h-full" viewBox="0 0 120 120">
                  <circle
                    cx="60"
                    cy="60"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="8"
                    className="text-zinc-700/50"
                  />
                  <circle
                    cx="60"
                    cy="60"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="8"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    className={passed ? 'text-cyan-400' : 'text-rose-500'}
                    style={{ transition: 'all 0.6s ease-out' }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="text-4xl sm:text-5xl font-bold font-mono text-[#00f5ff] drop-shadow-[0_0_15px_rgba(0,245,255,0.6)]">
                    {confidencePct}%
                  </div>
                  <div className="text-xs sm:text-sm text-zinc-400 font-mono mt-1">
                    Confidence
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-2xl border border-white/10 bg-[#0e0e10] p-4 text-center">
                  <div className="text-xs text-zinc-500 font-mono uppercase tracking-wider mb-2">
                    Match likelihood
                  </div>
                  <div className={`text-lg sm:text-xl font-bold font-mono ${passed ? 'text-cyan-400' : 'text-rose-400'}`}>
                    {confidencePct}%
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-[#09090d] p-5 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.04)]">
                <div className="text-xs text-zinc-400 uppercase tracking-[0.22em] font-mono text-center mb-3">
                  Similarity score: {scoreDisplay}
                </div>
                <p className="text-xs sm:text-sm text-zinc-400 font-sans leading-relaxed text-center">
                  {passed
                    ? 'This result suggests the same speaker, but it is not a guaranteed identity match.'
                    : 'This result suggests a different speaker based on the current threshold.'}
                </p>
                {result?.rawSimilarity !== undefined && (
                  <div className="mt-4 grid grid-cols-2 gap-2 text-[10px] uppercase tracking-wider text-zinc-500 font-mono text-center">
                    <span>Raw: {result.rawSimilarity.toFixed(2)}</span>
                    <span>Segments: {result.segmentSimilarity?.toFixed(2) ?? '--'}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};