import React from 'react';
import { Activity, RefreshCw } from 'lucide-react';

interface HeaderProps {
  onResetAll: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onResetAll }) => {
  return (
    <header className="w-full mb-6">
      <div className="flex items-center justify-between gap-4 mb-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-[#1c1b1d] border border-white/10 px-3 py-1.5 rounded-lg">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-[#00f5ff]/20 to-[#a100f0]/30 border border-[#00f5ff]/40 flex items-center justify-center">
              <Activity className="w-3.5 h-3.5 text-[#00f5ff]" />
            </div>
            <span className="font-mono text-[10px] tracking-widest text-[#00f5ff] uppercase font-semibold">
              VoiceMatch AI
            </span>
          </div>

          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[#e5e1e4]">
            Voice Match AI
          </h1>
        </div>

        <button
          onClick={onResetAll}
          className="p-2 rounded-full bg-[#1c1b1d] border border-white/10 hover:border-[#00f5ff]/50 text-zinc-400 hover:text-[#00f5ff] transition-all"
          title="Reset All Inputs"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      <p className="text-sm text-zinc-400 font-sans leading-relaxed">
        Instant AI voice match—find out if two recordings are the same speaker.
      </p>
    </header>
  );
};