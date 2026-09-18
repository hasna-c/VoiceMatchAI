export interface VoiceSample {
  id: string;
  name: string;
  duration: number;
  size: string;
  fileUrl: string;
}

export interface VerificationResult {
  score: number;
  isMatch: boolean;
  confidence?: number;
  verdict?: string;
  explanation: string;
  requestedThreshold?: number;
  effectiveThreshold?: number;
  rawSimilarity?: number;
  segmentSimilarity?: number | null;
  durationSeconds?: [number, number];
  qualityScores?: [number, number];
}