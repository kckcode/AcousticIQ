import React from 'react';
import { AlertTriangle, CheckCircle2, Gauge, ShieldAlert } from 'lucide-react';

interface AnomalyGaugeProps {
  score: number;
  threshold: number;
  isAnomaly: boolean;
  decisionText: string;
  confidence: string;
  processingTimeMs: number;
  adaptiveGateAlpha?: number;
  isSynthetic?: boolean;
}

export const AnomalyGauge: React.FC<AnomalyGaugeProps> = ({
  score,
  threshold,
  isAnomaly,
  decisionText,
  confidence,
  processingTimeMs,
  adaptiveGateAlpha,
  isSynthetic,
}) => {
  // Max scale is 2.5 times threshold for comfortable visual range
  const maxScale = Math.max(threshold * 2.2, score * 1.25, 0.1);
  const scorePercent = Math.min(100, (score / maxScale) * 100);
  const thresholdPercent = Math.min(100, (threshold / maxScale) * 100);

  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Gauge className="w-5 h-5 text-blue-400" />
          <h3 className="text-sm font-mono font-medium text-white uppercase tracking-wider">
            Quantitative Acoustic Anomaly Decision
          </h3>
        </div>

        {isSynthetic && (
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
            SYNTHETIC FIXTURE
          </span>
        )}
      </div>

      {/* Decision Banner */}
      <div
        className={`p-4 rounded-lg border flex items-center justify-between ${
          isAnomaly
            ? 'bg-rose-950/40 border-rose-500/40 text-rose-200'
            : 'bg-emerald-950/40 border-emerald-500/40 text-emerald-200'
        }`}
      >
        <div className="flex items-center space-x-3">
          {isAnomaly ? (
            <AlertTriangle className="w-6 h-6 text-rose-400 flex-shrink-0 animate-bounce" />
          ) : (
            <CheckCircle2 className="w-6 h-6 text-emerald-400 flex-shrink-0" />
          )}
          <div>
            <div className="text-base font-semibold tracking-tight">
              {decisionText}
            </div>
            <p className="text-xs opacity-80 font-mono">
              Confidence Rating: <span className="font-semibold">{confidence}</span>
            </p>
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs font-mono text-slate-400">Inference Latency</div>
          <div className="text-sm font-mono font-semibold">{processingTimeMs.toFixed(1)} ms</div>
        </div>
      </div>

      {/* Score vs Threshold Visual Bar */}
      <div className="space-y-2 pt-1">
        <div className="flex justify-between text-xs font-mono">
          <span className="text-slate-400">
            Reconstruction Score: <span className="text-white font-bold">{score.toFixed(5)}</span>
          </span>
          <span className="text-slate-400">
            Decision Threshold: <span className="text-amber-400 font-bold">{threshold.toFixed(5)}</span>
          </span>
        </div>

        <div className="relative h-4 bg-industrial-950 rounded-full border border-industrial-800 overflow-hidden">
          {/* Normal region (below threshold) */}
          <div
            className="absolute top-0 bottom-0 left-0 bg-emerald-950/50"
            style={{ width: `${thresholdPercent}%` }}
          />
          {/* Anomaly region (above threshold) */}
          <div
            className="absolute top-0 bottom-0 right-0 bg-rose-950/50"
            style={{ width: `${100 - thresholdPercent}%` }}
          />

          {/* Current Score Fill Bar */}
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              isAnomaly ? 'bg-gradient-to-r from-amber-500 to-rose-500' : 'bg-gradient-to-r from-blue-500 to-emerald-500'
            }`}
            style={{ width: `${scorePercent}%` }}
          />

          {/* Threshold Marker Needle */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-amber-400 z-10 shadow-[0_0_6px_rgba(245,158,11,0.9)]"
            style={{ left: `${thresholdPercent}%` }}
            title={`Threshold: ${threshold.toFixed(5)}`}
          />
        </div>

        <div className="flex justify-between text-[10px] font-mono text-slate-500">
          <span>0.00000 (Low Deviation)</span>
          <span className="text-amber-400">▲ Threshold (95% Val Normal)</span>
          <span>{maxScale.toFixed(4)} (Extreme Anomaly)</span>
        </div>
      </div>

      {/* Adaptive Gating Diagnostics */}
      {adaptiveGateAlpha !== undefined && (
        <div className="pt-2 border-t border-industrial-800 flex items-center justify-between text-xs font-mono text-slate-400">
          <div className="flex items-center space-x-2">
            <span className="text-slate-500">Adaptive Fusion Weights:</span>
            <span className="text-blue-400 font-semibold">Mel {(adaptiveGateAlpha * 100).toFixed(1)}%</span>
            <span>+</span>
            <span className="text-cyan-400 font-semibold">Inverse-Mel {((1 - adaptiveGateAlpha) * 100).toFixed(1)}%</span>
          </div>
          <span className="text-[10px] text-slate-500">Learned Gating Gate α={adaptiveGateAlpha.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
};
