import React, { useState, useRef, useEffect } from 'react';
import { Layers, HelpCircle } from 'lucide-react';
import { SpectrogramData } from '../types';

interface SpectrogramViewerProps {
  spectrogramData?: SpectrogramData;
  durationSeconds: number;
}

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({
  spectrogramData,
  durationSeconds,
}) => {
  const [selectedBranch, setSelectedBranch] = useState<'mel' | 'inv_mel'>('mel');
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const matrix =
    selectedBranch === 'mel'
      ? spectrogramData?.mel_spectrogram || spectrogramData?.magnitudes
      : spectrogramData?.inverse_mel_spectrogram || spectrogramData?.magnitudes;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrix || matrix.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const nBins = matrix.length; // e.g. 32
    const nFrames = matrix[0].length; // e.g. 64

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const cellWidth = width / nFrames;
    const cellHeight = height / nBins;

    // Determine min/max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let r = 0; r < nBins; r++) {
      for (let c = 0; c < nFrames; c++) {
        const v = matrix[r][c];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    const range = Math.max(1e-5, maxVal - minVal);

    // Color gradient mapping: deep indigo -> blue -> teal -> yellow -> bright orange
    const getColor = (norm: number) => {
      // norm is in [0, 1]
      const clamped = Math.max(0, Math.min(1, norm));
      const r = Math.round(255 * Math.sin(clamped * Math.PI * 0.7));
      const g = Math.round(220 * Math.pow(clamped, 1.4));
      const b = Math.round(255 * (1 - clamped));
      return `rgb(${r}, ${g}, ${b})`;
    };

    // Draw spectrogram cells (frequency 0 at bottom)
    for (let r = 0; r < nBins; r++) {
      // Invert rows so low frequencies are at the bottom
      const y = height - (r + 1) * cellHeight;
      for (let c = 0; c < nFrames; c++) {
        const x = c * cellWidth;
        const norm = (matrix[r][c] - minVal) / range;
        ctx.fillStyle = getColor(norm);
        ctx.fillRect(x, y, cellWidth + 0.5, cellHeight + 0.5);
      }
    }
  }, [matrix]);

  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-4 space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
            Time-Frequency Spectrogram Analysis
          </span>
        </div>

        {/* Representation Branch Switcher */}
        <div className="flex items-center space-x-1 bg-industrial-950 p-1 rounded border border-industrial-800 text-xs font-mono">
          <button
            onClick={() => setSelectedBranch('mel')}
            className={`px-2.5 py-1 rounded transition-colors ${
              selectedBranch === 'mel'
                ? 'bg-blue-600 text-white font-medium'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Branch A: Log-Mel Spectrogram
          </button>
          <button
            onClick={() => setSelectedBranch('inv_mel')}
            className={`px-2.5 py-1 rounded transition-colors ${
              selectedBranch === 'inv_mel'
                ? 'bg-cyan-600 text-white font-medium'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Branch B: Inverse-Mel Reconstruction
          </button>
        </div>
      </div>

      {/* Canvas Heatmap Area */}
      <div className="relative border border-industrial-800 rounded bg-industrial-950 overflow-hidden">
        {matrix && matrix.length > 0 ? (
          <div className="p-2">
            <div className="flex">
              {/* Frequency Axis Labels (Vertical) */}
              <div className="flex flex-col justify-between pr-2 text-[10px] font-mono text-slate-500 text-right select-none h-44 w-12">
                <span>8.0 kHz</span>
                <span>4.0 kHz</span>
                <span>2.0 kHz</span>
                <span>500 Hz</span>
                <span>0 Hz</span>
              </div>

              {/* Heatmap Canvas */}
              <div className="flex-1 relative">
                <canvas
                  ref={canvasRef}
                  width={600}
                  height={176}
                  className="w-full h-44 rounded object-cover"
                />
              </div>
            </div>

            {/* Time Axis Labels (Horizontal) */}
            <div className="flex justify-between pl-14 pt-1 text-[10px] font-mono text-slate-500 select-none">
              <span>0.0s</span>
              <span>2.5s</span>
              <span>5.0s</span>
              <span>7.5s</span>
              <span>10.0s</span>
            </div>
          </div>
        ) : (
          <div className="h-44 flex items-center justify-center text-xs font-mono text-slate-500">
            No spectrogram telemetry available. Upload or select an audio sample to extract features.
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
        <div className="flex items-center space-x-2">
          <span>Scale:</span>
          <div className="flex items-center space-x-1">
            <span className="text-[10px] text-slate-500">Low dB</span>
            <div className="h-2 w-20 rounded bg-gradient-to-r from-blue-900 via-teal-500 to-amber-300" />
            <span className="text-[10px] text-slate-500">Peak dB</span>
          </div>
        </div>

        <span className="text-slate-500">
          {selectedBranch === 'mel'
            ? 'Non-linear Mel filterbank (64 bands, 0-8 kHz)'
            : 'Linear pseudo-inverse harmonic reconstruction'}
        </span>
      </div>
    </div>
  );
};
