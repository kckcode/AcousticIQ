import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, RotateCcw } from 'lucide-react';

interface WaveformVisualizerProps {
  waveform: number[];
  times?: number[];
  audioUrl?: string;
  audioBlob?: Blob;
  durationSeconds: number;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  waveform,
  times,
  audioUrl,
  audioBlob,
  durationSeconds,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [localUrl, setLocalUrl] = useState<string | null>(null);

  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      setLocalUrl(url);
      return () => URL.revokeObjectURL(url);
    } else if (audioUrl) {
      setLocalUrl(audioUrl);
    }
  }, [audioBlob, audioUrl]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(() => {});
    }
    setIsPlaying(!isPlaying);
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || durationSeconds <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const progress = Math.max(0, Math.min(1, clickX / rect.width));
    const seekTime = progress * durationSeconds;
    audioRef.current.currentTime = seekTime;
    setCurrentTime(seekTime);
  };

  const progressPercent = durationSeconds > 0 ? (currentTime / durationSeconds) * 100 : 0;

  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Volume2 className="w-4 h-4 text-blue-400" />
          <span className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
            Raw Acoustic Waveform & Audio Player
          </span>
        </div>
        <div className="text-xs font-mono text-slate-400">
          <span>{currentTime.toFixed(2)}s</span> / <span>{durationSeconds.toFixed(2)}s</span>
        </div>
      </div>

      {localUrl && (
        <audio
          ref={audioRef}
          src={localUrl}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleEnded}
        />
      )}

      {/* Waveform Canvas / SVG Container */}
      <div
        className="relative h-24 bg-industrial-950 border border-industrial-800 rounded cursor-pointer overflow-hidden group"
        onClick={handleSeek}
      >
        {/* Playhead scrub line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 z-10 shadow-[0_0_8px_rgba(6,182,212,0.8)]"
          style={{ left: `${progressPercent}%` }}
        />

        {/* Render waveform bars */}
        <div className="absolute inset-0 flex items-center justify-between px-1">
          {waveform && waveform.length > 0 ? (
            waveform.map((val, idx) => {
              const heightPercent = Math.max(4, Math.min(100, Math.abs(val) * 100));
              const isPast = (idx / waveform.length) * 100 <= progressPercent;
              return (
                <div
                  key={idx}
                  className={`w-1 rounded-full transition-colors duration-75 ${
                    isPast
                      ? 'bg-blue-400'
                      : 'bg-industrial-700 group-hover:bg-industrial-600'
                  }`}
                  style={{ height: `${heightPercent}%` }}
                />
              );
            })
          ) : (
            <div className="w-full text-center text-xs text-slate-500 font-mono">
              Waveform telemetry will render here upon upload.
            </div>
          )}
        </div>
      </div>

      {/* Playback Controls */}
      <div className="flex items-center justify-between pt-1">
        <div className="flex items-center space-x-2">
          <button
            onClick={togglePlay}
            disabled={!localUrl}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-mono font-medium transition-colors ${
              localUrl
                ? 'bg-blue-600 hover:bg-blue-500 text-white'
                : 'bg-industrial-800 text-slate-500 cursor-not-allowed'
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="w-3.5 h-3.5" />
                <span>PAUSE</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                <span>PLAY AUDIBLE SIGNAL</span>
              </>
            )}
          </button>

          <button
            onClick={() => {
              if (audioRef.current) {
                audioRef.current.currentTime = 0;
                setCurrentTime(0);
              }
            }}
            disabled={!localUrl}
            className="p-1.5 rounded bg-industrial-800 hover:bg-industrial-700 text-slate-400 hover:text-slate-200 transition-colors"
            title="Reset to 0s"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>

        <span className="text-[11px] font-mono text-slate-500">
          PCM 16-bit Mono @ 16 kHz
        </span>
      </div>
    </div>
  );
};
