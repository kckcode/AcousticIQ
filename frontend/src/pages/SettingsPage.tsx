import React from 'react';
import { 
  Settings, 
  Server, 
  Cpu, 
  Sliders, 
  HardDrive, 
  ShieldCheck, 
  FileText,
  ExternalLink 
} from 'lucide-react';
import { HealthResponse } from '../types';

interface SettingsPageProps {
  health?: HealthResponse | null;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ health }) => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-mono">
          System Configuration & Telemetry Status
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Runtime environment parameters, hardware compute backend, and audio processing hyperparameters.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backend & Hardware Telemetry */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
          <div className="flex items-center space-x-2 border-b border-industrial-800 pb-3">
            <Server className="w-5 h-5 text-blue-400" />
            <h2 className="text-sm font-mono font-medium text-white uppercase tracking-wider">
              Compute & Server Runtime
            </h2>
          </div>

          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Service Status:</span>
              <span className="text-emerald-400 font-bold">{health?.status.toUpperCase() || 'UNKNOWN'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Application Version:</span>
              <span className="text-slate-200">{health?.version || '1.0.0'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Environment:</span>
              <span className="text-slate-200">{health?.environment || 'development'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Database Engine:</span>
              <span className="text-slate-200">SQLite (Local ACID Persistence)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">PyTorch Compute Device:</span>
              <span className="text-cyan-400 font-bold">{health?.active_device || 'CPU'}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">CUDA Acceleration Available:</span>
              <span className={health?.cuda_available ? 'text-emerald-400' : 'text-slate-400'}>
                {health?.cuda_available ? 'True (NVIDIA RTX)' : 'False (CPU Fallback)'}
              </span>
            </div>
          </div>
        </div>

        {/* Audio Processing & Feature Hyperparameters */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
          <div className="flex items-center space-x-2 border-b border-industrial-800 pb-3">
            <Sliders className="w-5 h-5 text-cyan-400" />
            <h2 className="text-sm font-mono font-medium text-white uppercase tracking-wider">
              Acoustic Signal Processing Settings
            </h2>
          </div>

          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Default Sample Rate:</span>
              <span className="text-slate-200">16,000 Hz</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Target Audio Window:</span>
              <span className="text-slate-200">10.0 seconds (160,000 samples)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">FFT Window Size (n_fft):</span>
              <span className="text-slate-200">1,024</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Hop Length:</span>
              <span className="text-slate-200">512 samples</span>
            </div>
            <div className="flex justify-between py-1 border-b border-industrial-850">
              <span className="text-slate-400">Mel Filterbank Bins (n_mels):</span>
              <span className="text-slate-200">64 channels</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Max Upload File Limit:</span>
              <span className="text-slate-200">25 MB (WAV)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Research Documentation & Citations */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-3">
        <div className="flex items-center space-x-2">
          <FileText className="w-5 h-5 text-blue-400" />
          <h2 className="text-sm font-mono font-medium text-white uppercase tracking-wider">
            Research Dataset & Scientific Attribution
          </h2>
        </div>

        <div className="text-xs text-slate-300 leading-relaxed space-y-2">
          <p>
            The benchmark models are evaluated on the <b>MIMII Industrial Machine Sound Dataset</b> (Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection), hosted on Zenodo under record ID <b>3384388</b> and licensed under CC BY 4.0.
          </p>
          <div className="p-3 bg-industrial-950 rounded border border-industrial-800 font-mono text-[11px] text-slate-400">
            Citation: Harsh Purohit, Ryo Tanabe, Kenji Nishida, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, and Yohei Kawaguchi. "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection." In DCASE 2019 Workshop, 2019.
          </div>
        </div>
      </div>
    </div>
  );
};
