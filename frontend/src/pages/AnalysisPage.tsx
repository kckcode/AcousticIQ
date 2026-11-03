import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Play, 
  FileAudio, 
  Cpu, 
  Layers, 
  AlertCircle, 
  Check, 
  Loader2, 
  RotateCw,
  Info
} from 'lucide-react';
import { WaveformVisualizer } from '../components/WaveformVisualizer';
import { SpectrogramViewer } from '../components/SpectrogramViewer';
import { AnomalyGauge } from '../components/AnomalyGauge';
import { analyzeAudioFile, fetchSamples } from '../services/api';
import { AnalysisResponse, PreloadedSample } from '../types';

interface AnalysisPageProps {
  onAnalysisComplete?: (res: AnalysisResponse) => void;
}

export const AnalysisPage: React.FC<AnalysisPageProps> = ({ onAnalysisComplete }) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('valve');
  const [modelType, setModelType] = useState<'proposed' | 'baseline'>('proposed');
  const [file, setFile] = useState<File | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [currentFilename, setCurrentFilename] = useState<string>('');
  
  const [samples, setSamples] = useState<PreloadedSample[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    async function loadSamples() {
      setLoadingSamples(true);
      try {
        const s = await fetchSamples();
        setSamples(s);
      } catch (e) {
        console.error('Failed to load samples', e);
      } finally {
        setLoadingSamples(false);
      }
    }
    loadSamples();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setAudioBlob(selectedFile);
      setAudioUrl(null);
      setCurrentFilename(selectedFile.name);
      setAnalysisResult(null);
      setErrorMsg(null);
    }
  };

  const handleSelectPreloaded = async (s: PreloadedSample) => {
    setSelectedCategory(s.machine_category);
    setCurrentFilename(s.filename);
    setFile(null);
    setAnalysisResult(null);
    setErrorMsg(null);

    try {
      // Fetch the sample audio blob
      const res = await fetch(s.url);
      const blob = await res.blob();
      setAudioBlob(blob);
      setAudioUrl(s.url);
    } catch (e) {
      setErrorMsg('Failed to load preloaded audio sample.');
    }
  };

  const executeAnalysis = async () => {
    if (!audioBlob && !file) {
      setErrorMsg('Please upload a WAV audio recording or select a preloaded sample first.');
      return;
    }

    setAnalyzing(true);
    setErrorMsg(null);

    try {
      const targetBlob = file || audioBlob!;
      const filename = currentFilename || 'recording.wav';
      const result = await analyzeAudioFile(targetBlob, filename, selectedCategory, modelType);
      setAnalysisResult(result);
      if (onAnalysisComplete) {
        onAnalysisComplete(result);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'An unexpected error occurred during inference.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-mono">
          Acoustic Anomaly Detection & Inference
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Upload machine acoustic recordings or evaluate pre-calibrated sample waveforms through the deep autoencoder pipeline.
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-lg bg-rose-950/60 border border-rose-500/50 flex items-start space-x-3 text-rose-200 text-xs">
          <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Inference Error</div>
            <p>{errorMsg}</p>
          </div>
        </div>
      )}

      {/* Two Column Layout: Controls & Upload on Left, Telemetry & Decision on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Machine Configuration (5 Cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Machine Category Selector */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-4 space-y-3">
            <label className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider block">
              1. Select Industrial Machine Category
            </label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'valve', label: 'Valve (Pneumatic)' },
                { id: 'pump', label: 'Centrifugal Pump' },
                { id: 'fan', label: 'Ventilation Fan' },
                { id: 'slider', label: 'Linear Guide Rail' },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelectedCategory(m.id)}
                  className={`p-2.5 rounded border text-xs font-mono text-left transition-all ${
                    selectedCategory === m.id
                      ? 'bg-blue-600/20 border-blue-500 text-white font-semibold'
                      : 'bg-industrial-950 border-industrial-800 text-slate-400 hover:border-industrial-700'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Model Architecture Selector */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-4 space-y-3">
            <label className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider block">
              2. Select Neural Model Branch
            </label>
            <div className="grid grid-cols-1 gap-2">
              <button
                onClick={() => setModelType('proposed')}
                className={`p-3 rounded border text-left transition-all ${
                  modelType === 'proposed'
                    ? 'bg-emerald-950/40 border-emerald-500 text-white'
                    : 'bg-industrial-950 border-industrial-800 text-slate-400 hover:border-industrial-700'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-mono font-semibold text-emerald-400">
                  <span>Proposed Adaptive Fusion Autoencoder</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20">RESEARCH CONTRIBUTION</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Combines Log-Mel + Inverse-Mel feature streams with a learned latent gating gate conditioned on ambient noise.
                </p>
              </button>

              <button
                onClick={() => setModelType('baseline')}
                className={`p-3 rounded border text-left transition-all ${
                  modelType === 'baseline'
                    ? 'bg-blue-950/40 border-blue-500 text-white'
                    : 'bg-industrial-950 border-industrial-800 text-slate-400 hover:border-industrial-700'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-mono font-semibold text-blue-400">
                  <span>Baseline Convolutional Autoencoder</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20">STANDARD BENCHMARK</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Single-branch deep convolutional autoencoder reconstructing standard Log-Mel spectrograms.
                </p>
              </button>
            </div>
          </div>

          {/* Upload Area */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-4 space-y-3">
            <label className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider block">
              3. Upload WAV Audio Recording
            </label>
            
            <div className="relative border-2 border-dashed border-industrial-750 hover:border-blue-500/60 rounded-lg p-6 text-center transition-colors bg-industrial-950">
              <input
                type="file"
                accept=".wav,audio/wav,audio/x-wav"
                onChange={handleFileUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
              <div className="text-xs text-slate-300 font-medium">
                {currentFilename ? (
                  <span className="text-blue-400 font-mono font-bold">{currentFilename}</span>
                ) : (
                  <span>Drag & Drop WAV file or click to browse</span>
                )}
              </div>
              <p className="text-[11px] text-slate-500 mt-1 font-mono">
                Supports standard WAV format (up to 25 MB). Target window: 10.0 seconds @ 16 kHz.
              </p>
            </div>

            {/* Preloaded Samples Bar */}
            <div className="pt-2">
              <div className="text-[11px] font-mono text-slate-400 mb-2 flex items-center justify-between">
                <span>Or evaluate pre-calibrated machine sample:</span>
                {loadingSamples && <Loader2 className="w-3 h-3 animate-spin text-blue-400" />}
              </div>
              <div className="grid grid-cols-2 gap-1.5 max-h-36 overflow-y-auto pr-1">
                {samples.map((s) => (
                  <button
                    key={s.filename}
                    onClick={() => handleSelectPreloaded(s)}
                    className={`px-2 py-1.5 rounded text-[11px] font-mono text-left border truncate transition-all ${
                      currentFilename === s.filename
                        ? 'bg-blue-600/20 border-blue-500 text-white font-medium'
                        : 'bg-industrial-950 border-industrial-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span className={s.label === 'anomaly' ? 'text-rose-400' : 'text-emerald-400'}>
                      ● {s.machine_category.toUpperCase()} ({s.label})
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Analyze Action Button */}
            <button
              onClick={executeAnalysis}
              disabled={analyzing || (!audioBlob && !file)}
              className={`w-full py-3 rounded-lg text-xs font-mono font-semibold uppercase tracking-wider flex items-center justify-center space-x-2 transition-all ${
                analyzing || (!audioBlob && !file)
                  ? 'bg-industrial-800 text-slate-500 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/30'
              }`}
            >
              {analyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>PREPROCESSING & INFERRING...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>RUN ACOUSTIC INFERENCE</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Visualization & Results (7 Cols) */}
        <div className="lg:col-span-7 space-y-5">
          {/* Waveform Player */}
          <WaveformVisualizer
            waveform={analysisResult?.visualizations?.waveform_downsampled || []}
            times={analysisResult?.visualizations?.waveform_times || []}
            audioUrl={audioUrl || undefined}
            audioBlob={audioBlob || undefined}
            durationSeconds={analysisResult?.duration_seconds || 10.0}
          />

          {/* Anomaly Score Gauge (renders when analysis is done) */}
          {analysisResult ? (
            <AnomalyGauge
              score={analysisResult.anomaly_score}
              threshold={analysisResult.decision_threshold}
              isAnomaly={analysisResult.is_anomaly}
              decisionText={analysisResult.classification_decision}
              confidence={analysisResult.confidence_rating}
              processingTimeMs={analysisResult.processing_time_ms}
              adaptiveGateAlpha={analysisResult.feature_config?.adaptive_gate_alpha}
              isSynthetic={analysisResult.is_synthetic_fixture}
            />
          ) : (
            <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-6 text-center text-xs font-mono text-slate-500">
              Acoustic anomaly score, decision threshold meter, and model diagnostics will appear here after clicking Run Acoustic Inference.
            </div>
          )}

          {/* Time-Frequency Spectrogram Viewer */}
          <SpectrogramViewer
            spectrogramData={analysisResult?.visualizations}
            durationSeconds={analysisResult?.duration_seconds || 10.0}
          />

          {/* System Limitations & Disclaimer Box */}
          <div className="p-4 rounded-lg bg-industrial-900 border border-industrial-800 text-xs text-slate-400 space-y-1">
            <div className="flex items-center space-x-1.5 text-slate-300 font-mono font-medium">
              <Info className="w-4 h-4 text-blue-400" />
              <span>Research Protocol & Diagnostic Scope</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              This system implements acoustic anomaly detection to quantify deviation from normal machine baselines. An anomaly detection result flags an irregular sound pattern and warrants inspection; it does not confirm physical failure mechanisms or estimate Remaining Useful Life (RUL).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
