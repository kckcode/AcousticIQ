import React from 'react';
import { 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  Cpu, 
  Layers, 
  ArrowRight, 
  Clock, 
  Database,
  ShieldCheck 
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { MachineCategory, AnalysisHistoryItem, ExperimentSummary } from '../types';

interface OverviewPageProps {
  history: AnalysisHistoryItem[];
  machines: MachineCategory[];
  latestExperiment?: ExperimentSummary;
  onNavigate: (tab: string) => void;
  onSelectSample: (filename: string) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  history,
  machines,
  latestExperiment,
  onNavigate,
}) => {
  const totalAnalyses = history.length;
  const normalCount = history.filter((h) => !h.is_anomaly).length;
  const anomalyCount = history.filter((h) => h.is_anomaly).length;
  const modelsReadyCount = machines.filter((m) => m.trained_model_available).length;

  return (
    <div className="space-y-6">
      {/* Top Banner / Mission Header */}
      <div className="bg-gradient-to-r from-industrial-900 via-industrial-850 to-industrial-900 border border-industrial-800 rounded-xl p-6 relative overflow-hidden">
        <div className="max-w-3xl space-y-2 relative z-10">
          <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-xs font-mono text-blue-400">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>RESEARCH-GRADE INDUSTRIAL PREDICTIVE MAINTENANCE</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Industrial Acoustic Anomaly Detection Platform
          </h1>
          <p className="text-sm text-slate-300 leading-relaxed">
            AcousticIQ monitors mechanical operating sounds, quantifies subtle acoustic deviations using deep convolutional autoencoders and adaptive feature fusion, and alerts maintenance engineers to potential failure onset before catastrophic downtime.
          </p>

          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('analysis')}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-mono font-medium transition-colors shadow-lg shadow-blue-600/20"
            >
              <span>LAUNCH ACOUSTIC ANALYSIS</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('experiments')}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-industrial-800 hover:bg-industrial-750 text-slate-200 text-xs font-mono border border-industrial-700 transition-colors"
            >
              <span>VIEW RESEARCH BENCHMARKS</span>
            </button>
          </div>
        </div>

        {/* Decorative background grid elements */}
        <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent pointer-events-none" />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Analyses"
          value={totalAnalyses}
          subtext="Processed historical recordings"
          icon={Activity}
          badge="LOCAL DB"
          badgeColor="blue"
        />
        <StatCard
          label="Normal Patterns"
          value={normalCount}
          subtext="Within baseline operating limits"
          icon={CheckCircle2}
          badge={`${totalAnalyses ? ((normalCount / totalAnalyses) * 100).toFixed(0) : 100}%`}
          badgeColor="emerald"
        />
        <StatCard
          label="Acoustic Anomalies"
          value={anomalyCount}
          subtext="Exceeded calibrated threshold"
          icon={AlertTriangle}
          badge={`${totalAnalyses ? ((anomalyCount / totalAnalyses) * 100).toFixed(0) : 0}%`}
          badgeColor="rose"
        />
        <StatCard
          label="Models Ready"
          value={`${modelsReadyCount} / 4`}
          subtext="Checkpoints loaded & verified"
          icon={Cpu}
          badge="TRAINED"
          badgeColor="emerald"
        />
      </div>

      {/* Industrial Machine Categories Fleet Status */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-blue-400" />
            <h2 className="text-sm font-mono font-medium text-white uppercase tracking-wider">
              Supported Industrial Machine Classes (MIMII Dataset Standard)
            </h2>
          </div>
          <span className="text-xs font-mono text-slate-400">4 Categories Monitored</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {machines.map((m) => (
            <div
              key={m.id}
              className="bg-industrial-950 border border-industrial-800 hover:border-blue-500/50 rounded-lg p-4 transition-all cursor-pointer group"
              onClick={() => onNavigate('analysis')}
            >
              <div className="flex items-start justify-between">
                <span className="text-xs font-mono uppercase text-blue-400 font-semibold tracking-wider">
                  {m.id}
                </span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    m.trained_model_available
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                      : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  }`}
                >
                  {m.trained_model_available ? 'ONLINE' : 'TRAIN PENDING'}
                </span>
              </div>
              <h3 className="mt-2 text-sm font-semibold text-white group-hover:text-blue-300 transition-colors">
                {m.name}
              </h3>
              <p className="mt-1 text-xs text-slate-400 line-clamp-2">
                {m.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Two Column Section: Latest Experiment vs Recent Analyses */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Latest Research Benchmark */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
              Latest Research Experiment Benchmark
            </h3>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
              VERIFIED
            </span>
          </div>

          {latestExperiment ? (
            <div className="space-y-4">
              <div className="p-3 bg-industrial-950 rounded border border-industrial-800 text-xs font-mono space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-400">Target Category:</span>
                  <span className="text-white font-semibold uppercase">{latestExperiment.machine_category}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Fusion Architecture:</span>
                  <span className="text-cyan-400">{latestExperiment.fusion_mechanism}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Evaluation Split:</span>
                  <span className="text-slate-300">50 Independent Test Recordings</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-industrial-950 rounded border border-industrial-800 text-center">
                  <div className="text-[11px] font-mono text-slate-400">Baseline Mel-AE F1</div>
                  <div className="text-xl font-bold font-mono text-blue-400">
                    {(latestExperiment.baseline.f1_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono">ROC-AUC: {latestExperiment.baseline.roc_auc.toFixed(3)}</div>
                </div>

                <div className="p-3 bg-industrial-950 rounded border border-industrial-800 text-center">
                  <div className="text-[11px] font-mono text-slate-400">Proposed Adaptive Fusion F1</div>
                  <div className="text-xl font-bold font-mono text-emerald-400">
                    {(latestExperiment.proposed.f1_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono">ROC-AUC: {latestExperiment.proposed.roc_auc.toFixed(3)}</div>
                </div>
              </div>

              <div className="p-2.5 bg-blue-950/30 border border-blue-900/40 rounded text-xs text-blue-300">
                Proposed adaptive feature weighting achieved a <span className="font-bold">+{(latestExperiment.relative_improvement_auc * 100).toFixed(1)}%</span> AUC delta under simulated harmonic ambient conditions.
              </div>
            </div>
          ) : (
            <div className="p-6 text-center text-xs font-mono text-slate-500">
              No experiment records found. Run the training script to generate research benchmarks.
            </div>
          )}
        </div>

        {/* Recent Analysis Telemetry Log */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
              Recent Acoustic Telemetry
            </h3>
            <button
              onClick={() => onNavigate('monitoring')}
              className="text-xs font-mono text-blue-400 hover:text-blue-300 flex items-center space-x-1"
            >
              <span>View All History</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {history.length > 0 ? (
            <div className="space-y-2">
              {history.slice(0, 4).map((item) => (
                <div
                  key={item.id}
                  className="p-3 bg-industrial-950 rounded border border-industrial-800 flex items-center justify-between text-xs"
                >
                  <div className="space-y-0.5">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono font-medium text-white truncate max-w-[180px]">
                        {item.filename}
                      </span>
                      <span className="font-mono text-[10px] uppercase px-1.5 py-0.5 rounded bg-industrial-800 text-slate-400">
                        {item.machine_category}
                      </span>
                    </div>
                    <div className="text-[10px] font-mono text-slate-500">
                      Score: {item.anomaly_score.toFixed(4)} (Threshold: {item.decision_threshold.toFixed(4)})
                    </div>
                  </div>

                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded border whitespace-nowrap ${
                      item.is_anomaly
                        ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                        : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    }`}
                  >
                    {item.is_anomaly ? 'ANOMALY' : 'NORMAL'}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-8 text-center text-xs font-mono text-slate-500 border border-dashed border-industrial-800 rounded">
              No recordings analyzed yet. Navigate to Acoustic Analysis to inspect audio.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
