import React, { useState, useEffect } from 'react';
import { 
  FlaskConical, 
  BarChart2, 
  TrendingUp, 
  Clock, 
  FileCheck, 
  Layers,
  Image as ImageIcon,
  CheckCircle2
} from 'lucide-react';
import { fetchExperiments, fetchExperimentDetails } from '../services/api';
import { ExperimentSummary } from '../types';

export const ExperimentPage: React.FC = () => {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [currentExp, setCurrentExp] = useState<ExperimentSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function init() {
      try {
        const list = await fetchExperiments();
        setExperiments(list);
        if (list.length > 0) {
          const firstId = list[0].experiment_id;
          setSelectedExpId(firstId);
          const details = await fetchExperimentDetails(firstId);
          setCurrentExp(details);
        }
      } catch (e) {
        console.error('Failed to load experiments', e);
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  const handleSelectExp = async (id: string) => {
    setSelectedExpId(id);
    try {
      const details = await fetchExperimentDetails(id);
      setCurrentExp(details);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-mono">
            Experimental Evaluation & Scientific Benchmarks
          </h1>
          <p className="text-xs sm:text-sm text-slate-400">
            Reproducible empirical comparison between Baseline and Proposed Adaptive Fusion on identical held-out test splits.
          </p>
        </div>

        {/* Experiment Selector */}
        {experiments.length > 0 && (
          <div className="flex items-center space-x-2 bg-industrial-900 border border-industrial-800 rounded-lg px-3 py-1.5 text-xs font-mono">
            <FlaskConical className="w-4 h-4 text-cyan-400" />
            <select
              value={selectedExpId || ''}
              onChange={(e) => handleSelectExp(e.target.value)}
              className="bg-transparent text-slate-200 outline-none cursor-pointer"
            >
              {experiments.map((exp) => (
                <option key={exp.experiment_id} value={exp.experiment_id} className="bg-industrial-900">
                  {exp.machine_category.toUpperCase()} — {exp.experiment_id}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {currentExp ? (
        <div className="space-y-6">
          {/* Scientific Metric Comparison Table */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <BarChart2 className="w-5 h-5 text-blue-400" />
                <h2 className="text-sm font-mono font-medium text-white uppercase tracking-wider">
                  Diagnostic Performance Summary ({currentExp.machine_category.toUpperCase()})
                </h2>
              </div>
              <div className="text-xs font-mono text-slate-400">
                Protocol: <span className="text-slate-200">Zero-leakage Split (Threshold tuned on Val)</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-industrial-950 border-b border-industrial-800 text-slate-400 uppercase text-[10px]">
                  <tr>
                    <th className="py-2.5 px-3">Evaluation Metric</th>
                    <th className="py-2.5 px-3">Baseline Mel-AE</th>
                    <th className="py-2.5 px-3 text-emerald-400 font-bold">Proposed Adaptive Fusion</th>
                    <th className="py-2.5 px-3">Delta (Impact)</th>
                    <th className="py-2.5 px-3">Research Significance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-industrial-800">
                  <tr>
                    <td className="py-2.5 px-3 text-white font-semibold">ROC-AUC</td>
                    <td className="py-2.5 px-3 text-slate-300">{currentExp.baseline.roc_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">{currentExp.proposed.roc_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-200">
                      {(currentExp.proposed.roc_auc - currentExp.baseline.roc_auc) >= 0 ? '+' : ''}
                      {(currentExp.proposed.roc_auc - currentExp.baseline.roc_auc).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">Primary discriminative power</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-white font-semibold">Partial ROC-AUC (pAUC @ FPR 0.1)</td>
                    <td className="py-2.5 px-3 text-slate-300">{currentExp.baseline.p_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">{currentExp.proposed.p_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-200">
                      {(currentExp.proposed.p_auc - currentExp.baseline.p_auc) >= 0 ? '+' : ''}
                      {(currentExp.proposed.p_auc - currentExp.baseline.p_auc).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">Strict DCASE industrial low false-alarm standard</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-white font-semibold">F1-Score</td>
                    <td className="py-2.5 px-3 text-slate-300">{currentExp.baseline.f1_score.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">{currentExp.proposed.f1_score.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-semibold text-emerald-300">
                      {(currentExp.proposed.f1_score - currentExp.baseline.f1_score) >= 0 ? '+' : ''}
                      {(currentExp.proposed.f1_score - currentExp.baseline.f1_score).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">Harmonic balance of precision and recall</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-white font-semibold">Precision</td>
                    <td className="py-2.5 px-3 text-slate-300">{currentExp.baseline.precision.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">{currentExp.proposed.precision.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-semibold text-emerald-300">
                      {(currentExp.proposed.precision - currentExp.baseline.precision) >= 0 ? '+' : ''}
                      {(currentExp.proposed.precision - currentExp.baseline.precision).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">Minimization of false positive alarms</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-white font-semibold">Recall</td>
                    <td className="py-2.5 px-3 text-slate-300">{currentExp.baseline.recall.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">{currentExp.proposed.recall.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-200">
                      {(currentExp.proposed.recall - currentExp.baseline.recall) >= 0 ? '+' : ''}
                      {(currentExp.proposed.recall - currentExp.baseline.recall).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">True anomaly capture sensitivity</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-white font-semibold">Inference Latency</td>
                    <td className="py-2.5 px-3 text-slate-300">{currentExp.baseline.mean_inference_latency_ms.toFixed(2)} ms</td>
                    <td className="py-2.5 px-3 text-slate-300">{currentExp.proposed.mean_inference_latency_ms.toFixed(2)} ms</td>
                    <td className="py-2.5 px-3 text-slate-400">
                      +{(currentExp.proposed.mean_inference_latency_ms - currentExp.baseline.mean_inference_latency_ms).toFixed(2)} ms
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">Dual-branch computation overhead</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Generated Research Plots Gallery */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
            <div className="flex items-center space-x-2">
              <ImageIcon className="w-5 h-5 text-cyan-400" />
              <h2 className="text-sm font-mono font-medium text-white uppercase tracking-wider">
                Publication-Ready Empirical Research Figures
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Figure 1: ROC Curve */}
              <div className="bg-industrial-950 border border-industrial-800 rounded p-3 space-y-2">
                <span className="text-[11px] font-mono text-slate-300 font-semibold block">
                  1. Receiver Operating Characteristic (ROC)
                </span>
                <div className="bg-industrial-900 rounded overflow-hidden flex items-center justify-center p-1">
                  <img
                    src={`/api/figures/${currentExp.experiment_id}_roc_curve.png`}
                    alt="ROC Curve"
                    className="w-full h-auto rounded"
                    onError={(e) => {
                      (e.target as any).style.display = 'none';
                    }}
                  />
                </div>
                <p className="text-[10px] text-slate-500 font-mono">
                  Comparative ROC trajectories highlighting pAUC cut-off at FPR = 0.1.
                </p>
              </div>

              {/* Figure 2: Metric Comparison */}
              <div className="bg-industrial-950 border border-industrial-800 rounded p-3 space-y-2">
                <span className="text-[11px] font-mono text-slate-300 font-semibold block">
                  2. Diagnostic Metric Summary
                </span>
                <div className="bg-industrial-900 rounded overflow-hidden flex items-center justify-center p-1">
                  <img
                    src={`/api/figures/${currentExp.experiment_id}_metrics_comparison.png`}
                    alt="Metrics Comparison"
                    className="w-full h-auto rounded"
                    onError={(e) => {
                      (e.target as any).style.display = 'none';
                    }}
                  />
                </div>
                <p className="text-[10px] text-slate-500 font-mono">
                  Bar breakdown of ROC-AUC, pAUC, F1-score, Precision, and Recall.
                </p>
              </div>

              {/* Figure 3: Loss Convergence */}
              <div className="bg-industrial-950 border border-industrial-800 rounded p-3 space-y-2">
                <span className="text-[11px] font-mono text-slate-300 font-semibold block">
                  3. Training Loss Convergence
                </span>
                <div className="bg-industrial-900 rounded overflow-hidden flex items-center justify-center p-1">
                  <img
                    src={`/api/figures/${currentExp.experiment_id}_loss_curves.png`}
                    alt="Loss Curves"
                    className="w-full h-auto rounded"
                    onError={(e) => {
                      (e.target as any).style.display = 'none';
                    }}
                  />
                </div>
                <p className="text-[10px] text-slate-500 font-mono">
                  MSE loss descent over training epochs on normal operating audio.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-12 text-center text-xs font-mono text-slate-500">
          No experiments found in ml/experiments/. Run the training script to evaluate models.
        </div>
      )}
    </div>
  );
};
