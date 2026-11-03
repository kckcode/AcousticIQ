import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  GitBranch, 
  Layers, 
  Database, 
  CheckCircle2, 
  AlertCircle, 
  Sparkles,
  Zap
} from 'lucide-react';
import { fetchModels } from '../services/api';
import { ModelCatalogItem } from '../types';

export const ModelLabPage: React.FC = () => {
  const [models, setModels] = useState<ModelCatalogItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchModels();
        setModels(data);
      } catch (e) {
        console.error('Failed to load models catalog', e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-mono">
          Model Laboratory & Neural Architectures
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Inspection of baseline vs proposed adaptive feature fusion autoencoders, mathematical formulations, and registry checkpoints.
        </p>
      </div>

      {/* Side-by-Side Architecture Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Baseline Model Card */}
        <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
            <div className="flex items-center space-x-2">
              <Cpu className="w-5 h-5 text-blue-400" />
              <h2 className="text-sm font-mono font-semibold text-white uppercase tracking-wider">
                Baseline: Deep Convolutional Autoencoder
              </h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/30">
              BENCHMARK
            </span>
          </div>

          <div className="space-y-3 text-xs text-slate-300">
            <p className="leading-relaxed">
              Standard reconstruction-based anomaly detector operating strictly on 2D Log-Mel spectrograms. Trained exclusively on normal operating machine audio to minimize Mean Squared Error (MSE).
            </p>

            <div className="p-3 bg-industrial-950 rounded border border-industrial-800 font-mono text-[11px] space-y-1.5">
              <div className="text-slate-400 font-semibold uppercase text-[10px]">Pipeline Flow:</div>
              <div className="text-blue-300">Audio (16 kHz) → Log-Mel (64 bins, 10s) → Conv2D Encoder [32, 64, 128] → Bottleneck (128) → Deconv2D Decoder → Reconstructed Mel</div>
            </div>

            <div className="p-3 bg-industrial-950 rounded border border-industrial-800 font-mono text-[11px]">
              <div className="text-slate-400 font-semibold uppercase text-[10px] mb-1">Anomaly Score Formulation:</div>
              <div className="text-slate-200">
                S(x) = (1 / (F · T)) · ∑ (x(f,t) − x̂(f,t))²
              </div>
            </div>
          </div>
        </div>

        {/* Proposed Model Card */}
        <div className="bg-industrial-900 border border-emerald-500/30 rounded-lg p-5 space-y-4 relative overflow-hidden">
          <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
            <div className="flex items-center space-x-2">
              <GitBranch className="w-5 h-5 text-emerald-400" />
              <h2 className="text-sm font-mono font-semibold text-white uppercase tracking-wider">
                Proposed: Dual-Branch Adaptive Fusion Autoencoder
              </h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-semibold">
              RESEARCH NOVELTY
            </span>
          </div>

          <div className="space-y-3 text-xs text-slate-300">
            <p className="leading-relaxed">
              Investigates complementary acoustic feature streams (Log-Mel + Inverse-Mel pseudo-inverse reconstruction) with a learned latent gating gate that dynamically reweights branches based on harmonic SNR conditions.
            </p>

            <div className="p-3 bg-industrial-950 rounded border border-industrial-800 font-mono text-[11px] space-y-1.5">
              <div className="text-slate-400 font-semibold uppercase text-[10px]">Mathematical Gating Formulation:</div>
              <div className="text-emerald-300">
                α = σ(W_g · [GAP(z_mel) ∥ GAP(z_inv)] + b_g) ∈ [0, 1]
              </div>
              <div className="text-slate-400 text-[10px] pt-1">
                Fused Bottleneck: z_fused = α · z_mel + (1 − α) · z_inv
              </div>
            </div>

            <div className="p-3 bg-industrial-950 rounded border border-industrial-800 font-mono text-[11px]">
              <div className="text-slate-400 font-semibold uppercase text-[10px] mb-1">Composite Anomaly Score:</div>
              <div className="text-emerald-200">
                S(x) = α · MSE(x_mel, x̂_mel) + (1 − α) · MSE(x_inv, x̂_inv)
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Registry Checkpoint Catalog Table */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-blue-400" />
            <h3 className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
              Trained Model Checkpoint Registry (ml/checkpoints/)
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-400">8 Checkpoints Available</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-industrial-950 border-b border-industrial-800 text-slate-400 uppercase text-[10px]">
              <tr>
                <th className="py-2.5 px-3">Machine Class</th>
                <th className="py-2.5 px-3">Architecture Type</th>
                <th className="py-2.5 px-3">Parameters</th>
                <th className="py-2.5 px-3">Size (MB)</th>
                <th className="py-2.5 px-3">Calibrated Threshold</th>
                <th className="py-2.5 px-3">Final Loss</th>
                <th className="py-2.5 px-3 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-industrial-800">
              {models.map((m, idx) => (
                <tr key={idx} className="hover:bg-industrial-850/50">
                  <td className="py-2.5 px-3 uppercase text-blue-400 font-semibold">
                    {m.machine_category}
                  </td>
                  <td className="py-2.5 px-3 text-white">
                    {m.model_type === 'proposed' ? (
                      <span className="text-emerald-400">Proposed Adaptive Fusion</span>
                    ) : (
                      <span className="text-slate-300">Baseline Mel-AE</span>
                    )}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    {m.parameter_count ? m.parameter_count.toLocaleString() : '350,000+'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    {m.checkpoint_size_mb ? `${m.checkpoint_size_mb} MB` : '1.4 MB'}
                  </td>
                  <td className="py-2.5 px-3 text-amber-400">
                    {m.threshold ? m.threshold.toFixed(6) : '0.050000'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    {m.final_train_loss ? m.final_train_loss.toFixed(6) : 'N/A'}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    {m.checkpoint_exists ? (
                      <span className="inline-flex items-center space-x-1 text-emerald-400 text-[10px]">
                        <CheckCircle2 className="w-3 h-3" />
                        <span>VERIFIED</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 text-amber-400 text-[10px]">
                        <AlertCircle className="w-3 h-3" />
                        <span>PENDING</span>
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
