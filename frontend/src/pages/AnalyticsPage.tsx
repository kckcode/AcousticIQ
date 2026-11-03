import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell, 
  LineChart, 
  Line,
  CartesianGrid 
} from 'recharts';
import { BarChart3, PieChart as PieIcon, TrendingUp, Activity } from 'lucide-react';
import { AnalysisHistoryItem } from '../types';

interface AnalyticsPageProps {
  history: AnalysisHistoryItem[];
}

export const AnalyticsPage: React.FC<AnalyticsPageProps> = ({ history }) => {
  // Aggregate analyses by category
  const categories = ['valve', 'pump', 'fan', 'slider'];
  const categoryData = categories.map((cat) => {
    const subset = history.filter((h) => h.machine_category === cat);
    const normal = subset.filter((h) => !h.is_anomaly).length;
    const anomaly = subset.filter((h) => h.is_anomaly).length;
    return {
      name: cat.toUpperCase(),
      Normal: normal,
      Anomaly: anomaly,
      total: subset.length,
    };
  });

  // Aggregate normal vs anomaly totals
  const totalNormal = history.filter((h) => !h.is_anomaly).length;
  const totalAnomaly = history.filter((h) => h.is_anomaly).length;
  const pieData = [
    { name: 'Normal Operating Sound', value: totalNormal, color: '#10b981' },
    { name: 'Acoustic Anomaly Detected', value: totalAnomaly, color: '#f43f5e' },
  ].filter((d) => d.value > 0);

  // Score timeline (chronological order)
  const timelineData = [...history]
    .reverse()
    .map((h, idx) => ({
      index: idx + 1,
      score: Number(h.anomaly_score.toFixed(4)),
      threshold: Number(h.decision_threshold.toFixed(4)),
      category: h.machine_category.toUpperCase(),
      status: h.is_anomaly ? 'Anomaly' : 'Normal',
    }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-mono">
          Acoustic Intelligence & Fleet Analytics
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Statistical aggregations, anomaly distributions, and temporal trends derived from persisted telemetry.
        </p>
      </div>

      {history.length > 0 ? (
        <>
          {/* Top Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Category Breakdown Bar Chart (7 Cols) */}
            <div className="lg:col-span-7 bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-3">
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4 text-blue-400" />
                <h3 className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
                  Analyses by Industrial Machine Category
                </h3>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categoryData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} fontFamily="monospace" />
                    <YAxis stroke="#64748b" fontSize={11} fontFamily="monospace" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        borderRadius: '6px',
                        fontFamily: 'monospace',
                        fontSize: '11px',
                      }}
                    />
                    <Bar dataKey="Normal" fill="#10b981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Anomaly" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="flex items-center justify-center space-x-6 text-xs font-mono text-slate-400 pt-2">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-emerald-500 rounded" />
                  <span>Normal Sound Pattern</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-rose-500 rounded" />
                  <span>Acoustic Anomaly</span>
                </div>
              </div>
            </div>

            {/* Decision Ratio Donut Chart (5 Cols) */}
            <div className="lg:col-span-5 bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-3">
              <div className="flex items-center space-x-2">
                <PieIcon className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
                  Normal vs Anomaly Ratio
                </h3>
              </div>

              <div className="h-64 w-full flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={85}
                      paddingAngle={4}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        borderRadius: '6px',
                        fontFamily: 'monospace',
                        fontSize: '11px',
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="flex items-center justify-center space-x-4 text-xs font-mono text-slate-300">
                <span>Normal: <b className="text-emerald-400">{totalNormal}</b></span>
                <span>•</span>
                <span>Anomalies: <b className="text-rose-400">{totalAnomaly}</b></span>
              </div>
            </div>
          </div>

          {/* Anomaly Score Timeline Chart */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-amber-400" />
                <h3 className="text-xs font-mono font-medium text-slate-300 uppercase tracking-wider">
                  Reconstruction Error Score Trend Over Sequential Analyses
                </h3>
              </div>
              <span className="text-xs font-mono text-slate-500">{history.length} Datapoints</span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timelineData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="index" stroke="#64748b" fontSize={11} fontFamily="monospace" label={{ value: 'Sequential Recording #', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                  <YAxis stroke="#64748b" fontSize={11} fontFamily="monospace" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '6px',
                      fontFamily: 'monospace',
                      fontSize: '11px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="score"
                    name="Anomaly Score"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ fill: '#3b82f6', r: 3 }}
                  />
                  <Line
                    type="step"
                    dataKey="threshold"
                    name="Calibrated Threshold"
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      ) : (
        <div className="bg-industrial-900 border border-industrial-800 rounded-lg p-12 text-center text-xs font-mono text-slate-500 space-y-2">
          <Activity className="w-8 h-8 text-slate-600 mx-auto" />
          <div className="text-sm text-slate-400 font-semibold">No Analytics Telemetry Recorded Yet</div>
          <p>Analyze sound recordings or execute the training pipeline to populate fleet telemetry charts.</p>
        </div>
      )}
    </div>
  );
};
