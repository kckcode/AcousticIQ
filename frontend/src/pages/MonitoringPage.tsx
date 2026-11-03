import React, { useState } from 'react';
import { 
  History, 
  Filter, 
  Download, 
  ChevronLeft, 
  ChevronRight, 
  Eye, 
  CheckCircle2, 
  AlertTriangle,
  X
} from 'lucide-react';
import { AnalysisHistoryItem } from '../types';

interface MonitoringPageProps {
  history: AnalysisHistoryItem[];
  total: number;
  currentPage: number;
  pageSize: number;
  onPageChange: (newPage: number) => void;
  selectedCategory: string;
  onCategoryChange: (cat: string) => void;
}

export const MonitoringPage: React.FC<MonitoringPageProps> = ({
  history,
  total,
  currentPage,
  pageSize,
  onPageChange,
  selectedCategory,
  onCategoryChange,
}) => {
  const [selectedRecord, setSelectedRecord] = useState<AnalysisHistoryItem | null>(null);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const exportCSV = () => {
    if (history.length === 0) return;
    const headers = [
      'ID',
      'Timestamp',
      'Filename',
      'Machine Category',
      'Duration (s)',
      'Anomaly Score',
      'Threshold',
      'Is Anomaly',
      'Decision',
      'Latency (ms)',
      'Model Version',
      'Synthetic Fixture',
    ];

    const rows = history.map((item) => [
      item.id,
      item.created_at,
      item.filename,
      item.machine_category,
      item.duration_seconds,
      item.anomaly_score,
      item.decision_threshold,
      item.is_anomaly,
      `"${item.classification_decision}"`,
      item.processing_time_ms,
      item.model_version,
      item.is_synthetic,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `acousticiq_telemetry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight font-mono">
            Acoustic Telemetry & Analysis History
          </h1>
          <p className="text-xs sm:text-sm text-slate-400">
            Persistent log of all acoustic recordings evaluated on the platform.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {/* Category Filter */}
          <div className="flex items-center space-x-1.5 bg-industrial-900 border border-industrial-800 rounded-lg px-2.5 py-1.5 text-xs font-mono">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={selectedCategory}
              onChange={(e) => onCategoryChange(e.target.value)}
              className="bg-transparent text-slate-200 outline-none cursor-pointer"
            >
              <option value="all" className="bg-industrial-900">All Categories</option>
              <option value="valve" className="bg-industrial-900">Valves</option>
              <option value="pump" className="bg-industrial-900">Pumps</option>
              <option value="fan" className="bg-industrial-900">Fans</option>
              <option value="slider" className="bg-industrial-900">Sliders</option>
            </select>
          </div>

          <button
            onClick={exportCSV}
            disabled={history.length === 0}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-industrial-850 hover:bg-industrial-800 text-slate-200 border border-industrial-700 text-xs font-mono transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-industrial-900 border border-industrial-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-industrial-950 border-b border-industrial-800 text-slate-400 font-mono uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Filename</th>
                <th className="py-3 px-4">Machine</th>
                <th className="py-3 px-4">Anomaly Score</th>
                <th className="py-3 px-4">Threshold</th>
                <th className="py-3 px-4">Decision</th>
                <th className="py-3 px-4">Latency</th>
                <th className="py-3 px-4 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-industrial-800 font-mono">
              {history.length > 0 ? (
                history.map((row) => (
                  <tr key={row.id} className="hover:bg-industrial-850/60 transition-colors">
                    <td className="py-3 px-4 text-slate-400 whitespace-nowrap">
                      {new Date(row.created_at).toLocaleString([], {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}
                    </td>
                    <td className="py-3 px-4 text-white font-medium max-w-[200px] truncate" title={row.filename}>
                      {row.filename}
                    </td>
                    <td className="py-3 px-4 uppercase text-blue-400 font-semibold">
                      {row.machine_category}
                    </td>
                    <td className="py-3 px-4 text-slate-200 font-bold">
                      {row.anomaly_score.toFixed(5)}
                    </td>
                    <td className="py-3 px-4 text-amber-400">
                      {row.decision_threshold.toFixed(5)}
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] border ${
                          row.is_anomaly
                            ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                            : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                        }`}
                      >
                        {row.is_anomaly ? (
                          <>
                            <AlertTriangle className="w-3 h-3" />
                            <span>ANOMALY</span>
                          </>
                        ) : (
                          <>
                            <CheckCircle2 className="w-3 h-3" />
                            <span>NORMAL</span>
                          </>
                        )}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400">
                      {row.processing_time_ms.toFixed(1)} ms
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => setSelectedRecord(row)}
                        className="p-1 rounded bg-industrial-800 hover:bg-industrial-700 text-slate-300 transition-colors"
                        title="View Full Diagnostics"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-500 font-mono">
                    No historical analysis telemetry records found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="p-3 bg-industrial-950 border-t border-industrial-800 flex items-center justify-between text-xs font-mono text-slate-400">
          <div>
            Showing {history.length} of {total} records
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage <= 1}
              className="p-1.5 rounded bg-industrial-900 border border-industrial-800 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-industrial-800"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span>
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages}
              className="p-1.5 rounded bg-industrial-900 border border-industrial-800 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-industrial-800"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-industrial-900 border border-industrial-750 rounded-xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
              <div className="flex items-center space-x-2">
                <History className="w-5 h-5 text-blue-400" />
                <h3 className="font-mono font-semibold text-white text-sm">
                  Telemetry Record Details
                </h3>
              </div>
              <button
                onClick={() => setSelectedRecord(null)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Analysis UUID:</span>
                <span className="text-white truncate max-w-[240px]">{selectedRecord.id}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Source Recording:</span>
                <span className="text-white truncate max-w-[240px]">{selectedRecord.filename}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Machine Class:</span>
                <span className="text-blue-400 uppercase font-semibold">{selectedRecord.machine_category}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Anomaly Reconstruction Score:</span>
                <span className="text-white font-bold">{selectedRecord.anomaly_score.toFixed(6)}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Decision Threshold:</span>
                <span className="text-amber-400">{selectedRecord.decision_threshold.toFixed(6)}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Classification Outcome:</span>
                <span className={selectedRecord.is_anomaly ? 'text-rose-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                  {selectedRecord.classification_decision}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Execution Duration:</span>
                <span className="text-slate-200">{selectedRecord.processing_time_ms.toFixed(1)} ms</span>
              </div>
              <div className="flex justify-between py-1 border-b border-industrial-850">
                <span className="text-slate-400">Model Version:</span>
                <span className="text-slate-300">{selectedRecord.model_version}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Synthetic Fixture:</span>
                <span className="text-slate-300">{selectedRecord.is_synthetic ? 'Yes (Calibration Test)' : 'No (Field Audio)'}</span>
              </div>
            </div>

            <div className="pt-2">
              <button
                onClick={() => setSelectedRecord(null)}
                className="w-full py-2 bg-industrial-800 hover:bg-industrial-700 text-white rounded text-xs font-mono font-medium"
              >
                CLOSE
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
