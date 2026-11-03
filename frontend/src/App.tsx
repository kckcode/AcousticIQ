import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { OverviewPage } from './pages/OverviewPage';
import { AnalysisPage } from './pages/AnalysisPage';
import { MonitoringPage } from './pages/MonitoringPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { ModelLabPage } from './pages/ModelLabPage';
import { ExperimentPage } from './pages/ExperimentPage';
import { SettingsPage } from './pages/SettingsPage';
import { fetchHealth, fetchMachines, fetchHistory, fetchMetrics } from './services/api';
import { HealthResponse, MachineCategory, AnalysisHistoryItem, ExperimentSummary } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [machines, setMachines] = useState<MachineCategory[]>([]);
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
  const [historyTotal, setHistoryTotal] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [latestExperiment, setLatestExperiment] = useState<ExperimentSummary | undefined>(undefined);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean>(false);

  const loadInitialData = async () => {
    try {
      const h = await fetchHealth();
      setHealth(h);
      setIsBackendHealthy(true);
    } catch {
      setIsBackendHealthy(false);
    }

    try {
      const m = await fetchMachines();
      setMachines(m);
    } catch (e) {
      console.error(e);
    }

    try {
      const hist = await fetchHistory(currentPage, 20, selectedCategory);
      setHistory(hist.results);
      setHistoryTotal(hist.total);
    } catch (e) {
      console.error(e);
    }

    try {
      const met = await fetchMetrics();
      if (met.latest_experiment) {
        setLatestExperiment(met.latest_experiment);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, [currentPage, selectedCategory]);

  const handleAnalysisComplete = () => {
    // Reload history to include the new analysis
    fetchHistory(currentPage, 20, selectedCategory).then((h) => {
      setHistory(h.results);
      setHistoryTotal(h.total);
    });
  };

  return (
    <div className="min-h-screen bg-industrial-950 flex flex-col">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendHealthy={isBackendHealthy}
        activeDevice={health?.active_device || 'CPU'}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'overview' && (
          <OverviewPage
            history={history}
            machines={machines}
            latestExperiment={latestExperiment}
            onNavigate={setActiveTab}
            onSelectSample={() => setActiveTab('analysis')}
          />
        )}

        {activeTab === 'analysis' && (
          <AnalysisPage onAnalysisComplete={handleAnalysisComplete} />
        )}

        {activeTab === 'monitoring' && (
          <MonitoringPage
            history={history}
            total={historyTotal}
            currentPage={currentPage}
            pageSize={20}
            onPageChange={setCurrentPage}
            selectedCategory={selectedCategory}
            onCategoryChange={(cat) => {
              setSelectedCategory(cat);
              setCurrentPage(1);
            }}
          />
        )}

        {activeTab === 'analytics' && <AnalyticsPage history={history} />}

        {activeTab === 'lab' && <ModelLabPage />}

        {activeTab === 'experiments' && <ExperimentPage />}

        {activeTab === 'settings' && <SettingsPage health={health} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-industrial-800 bg-industrial-900 py-4 text-xs font-mono text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            AcousticIQ — Acoustic Deep Learning Based Predictive Maintenance System
          </div>
          <div>
            GitHub: <a href="https://github.com/kckcode/AcousticIQ" target="_blank" rel="noreferrer" className="text-blue-400 hover:underline">kckcode/AcousticIQ</a> • MIT License
          </div>
        </div>
      </footer>
    </div>
  );
};
