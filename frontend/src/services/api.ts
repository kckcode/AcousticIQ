import {
  HealthResponse,
  MachineCategory,
  AnalysisResponse,
  PaginatedAnalysisResponse,
  ExperimentSummary,
  ModelCatalogItem,
  PreloadedSample,
} from '../types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchMachines(): Promise<MachineCategory[]> {
  const res = await fetch(`${API_BASE}/machines`);
  if (!res.ok) throw new Error(`Failed to fetch machines: ${res.statusText}`);
  const data = await res.json();
  return data.machines;
}

export async function analyzeAudioFile(
  file: File | Blob,
  filename: string,
  machineCategory: string,
  modelType: string = 'proposed'
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file, filename);
  formData.append('machine_category', machineCategory);
  formData.append('model_type', modelType);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Audio analysis failed.');
  }
  return res.json();
}

export async function fetchHistory(
  page: number = 1,
  pageSize: number = 20,
  category?: string
): Promise<PaginatedAnalysisResponse> {
  let url = `${API_BASE}/results?page=${page}&page_size=${pageSize}`;
  if (category && category !== 'all') {
    url += `&machine_category=${encodeURIComponent(category)}`;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load history: ${res.statusText}`);
  return res.json();
}

export async function fetchMetrics(): Promise<{ status: string; latest_experiment?: ExperimentSummary }> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelCatalogItem[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  const data = await res.json();
  return data.models;
}

export async function fetchExperiments(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error(`Failed to fetch experiments: ${res.statusText}`);
  const data = await res.json();
  return data.experiments;
}

export async function fetchExperimentDetails(id: string): Promise<ExperimentSummary> {
  const res = await fetch(`${API_BASE}/experiments/${id}`);
  if (!res.ok) throw new Error(`Failed to load experiment: ${res.statusText}`);
  return res.json();
}

export async function fetchSamples(): Promise<PreloadedSample[]> {
  const res = await fetch(`${API_BASE}/audio/samples`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.samples;
}
