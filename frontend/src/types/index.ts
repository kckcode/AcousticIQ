export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  database_connected: boolean;
  cuda_available: boolean;
  active_device: string;
  loaded_models: Record<string, string>;
}

export interface MachineCategory {
  id: string;
  name: string;
  description: string;
  trained_model_available: boolean;
  sample_count?: number;
}

export interface SpectrogramData {
  frequencies: number[];
  times: number[];
  magnitudes: number[][];
  mel_spectrogram?: number[][];
  inverse_mel_spectrogram?: number[][];
  waveform_downsampled?: number[];
  waveform_times?: number[];
}

export interface AnalysisResponse {
  analysis_id: string;
  timestamp: string;
  filename: string;
  machine_category: string;
  sample_rate: number;
  duration_seconds: number;
  model_version: string;
  model_architecture: string;
  anomaly_score: number;
  decision_threshold: number;
  is_anomaly: boolean;
  classification_decision: string;
  processing_time_ms: number;
  confidence_rating: string;
  feature_config: {
    n_mels: number;
    n_fft: number;
    hop_length: number;
    adaptive_gate_alpha?: number;
  };
  warnings: string[];
  limitations_note: string;
  visualizations?: SpectrogramData;
  is_synthetic_fixture: boolean;
}

export interface AnalysisHistoryItem {
  id: string;
  created_at: string;
  filename: string;
  machine_category: string;
  duration_seconds: number;
  anomaly_score: number;
  decision_threshold: number;
  is_anomaly: boolean;
  classification_decision: string;
  processing_time_ms: number;
  model_version: string;
  is_synthetic: boolean;
}

export interface PaginatedAnalysisResponse {
  total: number;
  page: number;
  page_size: number;
  results: AnalysisHistoryItem[];
}

export interface MetricSummary {
  model_name: string;
  roc_auc: number;
  p_auc: number;
  f1_score: number;
  precision: number;
  recall: number;
  false_positive_rate: number;
  decision_threshold: number;
  confusion_matrix: number[][];
  total_test_samples: number;
  normal_test_samples: number;
  anomaly_test_samples: number;
  mean_inference_latency_ms: number;
  fpr_curve: number[];
  tpr_curve: number[];
}

export interface ExperimentSummary {
  experiment_id: string;
  timestamp: string;
  machine_category: string;
  is_synthetic: boolean;
  dataset_subset: string;
  fusion_mechanism: string;
  epochs: number;
  batch_size: number;
  baseline: MetricSummary;
  proposed: MetricSummary;
  relative_improvement_auc: number;
  relative_improvement_pauc: number;
  figures: Record<string, string>;
  status: string;
  notes: string;
}

export interface ModelCatalogItem {
  machine_category: string;
  model_type: string;
  architecture: string;
  checkpoint_exists: boolean;
  checkpoint_size_mb: number;
  threshold?: number;
  parameter_count?: number;
  training_duration_seconds?: number;
  final_train_loss?: number;
}

export interface PreloadedSample {
  filename: string;
  machine_category: string;
  label: string;
  is_synthetic: boolean;
  url: string;
}
