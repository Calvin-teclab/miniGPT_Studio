// Training parameter configuration
export interface ParamConfig {
  key: string;
  label: string;
  description: string;  // tooltip description
  type: 'select' | 'number' | 'slider' | 'toggle' | 'text';
  options?: { value: string | number; label: string }[];
  default: string | number | boolean;
  min?: number;
  max?: number;
  step?: number;
  group: 'model' | 'training' | 'data' | 'optimizer';
}

// Pipeline step
export interface PipelineStep {
  id: string;
  name: string;
  description: string;
  detailedHelp: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'skipped';
  progress?: number;
  stage: string; // backend stage name
}

// Training metrics
export interface TrainingMetrics {
  step: number;
  totalSteps: number;
  loss: number;
  lr: number;
  tokensPerSec: number;
  eta: string;
  elapsed: string;
  valLoss?: number;
  bpb?: number;
  gradNorm?: number;
}

// Layer activation data for heatmap
export interface LayerActivation {
  step: number;
  layers: {
    name: string;
    meanActivation: number;
    maxActivation: number;
    gradMean: number;
    gradMax: number;
  }[];
}

// Generated sample during training
export interface TrainingSample {
  step: number;
  prompt: string;
  generated: string;
  timestamp: string;
}

// Loss history point
export interface LossPoint {
  step: number;
  trainLoss: number;
  valLoss?: number;
  lr?: number;
}

// Checkpoint info
export interface Checkpoint {
  path: string;
  checkpointId?: string;
  name: string;
  modelName?: string;
  displayName?: string;
  depth: number;
  step: number;
  nEmbd: number;
  isSft: boolean;
  windowStrategy?: string;
  dataDomain?: string;
  source?: string;
  date?: number;
}

// Evaluation result
export interface EvalResult {
  benchmark: string;
  score: number;
  total: number;
  accuracy: number;
  details?: Record<string, number>;
}

// External model evaluation
export interface ExternalEvalConfig {
  provider: 'openai' | 'anthropic' | 'custom';
  apiKey: string;
  endpoint?: string;
  model: string;
}

export interface ExternalEvalResult {
  prompt: string;
  localResponse: string;
  externalResponse: string;
  externalScore: number;
  externalFeedback: string;
}

// Chat message
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
}

// Chat settings
export interface ChatSettings {
  temperature: number;
  topK: number;
  repetitionPenalty: number;
  maxTokens: number;
}

// System status
export interface SystemStatus {
  dataReady: boolean;
  tokenizerReady: boolean;
  modelReady: boolean;
  sftReady: boolean;
  chatLoaded: boolean;
  running: boolean;
  currentStage?: string;
}
