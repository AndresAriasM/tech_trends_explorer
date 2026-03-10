// ============================================================
// Types — mirrors the data models from master/src/hype_cycle_storage.py
// ============================================================

export type HypePhase =
  | "Innovation Trigger"
  | "Peak of Inflated Expectations"
  | "Trough of Disillusionment"
  | "Slope of Enlightenment"
  | "Plateau of Productivity"
  | "Unknown";

export interface SearchTerm {
  value: string;
  operator: "AND" | "OR" | "NOT";
  exact_match: boolean;
}

export interface HypeCycleMetrics {
  phase: HypePhase;
  confidence: number;
  total_mentions: number;
  peak_mentions: number;
  latest_year: number;
  sentiment_avg: number;
  sentiment_trend: number;
  hype_cycle_position_x: number;
  hype_cycle_position_y: number;
  time_to_plateau: string;
  innovation_trigger_year?: number;
  peak_year?: number;
  trough_year?: number;
  inflection_points?: Record<string, number>;
}

export interface YearlyStat {
  year: number;
  mentions: number;
  sentiment: number;
  growth_rate: number | null;
}

export interface HypeCycleQuery {
  query_id: string;
  category_id: string;
  search_query: string;
  search_terms: SearchTerm[];
  execution_date: string;
  api_usage: Record<string, number>;
  hype_metrics: HypeCycleMetrics;
  yearly_stats: YearlyStat[];
  search_parameters: {
    min_year: number;
    sources_filter: string[];
    max_results: number;
  };
  technology_name?: string;
  technology_description?: string;
  notes?: string;
  version: string;
  processing_time?: number;
}

export interface Category {
  category_id: string;
  name: string;
  description: string;
  created_at: string;
  parent_category_id?: string;
  technologies: TechnologyEntry[];
}

export interface TechnologyEntry {
  tech_id: string;
  name: string;
  description?: string;
  phase?: HypePhase;
}

// ============================================================
// API request/response types
// ============================================================

export interface AnalyzeRequest {
  search_terms: SearchTerm[];
  technology_name?: string;
  technology_description?: string;
  category_id: string;
  search_parameters: {
    min_year: number;
    sources_filter: string[];
    max_results: number;
  };
  notes?: string;
  auto_save: boolean;
}

export interface AnalyzeResponse {
  task_id: string;
  status: "queued" | "running" | "completed" | "failed";
  message: string;
}

export interface TaskProgress {
  task_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number; // 0-100
  message: string;
  result?: HypeCycleQuery;
  error?: string;
}
