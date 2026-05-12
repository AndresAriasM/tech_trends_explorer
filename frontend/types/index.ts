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

export interface NewsItem {
  title: string;
  link: string;
  snippet: string;
  source: string;
  date: string;
  year: number;
  sentiment: number;
  country: string | null;
  keywords: string[];
}

export interface CountryStat {
  country: string;
  mentions: number;
  sentiment_avg: number;
}

export interface KeywordStat {
  keyword: string;
  count: number;
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
  news_items?: NewsItem[];
  country_stats?: CountryStat[];
  keyword_stats?: KeywordStat[];
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
// S-Curve types
// ============================================================

export interface SCurveYearlyData {
  year: number;
  annual: number;
  cumulative: number;
}

export interface SCurveSigmoidParams {
  L: number;
  x0: number;
  k: number;
  r2: number;
}

export interface SCurvePhase {
  phase: "Emergente" | "Crecimiento" | "Aceleración" | "Maduración" | "Saturación" | "Sin datos";
  color: string;
  description: string;
  saturation_pct: number;
}

export interface SCurveResult {
  query: string;
  total_publications: number;
  year_from: number;
  year_to: number;
  yearly_data: SCurveYearlyData[];
  sigmoid_params: SCurveSigmoidParams | null;
  sigmoid_curve: Array<{ year: number; value: number }>;
  phase: SCurvePhase;
  inflection_year: number | null;
  saturation_level: number | null;
  r2: number | null;
  analyzed_at: string;
}

export interface SCurveProgress {
  status: "running" | "completed" | "error";
  progress: number;
  message: string;
  result?: SCurveResult;
}

export interface SCurveRequest {
  query: string;
  api_key: string;
  year_from?: number;
  year_to?: number;
  max_results?: number;
}

// ============================================================
// Workshop / Taller Prospectivo LACCEI 2032
// ============================================================

export type RatingValue = 1 | 2 | 3 | 4;
export type RatingLabel = "Malo" | "Regular" | "Bueno" | "Excelente";

export interface WorkshopVariable {
  id:          number;
  name:        string;
  category:    string;
  description: string;
}

export interface WorkshopSession {
  session_id:     string;
  code:           string;
  title:          string;
  description:    string;
  status:         "active" | "closed";
  created_at:     string;
  closed_at:      string | null;
  ai_votes_ready: boolean;
  gforms_url:     string;
}

export interface VoteItem {
  variable_id:    number;
  rating_current: RatingValue;
  rating_2032:    RatingValue;
  justification?: string;
}

export interface ExpertVote extends VoteItem {
  vote_id:       string;
  session_id:    string;
  variable_name: string;
  expert_id:     string;
  expert_name:   string;
  is_ai_vote:    boolean;
  ai_evidence?:  string;
  timestamp:     string;
}

export interface VariableSynthesis {
  variable_id:      number;
  variable_name:    string;
  category:         string;
  n_responses:      number;
  avg_current:      number | null;
  avg_2032:         number | null;
  std_current:      number | null;
  std_2032:         number | null;
  label_current:    RatingLabel | null;
  label_2032:       RatingLabel | null;
  consensus_score:  number | null;
  distribution_current: Record<RatingLabel, number>;
  distribution_2032:    Record<RatingLabel, number>;
  ai_vote:              ExpertVote | null;
  ai_divergence_current: number | null;
  ai_divergence_2032:    number | null;
}

export interface WorkshopSynthesis {
  session_id:  string;
  n_experts:   number;
  n_variables: number;
  variables:   VariableSynthesis[];
  divergences: VariableSynthesis[];
  overall: {
    avg_current:   number | null;
    avg_2032:      number | null;
    label_current: RatingLabel | null;
    label_2032:    RatingLabel | null;
  };
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
