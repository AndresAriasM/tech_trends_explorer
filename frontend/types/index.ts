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
  /** Error absoluto en publicaciones. El R² sobre un acumulado es siempre alto. */
  rmse: number;
  L_stderr: number | null;
  x0_stderr: number | null;
  k_stderr: number | null;
  t_L: number | null;
  t_k: number | null;
  L_significant: boolean;
  k_significant: boolean;
  /** L_stderr / L. Por encima de 0.20 el techo no es utilizable. */
  L_rel_error: number | null;
  /**
   * Gobierna si se puede reportar saturación. Más estricto que L_significant:
   * el t-test de L contrasta contra cero y casi siempre pasa.
   */
  L_identifiable: boolean;
  /** Intervalo de confianza al 95 % de x0 — para x0 el valor t no informa. */
  x0_ci95: [number, number] | null;
  /** x0 fuera del rango observado: la inflexión está extrapolada. */
  x0_extrapolated: boolean;
}

export type SCurvePhaseName =
  | "Emergente"
  | "Crecimiento"
  | "Aceleración"
  | "Maduración"
  | "Saturación"
  | "Pre-inflexión"
  | "Post-inflexión"
  | "Sin datos";

export interface SCurvePhase {
  phase: SCurvePhaseName;
  color: string;
  description: string;
  /** null cuando L no es identificable: no hay porcentaje que reportar. */
  saturation_pct: number | null;
  /** La fase se apoya sólo en el eje temporal; trátala como no concluyente. */
  provisional: boolean;
  basis: "saturation+temporal" | "temporal" | "none";
  before_inflection?: boolean;
}

export interface SCurveResult {
  query: string;
  total_publications: number;
  year_from: number;
  year_to: number;
  yearly_data: SCurveYearlyData[];
  sigmoid_params: SCurveSigmoidParams | null;
  sigmoid_curve: Array<{ year: number; value: number }>;
  /** Velocidad de adopción: derivada analítica de la sigmoide. */
  growth_curve: Array<{ year: number; rate: number }>;
  /** Residuos por año: revelan estructura que el R² oculta. */
  residuals: Array<{ year: number; residual: number }>;
  /** Cinco años de predicción. Dibujar punteado: no es ajuste. */
  projections: Array<{ year: number; value: number }>;
  /** Variación interanual del flujo anual, no del acumulado. */
  annual_growth_rate: Array<{ year: number; pct: number | null }>;
  peak_growth_year?: number;
  peak_growth_rate?: number;
  phase: SCurvePhase;
  /** Lo que Scopus dice que existe, frente a lo que se pudo descargar. */
  total_reported: number | null;
  total_retrieved: number;
  entries_without_date: number;
  /** Scopus cortó resultados en algún periodo: la curva los subestima. */
  truncated: boolean;
  truncated_windows: Array<{ from: number; to: number; total: number | null }>;
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
