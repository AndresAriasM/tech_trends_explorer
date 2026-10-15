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
  /**
   * Autocorrelación de los residuos (0–4; 2 = sin estructura). Muy por debajo
   * de 2 delata sesgo sistemático por tramos que el R² no ve.
   */
  durbin_watson: number | null;
  autocorrelation: SCurveAutocorrelation;
}

// ── Ajuste multi-modelo ───────────────────────────────────────

export type SCurveAutocorrelation = "none" | "moderate" | "severe" | "unknown";

export type SCurveModelKey =
  | "logistic"
  | "gompertz"
  | "weibull"
  | "hill"
  | "chapman_richards";

export interface SCurveModelFit {
  key: SCurveModelKey;
  label: string;
  converged: boolean;
  /** Nombres según el modelo: (L, x0, k), (L, lambda, beta), (L, Kh, n)… */
  params: Record<string, number> | null;
  param_stderr: Record<string, number | null> | null;
  r2: number | null;
  /** R² penalizado por los 3 parámetros del modelo. */
  r2_adj: number | null;
  /** Error estándar de la estimación, en unidades de la serie. */
  se: number | null;
  rmse: number | null;
  durbin_watson: number | null;
  autocorrelation: SCurveAutocorrelation;
  aicc: number | null;
  inflection_year: number | null;
  /** La inflexión ya está dentro de los datos observados. */
  inflection_passed: boolean;
  saturation_L: number | null;
  saturation_pct: number | null;
  /** Parámetros que quedaron pegados a su cota: no son estimaciones. */
  params_at_bound: string[];
  /** L topado contra el límite del ajuste: el techo no es estimable. */
  L_at_bound: boolean;
  residuals: Array<{ year: number; residual: number }>;
  is_best: boolean;
}

export interface SCurveModelComparison {
  models: SCurveModelFit[];
  best: SCurveModelKey | null;
  best_label: string | null;
  best_model: SCurveModelFit | null;
  n_points: number;
  /** Motivo por el que no se comparó (menos de 5 años). */
  note: string | null;
  /** Dispersión de la inflexión entre modelos: la incertidumbre real. */
  inflection_range: [number, number] | null;
  saturation_range: [number, number] | null;
  /** Aviso de autocorrelación o de techo no estimable. */
  warning: string | null;
  converged_count: number;
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

export type SCurveSource = "scopus" | "epo" | "patents" | "google_patents" | "local";

export interface SCurveResult {
  query: string;
  /** De dónde salió la serie: condiciona cómo se etiqueta y se compara. */
  source?: SCurveSource;
  source_label?: string;
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
  /** Máximo del flujo anual observado (no del ajuste). */
  peak_annual?: number;
  peak_annual_year?: number | null;
  /**
   * Los cinco modelos de 3 parámetros comparados. `sigmoid_params` sigue
   * siendo el logístico: esto se añade al lado, no lo sustituye.
   */
  model_comparison?: SCurveModelComparison;
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

  // ── Sólo en la ruta EPO OPS ──────────────────────────────
  /** Peticiones gastadas contra OPS (una por año). */
  epo_requests?: number;
  /** Años que OPS no pudo servir: la serie los tiene como hueco. */
  epo_failed_years?: number[];
  /** Cabecera X-Throttling-Control de la última respuesta. */
  epo_throttle?: string | null;
  /** OPS recortó el presupuesto de búsquedas durante la descarga. */
  epo_throttled?: boolean;
  /** La consulta CQL que se envió realmente. */
  epo_cql?: string;
}

// ── Comparación papers ↔ patentes ─────────────────────────────

export interface SCurveComparisonSide {
  kind: "papers" | "patents";
  label: string;
  query: string | null;
  records: number | null;
  year_from: number | null;
  year_to: number | null;
  peak_annual: number | null;
  peak_annual_year: number | null;
  phase: string | null;
  phase_color: string | null;
  saturation_pct: number | null;
  best_model: string | null;
  inflection_year: number | null;
  inflection_passed: boolean;
  inflection_range: [number, number] | null;
  durbin_watson: number | null;
  warning: string | null;
}

export interface SCurveMaturityMatrix {
  available: boolean;
  /** Posición en [-1,1]: 0 es la inflexión, positivo = ya la pasó. */
  x: number | null;
  y: number | null;
  reason?: string;
  quadrant?: "consolidada" | "ciencia_por_delante" | "industria_por_delante" | "incipiente";
  label?: string;
  description?: string;
  color?: string;
  /** Alguna inflexión está proyectada: la posición es una previsión. */
  provisional?: boolean;
  axes?: {
    x: { label: string; source: string | null };
    y: { label: string; source: string | null };
  };
}

export interface SCurveComparison {
  matrix: SCurveMaturityMatrix;
  papers: SCurveComparisonSide;
  patents: SCurveComparisonSide;
  /** Positivo = las patentes se inflexionan antes que los papers. */
  lag_years: number | null;
  direction: "patents_lead" | "papers_lead" | "simultaneous" | "unknown";
  interpretation: string;
  headline: string;
  /** Razones para no tomarse el desfase al pie de la letra. */
  caveats: string[];
  compared_at: string;
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

/**
 * Patentes vía EPO OPS. Sin api_key: las credenciales de OPS son del
 * servidor y no viajan desde el navegador, a diferencia de las de Scopus.
 */
export interface SCurvePatentsRequest {
  query: string;
  year_from?: number;
  year_to?: number;
  /** "epo" (por defecto) o "google" (Google Patents vía SerpAPI). */
  source?: "epo" | "google";
}

export interface EPOStatus {
  configured: boolean;
  authenticated: boolean;
  search_ok?: boolean;
  probe?: { cql: string; total: number };
  throttle?: string | null;
  /** Búsquedas por minuto que OPS permite ahora mismo. */
  search_quota_per_min?: number | null;
  message: string;
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
