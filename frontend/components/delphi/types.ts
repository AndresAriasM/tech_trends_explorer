/**
 * Tipos del módulo Delphi — espejo de `backend/core/delphi.py`.
 *
 * Viven aquí y no en `types/index.ts` porque el módulo se entrega como una
 * unidad; si se estabiliza, se sube al índice global.
 *
 * La distinción que atraviesa todo el fichero es la §6 de LOGICA_DELPHI.md:
 * la **población de agregado** (quienes contestaron la ronda) y la
 * **población de cambio** (la intersección entre dos rondas) son dos cosas
 * distintas y nunca comparten un número. El campo `poblacion` lo dice en
 * cada objeto.
 */

export type Poblacion = "agregado" | "agregado_restringido" | "cambio";

export type Clasificacion = "PRIORITARIA" | "EN DISCUSIÓN" | "DESCARTADA";

export type Cuadrante =
  | "CERRADA"
  | "CONVERGIENDO"
  | "DISENSO_PERSISTENTE"
  | "INESTABLE";

export interface DelphiStudy {
  id: string;
  title: string;
  description: string;
  status: "draft" | "active" | "closed";
  horizon_year: number | null;
  scale_type: string;
  dichotomy_threshold: number;
  max_rounds: number;
  stopping_rule: Record<string, number>;
  created_at: string;
  updated_at: string;
  n_rounds?: number;
  n_variables?: number;
  n_invited?: number;
  n_registered?: number;
  n_questions?: number;
}

export interface DelphiVariable {
  id: string;
  study_id: string;
  code: string;
  name: string;
  description: string | null;
  family: string | null;
  order_index: number;
}

export interface DelphiRound {
  id: string;
  study_id: string;
  round_number: number;
  title: string;
  instructions: string;
  status: "draft" | "open" | "closed";
  opens_at: string | null;
  closes_at: string | null;
  n_questions?: number;
  n_submitted?: number;
}

export interface DelphiParticipant {
  id: string;
  email: string;
  display_name: string | null;
  status: "invited" | "registered" | "active" | "dropped";
  kc: number | null;
  ka: number | null;
  k: number | null;
  education: string | null;
  affiliation: string | null;
  invitation_token?: string;
}

/** Estadística de una variable en una ronda — población de AGREGADO. */
export interface VariableRoundStats {
  variable_id: string;
  variable_name?: string;
  family?: string | null;
  round_number: number | null;
  poblacion: Poblacion;
  n_population: number;
  n: number;
  n_abstentions: number;
  n_no_response: number;
  response_rate: number | null;
  abstention_rate: number | null;
  mean: number | null;
  median: number | null;
  std: number | null;
  q1: number | null;
  q3: number | null;
  iqr: number | null;
  pct_above_threshold: number | null;
  pct_above_threshold_weighted: number | null;
  n_above_threshold: number;
  consensus: number | null;
  consensus_iqr: number | null;
  mean_k: number | null;
  distribution: Record<string, number>;
  classification: Clasificacion | null;
  threshold: number;
}

export interface McNemar {
  b: number;
  c: number;
  n_discordant: number;
  statistic: number | null;
  p_value: number;
  p_value_chi2: number | null;
  p_value_exact: number;
  method: string;
  continuity_correction: boolean;
  significant_05: boolean;
  interpretation: string;
}

export interface Wilcoxon {
  n_pairs: number;
  n_nonzero: number;
  n_ties: number;
  statistic: number | null;
  p_value: number | null;
  method: string | null;
  significant_05: boolean;
}

/** Comparación entre dos rondas — población de CAMBIO, y sólo ella. */
export interface VariableTransition {
  variable_id: string;
  variable_name?: string;
  family?: string | null;
  round_from: number;
  round_to: number;
  poblacion: "cambio";
  n_change_population: number;
  matrix: {
    a_mantiene_alto: number;
    b_bajo: number;
    c_subio: number;
    d_mantiene_bajo: number;
    pareados: number;
    no_comparables: number;
  };
  states: Record<string, number>;
  estabilidad: number | null;
  direccion_neta: number;
  mcnemar: McNemar;
  median_from: number | null;
  median_to: number | null;
  median_shift: number | null;
  iqr_from: number | null;
  iqr_to: number | null;
  iqr_reduction: number | null;
  wilcoxon: Wilcoxon;
}

/** §6.2 — el agregado completo frente al restringido a quienes siguieron. */
export interface AttritionCheck {
  variable_id: string;
  variable_name?: string;
  round_from: number;
  round_to: number;
  total: VariableRoundStats;
  restricted: VariableRoundStats;
  delta_pct_above_threshold: number | null;
  delta_median: number | null;
  delta_mean: number | null;
  attrition_non_random: boolean;
  note: string;
}

export interface PanelRoundRow {
  round_number: number;
  poblacion_agregado: number;
  response_rate: number | null;
  poblacion_cambio: number | null;
  attrition_rate: number | null;
  retention_rate: number | null;
  new_panelists: number | null;
}

export interface PanelStats {
  n_invited: number | null;
  n_panelists_seen: number;
  rounds: PanelRoundRow[];
  mean_k_all: number | null;
  mean_k_active: number | null;
  n_below_k_min: number | null;
  attrition_summary: {
    n_variables_checked: number;
    n_variables_non_random: number;
    verdict: string;
  };
  quadrants: Record<string, number>;
  stopping_rule: {
    stability_min: number;
    iqr_max: number;
    n_closed: number;
    n_open: number;
    next_round_variables: string[];
  };
}

export interface QuadrantItem {
  variable_id: string;
  variable_name: string;
  consensus: number | null;
  consensus_iqr: number | null;
  stability: number | null;
  iqr: number | null;
  median: number | null;
  n: number;
  round_number: number | null;
  direccion_neta: number | null;
  mcnemar_p: number | null;
  mcnemar_significant: boolean | null;
  quadrant: Cuadrante | null;
  action: string | null;
  stopping: {
    closed: boolean;
    by_stability: boolean;
    by_iqr: boolean;
    stability_min: number;
    iqr_max: number;
    reason: string;
  };
}

export interface VariableMap {
  axes: {
    x: { label: string; threshold: number };
    y: { label: string; threshold: number };
  };
  variables: QuadrantItem[];
  sin_clasificar: QuadrantItem[];
  counts: Record<string, number>;
  next_round_variables: string[];
}

export interface SpearmanPair {
  variable_a: string;
  variable_b: string;
  label_a: string;
  label_b: string;
  rho: number;
  p_value: number | null;
  n: number;
}

export interface SpearmanResult {
  round_number: number | null;
  variable_ids: string[];
  labels: string[];
  matrix: (number | null)[][];
  top_pairs: SpearmanPair[];
  n_pairs_computed: number;
}

export interface RoundReport {
  round_number: number;
  poblacion: "agregado";
  n_respondents: number;
  n_population: number;
  response_rate: number | null;
  n_variables: number;
  variables: VariableRoundStats[];
  by_family: { family: string; variables: VariableRoundStats[] }[];
  n_prioritarias: number;
  n_en_discusion: number;
  n_descartadas: number;
}

export interface StudyReport {
  config: {
    dichotomy_threshold: number;
    scale_min: number;
    scale_max: number;
    stability_min: number;
    iqr_max: number;
    consensus_min: number;
    k_min: number | null;
  };
  n_rounds: number;
  n_variables: number;
  rounds: RoundReport[];
  between: VariableTransition[];
  attrition_check: AttritionCheck[];
  panel: PanelStats;
  map: VariableMap;
  spearman: SpearmanResult[];
  study: {
    id: string;
    title: string;
    status: string;
    horizon_year: number | null;
    scale_type: string;
  };
}

export interface SchemaCheck {
  ready: boolean;
  missing: Record<string, string[]>;
  migration: string;
}
