// Mirrors webapp/parameter_registry.py's ParamMeta and the JSON shapes
// returned by webapp/routers/{parameters,scenario,runs,auth}.py. This is a
// thin, hand-written mirror rather than a codegen artifact - the registry
// itself (not this file) remains the single source of truth; if a field is
// added there, the API just starts including it and this type widens too.

export type EvidenceTier = "literature" | "calibrated" | "assumption" | "structural"
export type UiWidget = "slider" | "number+randomize-button" | "range-pair"

export interface ParamMeta {
  path: string
  label: string
  category: string
  evidence_tier: EvidenceTier
  rationale: string
  editable: boolean
  unit: string | null
  ui_min: number | null
  ui_max: number | null
  ui_widget: UiWidget
  ui_step: number | null
  is_integer: boolean
  // Present only when editable=true:
  form_name?: string
  default?: number | [number, number]
}

export interface ParametersByCategory {
  category: string
  editable: ParamMeta[]
  readonly: ParamMeta[]
}

export interface ParametersByEvidenceTier {
  tier: EvidenceTier
  label: string
  params: ParamMeta[]
}

export interface ParametersResponse {
  category_order: string[]
  by_category: ParametersByCategory[]
  by_evidence_tier: ParametersByEvidenceTier[]
}

// A scenario submission body: dynamically keyed by each editable param's
// form_name (see ParamMeta.form_name), e.g. { rota_recovery_rate: 0.2, ... }.
// Deliberately loose rather than a fixed interface - the set of fields is
// entirely driven by ParametersResponse at runtime (see requirement #4:
// zero hardcoded field lists in the frontend).
export type ScenarioFormValues = Record<string, number | boolean>

export type JobStatus = "queued" | "running" | "done" | "error"

export interface SimResultBundle {
  config_snapshot: Record<string, unknown>
  pathogen_names: string[]
  days: number[]
  u5_prevalence: Record<string, number[]>
  all_ages_prevalence: Record<string, number[]>
  cumulative_u5_illness_days: Record<string, number[]>
  mean_household_wealth: number[]
  cumulative_care_seeking_events: number[]
  spatial_grid_size: number
  spatial_daily_grids: number[][][]
  summary_metrics: Record<string, unknown>
  proportion_infected_at_least_once: number
  n_u5: number
  runtime_seconds: number
}

export interface RunDetail {
  job_id: string
  status: JobStatus
  created_at: number
  config_form: ScenarioFormValues
  error: string | null
  progress_day: number
  progress_total: number
  result: SimResultBundle | null
}

export interface RunSummary {
  job_id: string
  status: JobStatus
  created_at: number
  config_form: ScenarioFormValues
  summary: Partial<SimResultBundle> | null
  error: string | null
}

export interface RunListResponse {
  runs: RunSummary[]
}

export interface SubmitRunResponse {
  job_id: string
  status: JobStatus
}

export interface SessionResponse {
  authenticated: boolean
}
