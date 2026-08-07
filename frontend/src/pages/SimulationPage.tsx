import { lazy, Suspense, useEffect, useRef, useState } from "react"
import { useNavigate, useParams } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Progress } from "@/components/ui/progress"
import { CategoryAccordion } from "@/components/ParamForm/CategoryAccordion"
import { PathogenToggle } from "@/components/ParamForm/PathogenToggle"
import { RecentRunsPanel } from "@/components/RecentRunsPanel"
import { useParameters } from "@/hooks/useParameters"
import { useSubmitScenario, useRunDetail } from "@/hooks/useScenarioRun"
import { useRecentRuns } from "@/hooks/useRecentRuns"
import type { ScenarioFormValues } from "@/api/types"
import { ApiError } from "@/api/client"
import { Loader2 } from "lucide-react"

// ECharts (~500KB) is only needed once a run has actually completed - code
// split it out of the main bundle rather than paying for it on first paint.
const ResultsPanel = lazy(() => import("@/components/ResultsPanel").then((m) => ({ default: m.ResultsPanel })))

function buildInitialValues(data: ReturnType<typeof useParameters>["data"]): ScenarioFormValues {
  if (!data) return {}
  const values: ScenarioFormValues = { rota_enabled: true, campy_enabled: true }
  for (const cat of data.by_category) {
    for (const meta of cat.editable) {
      if (!meta.form_name) continue
      if (meta.ui_widget === "range-pair" && Array.isArray(meta.default)) {
        values[`${meta.form_name}_min`] = meta.default[0]
        values[`${meta.form_name}_max`] = meta.default[1]
      } else if (typeof meta.default === "number") {
        values[meta.form_name] = meta.default
      }
    }
  }
  return values
}

export default function SimulationPage() {
  const { data, isLoading, error } = useParameters()
  const [values, setValues] = useState<ScenarioFormValues>({})
  const initialized = useRef(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // The URL is the source of truth for which run is being viewed - makes a
  // past run bookmarkable/shareable ("/runs/:jobId") and keeps "just
  // submitted" and "clicked from history" the same code path (both just
  // navigate here), matching the plan's "GET /api/runs/{id} either way".
  const { jobId: routeJobId } = useParams<{ jobId?: string }>()
  const activeJobId = routeJobId ?? null
  const navigate = useNavigate()

  useEffect(() => {
    if (data && !initialized.current) {
      setValues(buildInitialValues(data))
      initialized.current = true
    }
  }, [data])

  const submitScenario = useSubmitScenario()
  const run = useRunDetail(activeJobId)
  const recentRuns = useRecentRuns()

  const handleChange = (formName: string, value: number | boolean) => {
    setValues((prev) => ({ ...prev, [formName]: value }))
  }

  const handleSubmit = async () => {
    setSubmitError(null)
    if (!values.rota_enabled && !values.campy_enabled) {
      setSubmitError("At least one pathogen must be enabled.")
      return
    }
    try {
      const result = await submitScenario.mutateAsync(values)
      navigate(`/runs/${result.job_id}`)
    } catch (e) {
      setSubmitError(e instanceof ApiError ? e.message : "Failed to submit scenario.")
    }
  }

  // Clicking a history row only previews its results - it must never
  // silently clobber parameters the user is mid-editing. Loading those
  // parameters into the form is a separate, explicit action.
  const handleViewRun = (jobId: string) => navigate(`/runs/${jobId}`)
  const handleLoadParams = (configForm: ScenarioFormValues) => setValues(configForm)

  const isRunning = run.data?.status === "queued" || run.data?.status === "running"

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Simulation</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Configure a scenario, run it, and review results on one page. Your settings stay put as you
          re-run this session — but finished runs are kept only temporarily (the last 50, ~2 hours),
          so download anything you want to keep.
        </p>
      </header>

      <RecentRunsPanel
        runs={recentRuns.data?.runs ?? []}
        activeJobId={activeJobId}
        onView={handleViewRun}
        onLoadParams={handleLoadParams}
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Pathogens</CardTitle>
        </CardHeader>
        <CardContent>
          <PathogenToggle values={values} onChange={handleChange} />
        </CardContent>
      </Card>

      {isLoading && (
        <div className="space-y-3">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )}

      {error && (
        <p className="text-sm text-destructive">Failed to load parameters. Try reloading the page.</p>
      )}

      {data && (
        <Card>
          <CardContent>
            <CategoryAccordion categories={data.by_category} values={values} onChange={handleChange} />
          </CardContent>
        </Card>
      )}

      <div className="flex items-center gap-3">
        <Button onClick={handleSubmit} disabled={submitScenario.isPending || isRunning} size="lg">
          {isRunning && <Loader2 className="size-4 animate-spin" />}
          {isRunning ? "Running…" : "Run simulation"}
        </Button>
        {submitError && <p className="text-sm text-destructive">{submitError}</p>}
      </div>

      {activeJobId && run.data && (
        <>
          {run.data.status !== "done" && run.data.status !== "error" && (
            <Card>
              <CardContent className="pt-6 space-y-3">
                {run.data.status === "queued" ? (
                  <p className="text-sm text-muted-foreground">Queued…</p>
                ) : (
                  <>
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">Running simulation…</span>
                      <span className="text-muted-foreground tabular-nums">
                        Day {run.data.progress_day} of {run.data.progress_total}
                      </span>
                    </div>
                    <Progress
                      value={
                        run.data.progress_total > 0
                          ? (run.data.progress_day / run.data.progress_total) * 100
                          : 0
                      }
                    />
                  </>
                )}
              </CardContent>
            </Card>
          )}
          {run.data.status === "error" && (
            <Card>
              <CardContent className="pt-6">
                <p className="text-sm text-destructive">Simulation failed: {run.data.error}</p>
              </CardContent>
            </Card>
          )}
          {run.data.status === "done" && run.data.result && (
            <Suspense fallback={<Skeleton className="h-80 w-full" />}>
              <ResultsPanel result={run.data.result} />
            </Suspense>
          )}
        </>
      )}
    </div>
  )
}
