import { useMemo, useRef } from "react"
import { Download } from "lucide-react"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { MetricGrid } from "@/components/MetricCard"
import { PrevalenceLineChart } from "@/components/charts/PrevalenceLineChart"
import { IllnessDaysAreaChart } from "@/components/charts/IllnessDaysAreaChart"
import { SingleSeriesLineChart } from "@/components/charts/SingleSeriesLineChart"
import { CampyRouteAreaChart } from "@/components/charts/CampyRouteAreaChart"
import { SpatialHeatmapChart } from "@/components/charts/SpatialHeatmapChart"
import { useBasemap } from "@/hooks/useBasemap"
import type { SimResultBundle } from "@/api/types"
import { downloadCsv, downloadJson, downloadChartPng, type EChartsLike } from "@/lib/download"

const PATHOGEN_LABEL: Record<string, string> = { rota: "Rotavirus", campy: "Campylobacter" }

function buildMetrics(result: SimResultBundle): { label: string; value: string }[] {
  const metrics = [
    { label: "Under-5 agents", value: String(result.n_u5) },
    { label: "Runtime", value: `${result.runtime_seconds.toFixed(1)}s` },
    { label: "Ever infected", value: `${(result.proportion_infected_at_least_once * 100).toFixed(1)}%` },
  ]
  for (const name of result.pathogen_names) {
    const peak = result.summary_metrics[`${name}_peak_u5_prevalence`]
    if (typeof peak === "number") {
      metrics.push({ label: `${PATHOGEN_LABEL[name] ?? name} peak U5 prevalence`, value: `${(peak * 100).toFixed(1)}%` })
    }
  }
  const careSeeking = result.summary_metrics.episode_care_seeking_rate
  if (typeof careSeeking === "number") {
    metrics.push({ label: "Episode care-seeking rate", value: `${(careSeeking * 100).toFixed(1)}%` })
  }
  return metrics
}

function ChartDownloadButton({ onClick }: { onClick: () => void }) {
  return (
    <Button
      variant="ghost"
      size="icon"
      className="size-7 text-muted-foreground hover:text-foreground"
      aria-label="Download chart as PNG"
      title="Download PNG"
      onClick={onClick}
    >
      <Download className="size-4" />
    </Button>
  )
}

export function ResultsPanel({ result }: { result: SimResultBundle }) {
  const basemap = useBasemap()

  // Live ECharts instances, surfaced by each chart via onReady, so the download
  // buttons in the card headers can export a PNG.
  const charts = useRef<Record<string, EChartsLike | undefined>>({})
  const register = useMemo(() => {
    const make = (key: string) => (chart: EChartsLike) => {
      charts.current[key] = chart
    }
    return {
      prevalence: make("prevalence"),
      illness: make("illness"),
      campy: make("campy"),
      care: make("care"),
    }
  }, [])
  // The spatial map isn't an ECharts chart; it exposes its own composite exporter.
  const spatialExportRef = useRef<(() => void) | null>(null)

  // New parents seeking care each day = day-over-day change in the running
  // total. A parent increments the counter at most once per day, so the diff is
  // exactly that day's number of care-seekers (day 0 = its own value).
  const cumCare = result.cumulative_care_seeking_events
  const dailyCareSeeking = cumCare.map((v, i) => (i === 0 ? v : v - cumCare[i - 1]))

  const campyRoutes = result.campy_daily_infections_by_route
  const showCampyRoutes =
    result.pathogen_names.includes("campy") &&
    !!campyRoutes &&
    Object.values(campyRoutes).some((arr) => Array.isArray(arr) && arr.length > 0)

  return (
    <div className="space-y-6">
      <MetricGrid metrics={buildMetrics(result)} />

      <Card>
        <CardContent className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-medium">Download this run</p>
            <p className="text-xs text-muted-foreground">
              Runs are temporary — export the data or charts to keep them.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => downloadCsv(result, "springs-abm-run.csv")}>
              <Download className="size-4" /> Data (CSV)
            </Button>
            <Button variant="outline" size="sm" onClick={() => downloadJson(result, "springs-abm-run.json")}>
              <Download className="size-4" /> Data (JSON)
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card data-testid="chart-card-prevalence">
        <CardHeader>
          <CardTitle className="text-base">Prevalence</CardTitle>
          <CardAction>
            <ChartDownloadButton onClick={() => downloadChartPng(charts.current.prevalence, "prevalence.png")} />
          </CardAction>
        </CardHeader>
        <CardContent>
          <PrevalenceLineChart
            days={result.days}
            u5Prevalence={result.u5_prevalence}
            allAgesPrevalence={result.all_ages_prevalence}
            pathogenNames={result.pathogen_names}
            onReady={register.prevalence}
          />
        </CardContent>
      </Card>

      <Card data-testid="chart-card-illness-days">
        <CardHeader>
          <CardTitle className="text-base">Cumulative under-5 illness-days</CardTitle>
          <CardAction>
            <ChartDownloadButton onClick={() => downloadChartPng(charts.current.illness, "cumulative-illness-days.png")} />
          </CardAction>
        </CardHeader>
        <CardContent>
          <IllnessDaysAreaChart
            days={result.days}
            cumulativeU5IllnessDays={result.cumulative_u5_illness_days}
            pathogenNames={result.pathogen_names}
            onReady={register.illness}
          />
        </CardContent>
      </Card>

      {showCampyRoutes && (
        <Card data-testid="chart-card-campy-routes">
          <CardHeader>
            <CardTitle className="text-base">Campylobacter infections by route</CardTitle>
            <CardAction>
              <ChartDownloadButton onClick={() => downloadChartPng(charts.current.campy, "campylobacter-routes.png")} />
            </CardAction>
          </CardHeader>
          <CardContent>
            <CampyRouteAreaChart days={result.days} infectionsByRoute={campyRoutes!} onReady={register.campy} />
          </CardContent>
        </Card>
      )}

      <Card data-testid="chart-card-care-seeking">
        <CardHeader>
          <CardTitle className="text-base">Parents seeking care per day</CardTitle>
          <CardAction>
            <ChartDownloadButton onClick={() => downloadChartPng(charts.current.care, "parents-seeking-care.png")} />
          </CardAction>
        </CardHeader>
        <CardContent>
          <SingleSeriesLineChart
            days={result.days}
            values={dailyCareSeeking}
            yAxisLabel="Parents/day"
            onReady={register.care}
          />
        </CardContent>
      </Card>

      <Card data-testid="chart-card-spatial">
        <CardHeader>
          <CardTitle className="text-base">Spatial spread</CardTitle>
          <CardAction>
            <ChartDownloadButton onClick={() => spatialExportRef.current?.()} />
          </CardAction>
        </CardHeader>
        <CardContent>
          {basemap.data ? (
            <SpatialHeatmapChart
              basemap={basemap.data}
              spatialDailyGrids={result.spatial_daily_grids}
              gridSize={result.spatial_grid_size}
              staticLayers={result.static_layers}
              exportRef={spatialExportRef}
            />
          ) : (
            <p className="text-sm text-muted-foreground">Loading basemap…</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
