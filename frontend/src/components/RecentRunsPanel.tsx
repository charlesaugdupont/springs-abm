import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { JobStatus, RunSummary, ScenarioFormValues } from "@/api/types"

const STATUS_STYLES: Record<JobStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  done: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  error: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
}

function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}

interface RecentRunsPanelProps {
  runs: RunSummary[]
  activeJobId: string | null
  onView: (jobId: string) => void
  onLoadParams: (configForm: ScenarioFormValues) => void
}

export function RecentRunsPanel({ runs, activeJobId, onView, onLoadParams }: RecentRunsPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Recent runs</CardTitle>
      </CardHeader>
      <CardContent>
        {runs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No runs yet - configure a scenario and run it.</p>
        ) : (
          <ul className="space-y-2">
            {runs.map((run) => {
              const pathogens = run.summary?.pathogen_names as string[] | undefined
              return (
                <li
                  key={run.job_id}
                  className={cn(
                    "flex items-center justify-between gap-3 rounded-lg border p-2.5",
                    run.job_id === activeJobId && "border-primary bg-muted/40",
                  )}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className={STATUS_STYLES[run.status]}>
                        {run.status}
                      </Badge>
                      <span className="text-xs text-muted-foreground">{formatTimestamp(run.created_at)}</span>
                    </div>
                    {pathogens && (
                      <p className="text-xs text-muted-foreground mt-1 truncate">{pathogens.join(", ")}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <Button size="sm" variant="outline" onClick={() => onView(run.job_id)}>
                      View
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => onLoadParams(run.config_form)}>
                      Load parameters
                    </Button>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}
