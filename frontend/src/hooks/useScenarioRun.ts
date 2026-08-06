import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api/client"
import type { ScenarioFormValues } from "@/api/types"

export function useSubmitScenario() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (values: ScenarioFormValues) => api.submitScenario(values),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] })
    },
  })
}

/** Polls GET /api/runs/{id} every 2s while the run is queued/running -
 * the direct replacement for the old HTMX `hx-trigger="load delay:2s"`
 * poll. Also used by Phase 3's history view to load a past run's detail
 * (a finished run's status stops the interval immediately, so reusing this
 * hook there costs nothing extra). */
export function useRunDetail(jobId: string | null) {
  return useQuery({
    queryKey: ["runs", jobId],
    queryFn: () => api.getRun(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === "queued" || status === "running" ? 2000 : false
    },
  })
}
