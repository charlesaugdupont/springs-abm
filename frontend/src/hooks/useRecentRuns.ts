import { useQuery } from "@tanstack/react-query"
import { api } from "@/api/client"

/** GET /api/runs - lightweight summaries only (see webapp/jobs.py's
 * _SUMMARY_FIELDS), never the full per-day/spatial payload. Submitting a
 * new scenario invalidates this query (see useSubmitScenario), and it
 * additionally polls while anything is queued/running so the list's status
 * badges update without a manual refresh. */
export function useRecentRuns(limit = 20) {
  return useQuery({
    queryKey: ["runs", "list", limit],
    queryFn: () => api.listRuns(limit),
    refetchInterval: (query) => {
      const runs = query.state.data?.runs ?? []
      const hasActive = runs.some((r) => r.status === "queued" || r.status === "running")
      return hasActive ? 2000 : false
    },
  })
}
