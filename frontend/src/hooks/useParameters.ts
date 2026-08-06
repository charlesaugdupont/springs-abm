import { useQuery } from "@tanstack/react-query"
import { api } from "@/api/client"

export function useParameters() {
  return useQuery({
    queryKey: ["parameters"],
    queryFn: api.getParameters,
    // The registry is fixed for the lifetime of a deployed build - no
    // point re-fetching it on window focus/remount.
    staleTime: Infinity,
  })
}
