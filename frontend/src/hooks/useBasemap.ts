import { useQuery } from "@tanstack/react-query"
import { fetchBasemapMeta } from "@/lib/spatialAxes"

export function useBasemap() {
  return useQuery({
    queryKey: ["basemap"],
    queryFn: fetchBasemapMeta,
    staleTime: Infinity,
  })
}
