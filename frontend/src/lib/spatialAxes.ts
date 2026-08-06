// Client-side port of the old webapp/routers/runs.py::_spatial_axes() -
// pure display-coordinate arithmetic with no server-side state dependency,
// so it moved to the frontend during the API conversion (see that
// function's removal note in webapp/routers/runs.py). Maps the downsampled
// spatial grid's row/col indices onto the basemap's geographic extent, so
// the heatmap lines up with the background image (row=y-bin index (0=south),
// col=x-bin index (0=west) - see webapp/simulation_runner.py::_spatial_grid).

export interface BasemapMeta {
  minx: number
  miny: number
  maxx: number
  maxy: number
  provider: string
  pixels: number
}

export interface SpatialAxes {
  x: number[]
  y: number[]
}

export function spatialAxes(basemap: BasemapMeta, gridSize: number): SpatialAxes {
  const { minx, maxx, miny, maxy } = basemap
  const x: number[] = []
  const y: number[] = []
  for (let k = 0; k < gridSize; k++) {
    x.push(minx + ((k + 0.5) / gridSize) * (maxx - minx))
    y.push(miny + ((k + 0.5) / gridSize) * (maxy - miny))
  }
  return { x, y }
}

export async function fetchBasemapMeta(): Promise<BasemapMeta> {
  const resp = await fetch("/static/img/akuse_basemap.json")
  if (!resp.ok) throw new Error("Failed to load basemap metadata")
  return resp.json()
}
