/* Renders the results-page Plotly charts from the JSON embedded by
 * runs/_results.html. Called both on the initial full page load
 * (DOMContentLoaded) and after every HTMX poll swap (htmx:afterSwap) -
 * HTMX-swapped-in <script> tags don't reliably execute via innerHTML
 * insertion, so chart rendering is triggered from here instead of an
 * inline <script> inside the swapped fragment. Safe to call repeatedly:
 * no-ops if the results data block isn't present, or already rendered. */
window.renderResultsCharts = function () {
  const dataEl = document.getElementById("run-data");
  if (!dataEl || dataEl.dataset.rendered === "1") return;
  dataEl.dataset.rendered = "1";

  const data = JSON.parse(dataEl.textContent);
  const label = (name) => (name === "rota" ? "Rotavirus" : "Campylobacter");
  const color = { rota: "#0f7a72", campy: "#d97706" };
  const commonLayout = { margin: { t: 10, r: 20, l: 55, b: 40 }, xaxis: { title: "Simulation day" } };
  const commonConfig = { responsive: true, displaylogo: false };

  const prevalenceTraces = [];
  data.pathogen_names.forEach((name) => {
    prevalenceTraces.push({
      x: data.days, y: data.u5_prevalence[name].map((v) => v * 100),
      mode: "lines", name: label(name) + " (under 5)", line: { color: color[name], width: 2.5 },
    });
    prevalenceTraces.push({
      x: data.days, y: data.all_ages_prevalence[name].map((v) => v * 100),
      mode: "lines", name: label(name) + " (all ages)", line: { color: color[name], width: 1.5, dash: "dot" },
    });
  });
  Plotly.newPlot("chart-prevalence", prevalenceTraces, {
    ...commonLayout, yaxis: { title: "% infectious", rangemode: "tozero" }, legend: { orientation: "h", y: -0.25 },
  }, commonConfig);

  const illnessTraces = data.pathogen_names.map((name) => ({
    x: data.days, y: data.cumulative_u5_illness_days[name],
    mode: "lines", name: label(name), line: { color: color[name], width: 2.5 }, fill: "tozeroy",
  }));
  Plotly.newPlot("chart-illness-days", illnessTraces, {
    ...commonLayout, yaxis: { title: "Cumulative U5 illness-days", rangemode: "tozero" }, legend: { orientation: "h", y: -0.25 },
  }, commonConfig);

  Plotly.newPlot("chart-wealth", [{
    x: data.days, y: data.mean_household_wealth, mode: "lines", line: { color: "#0f7a72", width: 2.5 },
  }], { ...commonLayout, yaxis: { title: "Fraction of max wealth" }, showlegend: false }, commonConfig);

  Plotly.newPlot("chart-care-seeking", [{
    x: data.days, y: data.cumulative_care_seeking_events, mode: "lines", line: { color: "#d97706", width: 2.5 },
  }], { ...commonLayout, yaxis: { title: "Cumulative events", rangemode: "tozero" }, showlegend: false }, commonConfig);

  // ---- Spatial scrubber: heatmap trace over the static Akuse basemap image ----
  const grids = data.spatial_daily_grids;
  const axes = data.spatial_axes;
  const basemap = data.basemap;
  let zmax = 0;
  for (const day of grids) for (const row of day) for (const v of row) if (v > zmax) zmax = v;
  if (zmax <= 0) zmax = 1;

  const slider = document.getElementById("spatial-day-slider");
  const dayLabel = document.getElementById("spatial-day-label");
  slider.max = String(grids.length - 1);
  slider.value = String(grids.length - 1); // default to the final day - the fullest picture

  function drawSpatialDay(dayIdx) {
    Plotly.react("chart-spatial", [{
      type: "heatmap", x: axes.x, y: axes.y, z: grids[dayIdx],
      zmin: 0, zmax: zmax, opacity: 0.7, colorscale: "Hot", showscale: true,
      colorbar: { title: "Cumulative<br>infections" },
    }], {
      margin: { t: 10, r: 20, l: 50, b: 40 },
      // constrain:"domain" on both axes matters here: scaleanchor locks the
      // map to true geographic proportions, but Plotly's default response
      // to a container wider than the data's own aspect ratio is to EXPAND
      // the x-axis range to fill it - leaving a gap past the basemap
      // image's true edge (confirmed visually: a grey band with no image
      // and a stray heatmap cell floating past it). constrain:"domain"
      // shrinks the plotting area with padding instead, keeping the axis
      // range locked to the image's actual extent.
      xaxis: { title: "Longitude", range: [basemap.minx, basemap.maxx], constrain: "domain" },
      yaxis: { title: "Latitude", range: [basemap.miny, basemap.maxy], scaleanchor: "x", constrain: "domain" },
      images: [{
        source: "/static/img/akuse_basemap.png", xref: "x", yref: "y",
        x: basemap.minx, y: basemap.maxy,
        sizex: basemap.maxx - basemap.minx, sizey: basemap.maxy - basemap.miny,
        xanchor: "left", yanchor: "top", layer: "below",
        // Plotly's default sizing ("contain") preserves the source image's
        // own aspect ratio and letterboxes it into sizex/sizey, leaving a
        // visible blank gap since the basemap PNG is a square 1000x1000
        // render of a non-square (wider than tall) geographic box.
        // "stretch" fills the box exactly, matching how the image was
        // actually baked (see scripts/bake_basemap.py - it renders into a
        // square canvas from the same non-square lon/lat extent, so this
        // un-does that squeeze rather than adding a new one).
        sizing: "stretch",
      }],
    }, commonConfig);
    dayLabel.textContent = "Day " + dayIdx;
  }

  drawSpatialDay(grids.length - 1);
  slider.addEventListener("input", () => drawSpatialDay(parseInt(slider.value, 10)));
};

document.addEventListener("DOMContentLoaded", () => window.renderResultsCharts());
document.body.addEventListener("htmx:afterSwap", () => window.renderResultsCharts());
