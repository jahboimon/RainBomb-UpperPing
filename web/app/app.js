(() => {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const DATA = "../data/derived/";
  const FILES = {
    metadata: "vector/station_metadata.geojson",
    summary: "vector/station_summary_2025.geojson",
    events: "vector/def5_events_2025.geojson",
    historicalEvents: "vector/def5_events_2012_2025.geojson",
    shap: "vector/station_shap_2025.geojson",
    admin: "vector/admin_boundaries.geojson",
    basin: "vector/upper_ping_boundary.geojson",
    rivers: "vector/rivers_upper_ping.geojson",
    hourly: "model_hourly_2025.json",
    metrics: "model_metrics_2025.json",
    lineage: "samples_data_lineage_2025.json",
    scenario: "scenario_baselines_2025.json",
    frozenModel: "model/xgb_final_compact20_v1.json",
    modelValidation: "model/xgb_validation_2025.json",
    manifest: "manifest.json"
  };

  const state = {
    data: {}, route: "map", selectedStation: null, selectedEvent: null,
    mapMode: "summary", mapSymbol: "count", mapThematic: "elevation",
    playbackMode: "events", timelineSequences: { events: [], every: [] },
    playback: null, xaiFeature: "Rain_t", xaiStation: null,
    scenarioBaseline: null, scenarioRuntime: null, scenarioValidation: null,
    eventPeriod: "evaluation"
  };
  const maps = {};

  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const byId = id => document.getElementById(id);

  function svg(tag, attrs = {}, text = "") {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text) node.textContent = text;
    return node;
  }

  function rasterRecord(layerName) {
    return state.data.manifest.rasters.find(item => item.source.endsWith(`/${layerName}`));
  }

  function initializeMap(id, options = {}) {
    const view = RainBombMaps.create(id, options);
    view.fitOptions = options.fitOptions || {};
    RainBombMaps.context(view, state.data);
    view.fit(view.layers.basin, view.fitOptions);
    return view;
  }

  function number(value, digits = 0) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "Not available";
    return Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function rawTime(value) {
    if (!value) return "Not available";
    return value.replace("T", " ").replace(/Z$/, "").slice(0, 16);
  }

  function shortTime(value) {
    if (!value) return "Not available";
    const parts = value.replace(/Z$/, "").split("T");
    const [year, month, day] = parts[0].split("-");
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return `${day} ${months[Number(month) - 1]} ${year} · ${parts[1].slice(0,5)}`;
  }

  function showToast(message) {
    const toast = byId("toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => toast.classList.remove("show"), 2400);
  }

  function metadataFor(station) {
    return state.data.metadata.features.find(feature => feature.properties.Station === station)?.properties || {};
  }

  function summaryFor(station) {
    return state.data.summary.features.find(feature => feature.properties.Station === station)?.properties || null;
  }

  function stationLabel(station) {
    const meta = metadataFor(station);
    return meta.Station_Name ? `${station} · ${meta.Station_Name}` : station;
  }

  async function loadData() {
    const entries = await Promise.all(Object.entries(FILES).map(async ([key, path]) => {
      const response = await fetch(DATA + path);
      if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
      return [key, await response.json()];
    }));
    state.data = Object.fromEntries(entries);
    const uniqueEventHours = new Map();
    state.data.events.features.forEach(feature => {
      const stored = String(feature.properties.Target_Datetime);
      const key = stored.slice(0, 13);
      if (!uniqueEventHours.has(key)) uniqueEventHours.set(key, stored);
    });
    state.timelineSequences.events = [...uniqueEventHours.values()].sort((a, b) => a.slice(0, 13).localeCompare(b.slice(0, 13)));
    state.timelineSequences.every = state.data.hourly.map(row => String(row.Target_Datetime));
  }

  function navigate(route, updateHash = true) {
    if (!byId("loading-screen").classList.contains("done")) return;
    state.route = route;
    $$(".route").forEach(page => {
      const active = page.dataset.page === route;
      page.hidden = !active;
      page.classList.toggle("active", active);
    });
    $$(".nav-link").forEach(link => {
      const active = link.dataset.routeTarget === route;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
    });
    byId("global-search-wrap").hidden = route !== "map";
    if (updateHash) history.replaceState(null, "", `#${route}`);
    if (route === "model") requestAnimationFrame(renderHourlyChart);
    if (route === "xai") requestAnimationFrame(renderXai);
    requestAnimationFrame(() => {
      const view = maps[route];
      if (!view) return;
      view.map.invalidateSize();
      if (!view.visibleFit) { view.fit(view.layers.basin, view.fitOptions); view.visibleFit = true; }
    });
  }

  function renderMapBase() {
    maps.map = initializeMap("map-leaflet", { zoomControl: false });
    setMapThematic(state.mapThematic);
    renderMainStations();
    renderMainEvents();
    updateMapLegend();
  }

  function stationStyle(summary, mode) {
    if (!summary) return { radius: 6, fill: "#10191a", missing: true };
    if (mode === "location") return { radius: 6, fill: "#78c3b8" };
    if (mode === "rate") {
      const ratio = Math.min(1, Number(summary.Def5_Rate_per100k) / 260);
      return { radius: 5 + Math.sqrt(ratio) * 9, fill: `rgb(${Math.round(120 + 105 * ratio)},${Math.round(195 - 29 * ratio)},${Math.round(184 - 110 * ratio)})` };
    }
    const ratio = Math.min(1, Number(summary.Def5_Count) / 9);
    return { radius: 5 + Math.sqrt(ratio) * 9, fill: `rgb(${Math.round(120 + 105 * ratio)},${Math.round(195 - 29 * ratio)},${Math.round(184 - 110 * ratio)})` };
  }

  function renderMainStations() {
    const group = L.layerGroup();
    if (!byId("gauges-toggle").checked) { maps.map.replace("stations", group); return; }
    state.data.metadata.features.forEach(feature => {
      const station = feature.properties.Station;
      const summary = summaryFor(station);
      const style = state.mapMode === "slice" ? { radius: 4, fill: "#6f817d", missing: false } : stationStyle(summary, state.mapSymbol);
      const [lon, lat] = feature.geometry.coordinates;
      const selected = state.selectedStation === station;
      L.circleMarker([lat, lon], {
        pane: "rainbomb-stations", radius: style.radius,
        color: selected ? "#fff1d6" : style.missing ? "#6f7e7a" : "#16201f",
        weight: selected ? 3 : 2, dashArray: style.missing ? "3 3" : null,
        fillColor: style.fill, fillOpacity: style.missing ? .15 : .92
      }).bindTooltip(state.mapMode === "slice" ? `${stationLabel(station)} · ตำแหน่งสถานี` : `${stationLabel(station)}${summary ? ` · Def5 ${summary.Def5_Count} เหตุการณ์` : " · ไม่มีข้อมูลสรุปช่วงประเมิน"}`)
        .on("click", () => selectMainStation(station)).addTo(group);
    });
    maps.map.replace("stations", group);
  }

  function selectMainStation(station) {
    state.selectedStation = station;
    const meta = metadataFor(station);
    const summary = summaryFor(station);
    byId("station-name").textContent = meta.Station_Name || "ไม่มีชื่อสถานี";
    byId("station-id").textContent = station;
    byId("station-count").textContent = summary ? `${number(summary.Def5_Count)} เหตุการณ์` : "ไม่มีข้อมูลช่วงประเมินปี 2025";
    byId("station-hours").textContent = summary ? number(summary.Available_Hours) : "ไม่มีข้อมูล";
    byId("station-rate").textContent = summary ? `${number(summary.Def5_Rate_per100k, 2)} ต่อ 100,000 ชั่วโมง` : "ไม่มีข้อมูล";
    byId("station-elevation").textContent = meta.Elevation_m == null ? "ไม่มีข้อมูลสถานีนี้" : `${number(meta.Elevation_m)} m`;
    byId("station-built").textContent = summary ? (Number(summary.BuiltUp_Binary) === 1 ? "พื้นที่สิ่งปลูกสร้าง" : "พื้นที่อื่น") : "ไม่มีข้อมูล";
    byId("station-suhi").textContent = summary?.SUHI_5K == null ? "ไม่มีข้อมูลสถานีนี้" : `${number(summary.SUHI_5K, 2)} · หน่วยยังไม่ยืนยัน`;
    const panel = byId("station-inspection");
    panel.classList.add("open");
    panel.setAttribute("aria-hidden", "false");
    renderMainStations();
  }

  function closeInspection() {
    state.selectedStation = null;
    const panel = byId("station-inspection");
    panel.classList.remove("open");
    panel.setAttribute("aria-hidden", "true");
    renderMainStations();
  }

  function renderMainEvents() {
    const group = L.layerGroup();
    let features = [];
    if (state.mapMode === "slice") {
      const selected = selectedTimelineKey();
      features = state.data.events.features.filter(feature => feature.properties.Target_Datetime.slice(0, 13) === selected);
    } else if (byId("events-toggle").checked) {
      features = state.data.events.features;
    }
    features.forEach(feature => {
      const p = feature.properties;
      const [lon, lat] = feature.geometry.coordinates;
      L.circleMarker([lat, lon], {
        pane: state.mapMode === "slice" ? "rainbomb-stations" : "rainbomb-events", radius: state.mapMode === "slice" ? 8 : 3.5,
        color: "#101817", weight: 1.2,
        fillColor: p.Outcome_0p5 === "TP" ? "#55b8aa" : "#d67d6e", fillOpacity: .88
      }).bindTooltip(`${stationLabel(p.Station)} · ${shortTime(p.Target_Datetime)} · ${outcomeLabel(p.Outcome_0p5)}`)
        .on("click", () => selectMainStation(p.Station)).addTo(group);
    });
    maps.map.replace("events", group);
    if (state.mapMode === "slice") {
      const tp = features.filter(feature => feature.properties.Outcome_0p5 === "TP").length;
      const fn = features.filter(feature => feature.properties.Outcome_0p5 === "FN").length;
      const stationOutcomes = features.map(feature => `${feature.properties.Station} · ${outcomeLabel(feature.properties.Outcome_0p5)}`).join(", ");
      const status = features.length
        ? `${stationOutcomes} · พบ Def5 จริง ${features.length} สถานี-ชั่วโมง · TP ${tp} · FN ${fn}`
        : "0 observed Def5 station-hours";
      byId("timeline-status").textContent = status;
      byId("timeline-status").title = status;
    }
  }

  const I = window.RainBombInterpretation;
  const outcomeLabel = value => value === "TP" ? "เกิดจริงและโมเดลตรวจพบ (TP)" : value === "FN" ? "เกิดจริงแต่โมเดลไม่ตรวจพบ (FN)" : value;

  function elevationLegendMarkup() {
    const stats = rasterRecord("Elev_m")?.source_statistics;
    if (!stats) return "";
    const ticks = [stats.p98, stats.p75, stats.median, stats.p25, stats.p02];
    return `<div class="elevation-key"><strong>ระดับความสูง (m)</strong><div class="elevation-scale"><span class="elevation-ramp"></span><span class="elevation-ticks">${ticks.map(value => `<i>${number(value)}</i>`).join("")}</span></div><small>ช่วงสีที่แสดง: เปอร์เซ็นไทล์ 2–98 · ไม่เปลี่ยนค่าต้นทาง</small></div>`;
  }

  function updateMapLegend() {
    const legend = byId("map-legend-content");
    const thematic = state.mapThematic === "elevation" ? elevationLegendMarkup() : "";
    const symbolLabel = state.mapSymbol === "count" ? "ขนาดวงกลม = จำนวน Def5 (เหตุการณ์/สถานี)" : state.mapSymbol === "rate" ? "ขนาดวงกลม = อัตรา Def5 (ต่อ 100,000 ชั่วโมง)" : "ตำแหน่งสถานี";
    const observations = state.mapMode === "slice"
      ? `<div class="legend-row"><i class="legend-symbol tp"></i><span>${outcomeLabel("TP")}</span></div><div class="legend-row"><i class="legend-symbol fn"></i><span>${outcomeLabel("FN")}</span></div><div class="legend-row"><i class="legend-symbol hollow"></i><span>ตำแหน่งสถานี</span></div>`
      : `<div class="legend-row"><i class="legend-symbol"></i><span>${symbolLabel}</span></div><div class="legend-row"><i class="legend-symbol hollow"></i><span>ไม่มีสรุปปี 2025</span></div>`;
    legend.innerHTML = `${thematic}${observations}<p class="legend-explanation">${I.uiText.station}</p>`;
  }

  function setControls(open) {
    byId("map-controls").classList.toggle("closed", !open);
    byId("drawer-toggle").classList.toggle("visible", !open);
    byId("drawer-toggle").setAttribute("aria-expanded", String(open));
  }

  function zoomMap(delta) { delta > 0 ? maps.map.map.zoomIn() : maps.map.map.zoomOut(); }

  function resetMap() { maps.map.fit(maps.map.layers.basin); }

  function timelineSequence() {
    return state.timelineSequences[state.playbackMode] || [];
  }

  function selectedTimelineDatetime() {
    return timelineSequence()[Number(byId("timeline-slider").value)] || "";
  }

  function selectedTimelineKey() {
    return selectedTimelineDatetime().slice(0, 13);
  }

  function formatStoredHour(value) {
    if (!value) return "No stored datetime available";
    const text = String(value);
    const [year, month, day] = text.slice(0, 10).split("-");
    const hour = text.slice(11, 16);
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return `${day} ${months[Number(month) - 1]} ${year} · ${hour} · เวลาในข้อมูลที่จัดเก็บ (ไม่แปลงเพิ่ม)`;
  }

  function configureTimelineSequence(preferredKey = "") {
    const slider = byId("timeline-slider");
    const sequence = timelineSequence();
    slider.max = String(Math.max(0, sequence.length - 1));
    const matchingIndex = preferredKey ? sequence.findIndex(value => value.slice(0, 13) === preferredKey) : -1;
    slider.value = String(matchingIndex >= 0 ? matchingIndex : 0);
    const labels = byId("timeline-sequence-labels");
    labels.innerHTML = state.playbackMode === "events"
      ? `<span>ชั่วโมงที่มีเหตุการณ์</span><span>${sequence.length} ชั่วโมงไม่ซ้ำ</span>`
      : `<span>May</span><span>Jun</span><span>Jul</span><span>Aug</span><span>Sep</span><span>Oct</span>`;
  }

  function updateTimeline() {
    const slider = byId("timeline-slider");
    slider.style.setProperty("--progress", `${Number(slider.value) / Number(slider.max) * 100}%`);
    if (state.mapMode === "summary") {
      byId("timeline-value").textContent = "พฤษภาคม–ตุลาคม 2025";
      byId("timeline-status").textContent = "สรุป 95 เหตุการณ์ที่ตรวจพบจริง";
    } else {
      byId("timeline-value").textContent = formatStoredHour(selectedTimelineDatetime());
    }
    renderMainEvents();
  }

  function stopPlayback() {
    clearInterval(state.playback);
    state.playback = null;
    byId("play-button").classList.remove("playing");
    byId("play-button").setAttribute("aria-label", "Play timeline");
  }

  function setMapMode(mode) {
    state.mapMode = mode;
    const summary = mode === "summary";
    byId("timeline").classList.toggle("summary", summary);
    byId("timeline-eyebrow").textContent = summary ? "สรุปทั้งช่วง" : "เลือกชั่วโมง";
    byId("timeline-slider").disabled = summary;
    byId("play-button").disabled = summary;
    if (summary) stopPlayback();
    else configureTimelineSequence();
    updateTimeline();
    renderMainStations();
    updateMapLegend();
  }

  function setPlaybackMode(mode) {
    const previousKey = selectedTimelineKey();
    stopPlayback();
    state.playbackMode = mode;
    configureTimelineSequence(previousKey);
    updateTimeline();
  }

  function togglePlayback() {
    if (state.mapMode === "summary") return;
    if (state.playback) { stopPlayback(); return; }
    const button = byId("play-button");
    button.classList.add("playing");
    button.setAttribute("aria-label", "Pause timeline");
    state.playback = setInterval(() => {
      const slider = byId("timeline-slider");
      slider.value = Number(slider.value) + 1 > Number(slider.max) ? 0 : Number(slider.value) + 1;
      updateTimeline();
    }, 650);
  }

  function setMapThematic(layer) {
    state.mapThematic = layer;
    if (layer === "elevation") maps.map.image("elevation", rasterRecord("Elev_m"), Number(byId("map-elevation-opacity").value) / 100, "elevation");
    else maps.map.replace("elevation", null);
    updateMapLegend();
  }

  const EVENT_PERIODS = {
    evaluation: { label: "Evaluation 2025", start: 2025, end: 2025, modelLinked: true },
    validation: { label: "Validation 2024", start: 2024, end: 2024 },
    training: { label: "Training 2020–2023", start: 2020, end: 2023 },
    "pre-model": { label: "ก่อนนำมาวิเคราะห์โมเดล 2012–2019", start: 2012, end: 2019 },
    "official-all": { label: "Official Def5 2012–2025", start: 2012, end: 2025 }
  };

  function activeEventFeatures() {
    const period = EVENT_PERIODS[state.eventPeriod];
    if (period.modelLinked) return state.data.events.features;
    return state.data.historicalEvents.features.filter(feature => {
      const year = Number(String(feature.properties.Target_Datetime).slice(0, 4));
      return year >= period.start && year <= period.end;
    });
  }

  function updateEventPeriod() {
    state.eventPeriod = byId("event-period-select").value;
    state.selectedEvent = null;
    const period = EVENT_PERIODS[state.eventPeriod];
    const features = activeEventFeatures();
    const stationSelect = byId("event-station-select");
    stationSelect.replaceChildren(new Option("All stations", "all"));
    [...new Set(features.map(feature => feature.properties.Station))].sort()
      .forEach(station => stationSelect.append(new Option(stationLabel(station), station)));
    const observedOnly = !period.modelLinked;
    byId("event-period-summary").textContent = `${number(features.length)} gauge-observed Def5 station-hours · ${period.label}`;
    byId("event-period-note").textContent = observedOnly
      ? `${number(features.length)} เหตุการณ์จาก frozen Official Def5 · แสดงข้อมูลสังเกตเท่านั้น ไม่มีผลโมเดล`
      : `${number(features.length)} เหตุการณ์ Evaluation 2025 ที่จับคู่กับผลโมเดล`;
    refreshEventOptions(true);
  }

  function setupEventsPage() {
    maps.events = initializeMap("event-map-leaflet", { zoomControl: true });
    const stationSelect = byId("event-station-select");
    byId("event-period-select").addEventListener("change", updateEventPeriod);
    stationSelect.addEventListener("change", () => refreshEventOptions(true));
    byId("event-select").addEventListener("change", event => selectEvent(Number(event.target.value)));
    updateEventPeriod();
  }

  function filteredEventIndexes() {
    const station = byId("event-station-select").value;
    return activeEventFeatures().map((feature, index) => ({ feature, index })).filter(item => station === "all" || item.feature.properties.Station === station).sort((a,b) => a.feature.properties.Target_Datetime.localeCompare(b.feature.properties.Target_Datetime));
  }

  function refreshEventOptions(selectFirst) {
    const items = filteredEventIndexes();
    const select = byId("event-select");
    select.replaceChildren();
    items.forEach(item => {
      const p = item.feature.properties;
      const status = p.Outcome_0p5 ? outcomeLabel(p.Outcome_0p5) : "Observed Def5";
      select.append(new Option(`${shortTime(p.Target_Datetime)} · ${p.Station} · ${status}`, String(item.index)));
    });
    const list = byId("event-list");
    list.replaceChildren();
    items.forEach(item => {
      const p = item.feature.properties;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `event-item ${p.Outcome_0p5 ? p.Outcome_0p5.toLowerCase() : "observed"}`;
      button.dataset.index = item.index;
      button.innerHTML = `<i></i><span><strong>${shortTime(p.Target_Datetime)}</strong><span>${p.Station} · ${p.Outcome_0p5 ? outcomeLabel(p.Outcome_0p5) : "Observed Def5"} · ฝนขณะเกิดเหตุ ${number(p.Rainfall,1)} mm/h</span></span>`;
      button.addEventListener("click", () => selectEvent(item.index));
      list.appendChild(button);
    });
    if (selectFirst && items.length) selectEvent(items[0].index);
  }

  function selectEvent(index) {
    const feature = activeEventFeatures()[index];
    if (!feature) return;
    state.selectedEvent = index;
    const p = feature.properties;
    byId("event-select").value = String(index);
    $$(".event-item").forEach(button => button.classList.toggle("active", Number(button.dataset.index) === index));
    byId("event-detail-title").textContent = stationLabel(p.Station);
    byId("event-detail-time").textContent = `${rawTime(p.Target_Datetime)} · เวลาในข้อมูลที่จัดเก็บ (ไม่แปลงเพิ่ม)`;
    byId("event-rain").textContent = `${number(p.Rainfall, 1)} mm/h`;
    byId("event-p99").textContent = `${number(p.P99_Frozen, 3)} mm/h`;
    byId("event-ante").textContent = `${number(p.Antecedent_3h_Rain, 1)} mm`;
    byId("event-post").textContent = `${number(p.Post1_Rain,1)} / ${number(p.Post2_Rain,1)} / ${number(p.Post3_Rain,1)} mm`;
    byId("event-postmax").textContent = `${number(p.Post3_Max,1)} mm/h`;
    byId("event-outcome").innerHTML = p.Outcome_0p5
      ? `<span class="outcome ${p.Outcome_0p5.toLowerCase()}">${outcomeLabel(p.Outcome_0p5)}</span>`
      : `<span class="outcome observed">ไม่มีผลโมเดล · observed-only period</span>`;
    byId("event-score").textContent = p.XGB_Final_Score == null ? "ไม่เกี่ยวข้องกับช่วงข้อมูลนี้" : number(p.XGB_Final_Score, 6);
    renderEventMap();
  }

  function renderEventMap() {
    const group = L.layerGroup();
    const filtered = filteredEventIndexes();
    filtered.forEach(item => {
      const p = item.feature.properties;
      const [lon, lat] = item.feature.geometry.coordinates;
      const selected = item.index === state.selectedEvent;
      L.circleMarker([lat, lon], {
        pane: "rainbomb-events", radius: selected ? 9 : 4,
        color: selected ? "#fff1d6" : "#18201f", weight: selected ? 3 : 1,
        fillColor: !p.Outcome_0p5 ? "#e1a64a" : p.Outcome_0p5 === "TP" ? "#55b8aa" : "#d67d6e",
        fillOpacity: selected ? 1 : .48
      }).bindTooltip(`${stationLabel(p.Station)} · ${shortTime(p.Target_Datetime)}`)
        .on("click", () => selectEvent(item.index)).addTo(group);
    });
    maps.events.replace("events", group);
    if (state.selectedEvent != null) {
      const selected = activeEventFeatures()[state.selectedEvent];
      const [lon, lat] = selected.geometry.coordinates;
      maps.events.map.panTo([lat, lon], { animate: false });
    }
  }

  function setupModelPage() {
    maps.model = initializeMap("model-map-leaflet", { zoomControl: true });
    const m = state.data.metrics;
    byId("metric-rows").textContent = number(m.evaluation_rows);
    byId("metric-actual").textContent = number(m.actual_def5);
    byId("metric-prevalence").textContent = `${number(m.prevalence * 100, 5)}%`;
    byId("metric-ap").textContent = number(m.average_precision, 8);
    byId("metric-auc").textContent = number(m.roc_auc, 8);
    byId("metric-precision").textContent = `${number(m.precision_at_0_5 * 100, 4)}%`;
    byId("metric-recall").textContent = `${number(m.recall_at_0_5 * 100, 2)}%`;
    ["tn","fp","fn","tp"].forEach(key => byId(`cm-${key}`).textContent = number(m.confusion_matrix[key.toUpperCase()]));
    byId("model-event-toggle").addEventListener("change", renderModelEventOverlay);
    renderModelStations();
    renderModelEventOverlay();
    renderHourlyChart();
  }

  function recallColor(value) {
    if (value == null) return "#10191a";
    const ratio = Math.max(0, Math.min(1, Number(value)));
    return `rgb(${Math.round(210 - 90 * ratio)},${Math.round(119 + 76 * ratio)},${Math.round(105 + 79 * ratio)})`;
  }

  function renderModelStations() {
    const group = L.layerGroup();
    state.data.summary.features.forEach(feature => {
      const p = feature.properties;
      const [lon, lat] = feature.geometry.coordinates;
      L.circleMarker([lat, lon], {
        pane: "rainbomb-stations", radius: 6 + Math.sqrt(Number(p.Def5_Count)) * 1.8,
        color: p.Recall == null ? "#73817e" : "#17201f", weight: 2,
        dashArray: p.Recall == null ? "3 3" : null,
        fillColor: recallColor(p.Recall), fillOpacity: p.Recall == null ? .15 : .9
      }).bindTooltip(`${stationLabel(p.Station)} · ตรวจพบเหตุการณ์จริง (Recall) ${p.Recall == null ? "ไม่มีข้อมูล" : number(p.Recall * 100,1) + "%"}`)
        .on("click", () => selectModelStation(p.Station)).addTo(group);
    });
    maps.model.replace("stations", group);
  }

  function selectModelStation(station) {
    const p = summaryFor(station);
    if (!p) return;
    byId("model-station-name").textContent = metadataFor(station).Station_Name || station;
    byId("model-station-id").textContent = station;
    byId("model-station-pos").textContent = `${number(p.TP)} / ${number(p.FN)}`;
    byId("model-station-neg").textContent = `${number(p.FP)} / ${number(p.TN)}`;
    byId("model-station-recall").textContent = p.Recall == null ? "ไม่มีเหตุการณ์จริง · TP + FN = 0" : `${number(p.Recall * 100, 1)}%`;
    byId("model-station-fpr").textContent = `${number(p.FP_Rate * 100, 2)}%`;
    byId("model-station-score").textContent = number(p.Mean_XGB_Score, 6);
  }

  function renderModelEventOverlay() {
    const group = L.layerGroup();
    if (!byId("model-event-toggle").checked) { maps.model.replace("events", group); return; }
    state.data.events.features.forEach(feature => {
      const [lon, lat] = feature.geometry.coordinates;
      L.circleMarker([lat, lon], {
        pane: "rainbomb-events", radius: 3.3, color: "#17201f", weight: 1,
        fillColor: feature.properties.Outcome_0p5 === "TP" ? "#55b8aa" : "#d67d6e", fillOpacity: .8
      }).bindTooltip(`${outcomeLabel(feature.properties.Outcome_0p5)} · ${stationLabel(feature.properties.Station)} · ${shortTime(feature.properties.Target_Datetime)}`).addTo(group);
    });
    maps.model.replace("events", group);
  }

  function renderHourlyChart() {
    const chart = byId("hourly-chart");
    if (!chart || !state.data.hourly?.length || chart.clientWidth === 0) return;
    chart.replaceChildren();
    const rows = state.data.hourly;
    const margin = { left: 44, right: 14, top: 15, bottom: 28 };
    const width = 1200 - margin.left - margin.right;
    const height = 280 - margin.top - margin.bottom;
    const maxY = Math.max(...rows.map(row => Number(row.Predicted_Count)), 1);
    const x = index => margin.left + index / (rows.length - 1) * width;
    const y = value => margin.top + height - Number(value) / maxY * height;
    [0,.25,.5,.75,1].forEach(part => {
      const yy = margin.top + height * (1 - part);
      chart.append(svg("line", { x1: margin.left, y1: yy, x2: 1200 - margin.right, y2: yy, class: "chart-grid" }));
      chart.append(svg("text", { x: margin.left - 8, y: yy + 4, "text-anchor": "end", class: "chart-axis-label" }, String(Math.round(maxY * part))));
    });
    const monthStarts = [0,31*24,61*24,92*24,123*24,153*24];
    ["May","Jun","Jul","Aug","Sep","Oct"].forEach((month,index) => chart.append(svg("text", { x: x(monthStarts[index]), y: 272, class: "chart-axis-label" }, month)));
    const pathFor = key => rows.map((row,index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(row[key]).toFixed(2)}`).join("");
    chart.append(svg("path", { d: pathFor("Predicted_Count"), class: "chart-line predicted" }));
    chart.append(svg("path", { d: pathFor("Actual_Count"), class: "chart-line actual" }));
    const cursor = svg("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + height, class: "chart-cursor", visibility: "hidden" });
    chart.append(cursor);
    const hit = svg("rect", { x: margin.left, y: margin.top, width, height, fill: "transparent" });
    hit.addEventListener("pointermove", event => {
      const rect = chart.getBoundingClientRect();
      const relative = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      const index = Math.min(rows.length - 1, Math.round(relative * (rows.length - 1)));
      const xx = x(index);
      cursor.setAttribute("x1", xx); cursor.setAttribute("x2", xx); cursor.setAttribute("visibility", "visible");
      const tip = byId("chart-tooltip");
      tip.hidden = false;
      tip.innerHTML = `<strong>${rawTime(rows[index].Target_Datetime)}</strong>Actual stations: ${rows[index].Actual_Count}<br>Predicted stations: ${rows[index].Predicted_Count}<br>Evaluated rows: ${rows[index].Evaluated_Rows}`;
      tip.style.left = `${Math.min(rect.width - 182, Math.max(8, event.clientX - rect.left + 10))}px`;
      tip.style.top = `${Math.max(8, event.clientY - rect.top - 68)}px`;
    });
    hit.addEventListener("pointerleave", () => { cursor.setAttribute("visibility", "hidden"); byId("chart-tooltip").hidden = true; });
    chart.append(hit);
  }

  const KDE_LAYERS = {
    Rain_t: "KDE_Rain_t_MeanAbs_R20km_C1km", q_850: "KDE_q_850_MeanAbs_R20km_C1km",
    u_850: "KDE_u_850_MeanAbs_R20km_C1km", u10: "KDE_u10_MeanAbs_R20km_C1km",
    Temp_t: "KDE_Temp_t_MeanAbs_R20km_C1km"
  };

  function setupXaiPage() {
    maps.xai = initializeMap("xai-map-leaflet", {
      zoomControl: true,
      fitOptions: { paddingTopLeft: [330, 24], paddingBottomRight: [350, 86], maxZoom: 9 }
    });
    maps.xai.image("elevation", rasterRecord("Elev_m"), .70, "elevation");
    maps.xai.visible("elevation", false);
    const available = new Set(state.data.shap.features.map(feature => feature.properties.Feature));
    const features = ["u_850", "q_850", "Rain_t", "u10", "Temp_t"].filter(feature => available.has(feature));
    const select = byId("xai-feature-select");
    features.forEach(feature => select.append(new Option(I.featureLabel(feature), feature)));
    select.value = state.xaiFeature;
    select.addEventListener("change", event => { state.xaiFeature = event.target.value; renderXai(); });
    ["xai-abs-toggle", "xai-signed-toggle", "xai-kde-toggle"].forEach(id => byId(id).addEventListener("change", renderXai));
    [
      ["xai-basin-toggle", "basin"], ["xai-admin-toggle", "admin"],
      ["xai-river-toggle", "rivers"], ["xai-elevation-toggle", "elevation"],
      ["xai-basemap-toggle", "basemap"]
    ].forEach(([id, layer]) => byId(id).addEventListener("change", event => {
      maps.xai.visible(layer, event.target.checked);
      if (id === "xai-elevation-toggle") byId("xai-elevation-legend").hidden = !event.target.checked;
    }));
    byId("xai-kde-opacity").addEventListener("input", event => maps.xai.opacity("kde", Number(event.target.value) / 100));
    byId("xai-elevation-opacity").addEventListener("input", event => maps.xai.opacity("elevation", Number(event.target.value) / 100));
    byId("xai-controls-close").addEventListener("click", () => setXaiPanel("controls", false));
    byId("xai-controls-toggle").addEventListener("click", () => setXaiPanel("controls", true));
    byId("xai-inspection-close").addEventListener("click", () => setXaiPanel("inspection", false));
    byId("xai-inspection-toggle").addEventListener("click", () => setXaiPanel("inspection", true));
    byId("xai-elevation-legend").innerHTML = elevationLegendMarkup() + "<p>สีและเงาเป็นวิธีแสดงผล ไม่เปลี่ยนค่าระดับความสูงต้นทาง</p>";
    setXaiPanel("inspection", false);
    renderXai();
  }

  function setXaiPanel(name, open) {
    const panel = byId(name === "controls" ? "xai-controls" : "xai-inspection");
    const toggle = byId(name === "controls" ? "xai-controls-toggle" : "xai-inspection-toggle");
    panel.classList.toggle("collapsed", !open);
    toggle.classList.toggle("visible", !open);
    toggle.setAttribute("aria-expanded", String(open));
    if (name === "inspection") panel.setAttribute("aria-hidden", String(!open));
  }

  function renderXai() {
    if (!state.data.shap) return;
    const records = state.data.shap.features.filter(feature => feature.properties.Feature === state.xaiFeature);
    const absValues = records.map(feature => Number(feature.properties.MeanAbsSHAP));
    const signedValues = records.map(feature => Number(feature.properties.MeanSignedSHAP));
    const absMin = Math.min(...absValues); const absMax = Math.max(...absValues); const absSpan = absMax - absMin || 1;
    const signedMax = Math.max(...signedValues.map(Math.abs), .000001);
    const showAbs = byId("xai-abs-toggle").checked;
    const showSigned = byId("xai-signed-toggle").checked;
    const group = L.layerGroup();
    if (showAbs || showSigned) records.forEach(feature => {
      const p = feature.properties;
      const absValue = Number(p.MeanAbsSHAP);
      const signedValue = Number(p.MeanSignedSHAP);
      const ratio = (absValue - absMin) / absSpan;
      const [lon, lat] = feature.geometry.coordinates;
      const selected = state.xaiStation === p.Station;
      L.circleMarker([lat, lon], {
        pane: "rainbomb-stations",
        radius: showAbs ? 3.5 + Math.sqrt(Math.max(0, ratio)) * 4.5 : 4.5,
        color: selected ? "#fff1d6" : !showSigned ? "#17201f" : signedValue < 0 ? "#2f87ad" : "#c77d3f",
        weight: selected ? 2.5 : showSigned ? 1.5 : 1,
        fillColor: showAbs ? `rgb(${Math.round(105 + 125 * ratio)},${Math.round(116 + 55 * ratio)},${Math.round(102 - 35 * ratio)})` : "#17201f",
        fillOpacity: showAbs ? .58 : .2
      }).bindTooltip(`${stationLabel(p.Station)} · ความสำคัญเฉลี่ย ${number(absValue, 6)} · ทิศทางเฉลี่ย ${number(signedValue, 6)}`)
        .on("click", () => selectXaiStation(p.Station)).addTo(group);
    });
    maps.xai.replace("stations", group);

    const kdeOn = byId("xai-kde-toggle").checked;
    if (kdeOn) maps.xai.image("kde", rasterRecord(KDE_LAYERS[state.xaiFeature]), Number(byId("xai-kde-opacity").value) / 100, "kde");
    else maps.xai.replace("kde", null);
    byId("xai-map-title").textContent = `${I.featureLabel(state.xaiFeature)} · บทบาทรายสถานี${kdeOn ? " และการกระจุกตัว KDE" : ""}`;
    byId("xai-shap-legend").innerHTML = `<strong>บทบาทตัวแปรรายสถานี · ${I.featureLabel(state.xaiFeature)}</strong>${showAbs ? `<div class="xai-ramp"></div><div class="xai-range"><span>${number(absMin,6)}</span><span>${number(absMax,6)}</span></div><p>วงกลมโปร่งขนาดเล็ก = ความสำคัญเฉลี่ยรายสถานี (MeanAbsSHAP); ค่ามากแสดงว่ามีบทบาทต่อผลโมเดลมากกว่า</p>` : ""}${showSigned ? `<div class="signed-key"><span class="negative">ลบ · ผลักคะแนนลง</span><span class="positive">บวก · ผลักคะแนนขึ้น</span></div><p>สีขอบ = ทิศทางเฉลี่ย (MeanSignedSHAP); ไม่ใช่ผลเชิงสาเหตุ</p>` : ""}${kdeOn ? `<p class="kde-legend-note">KDE: สีเข้มขึ้น = ความสำคัญของตัวแปรกระจุกตัวมากขึ้น ไม่ใช่ฝนหรือความเสี่ยง</p>` : ""}`;
    byId("xai-kde-legend").hidden = !kdeOn;
    if (state.xaiStation) selectXaiStation(state.xaiStation); else {
      byId("xai-station-name").textContent = "เลือกสถานีตรวจวัด";
      byId("xai-station-id").textContent = "คลิกจุดสถานีบนแผนที่";
      byId("xai-detail-feature").textContent = I.featureLabel(state.xaiFeature);
      ["xai-detail-group","xai-detail-abs","xai-detail-signed","xai-detail-n","xai-detail-pos","xai-detail-neg","xai-detail-recall"].forEach(id => byId(id).textContent = "—");
    }
  }

  function selectXaiStation(station) {
    state.xaiStation = station;
    const feature = state.data.shap.features.find(item => item.properties.Feature === state.xaiFeature && item.properties.Station === station);
    if (!feature) return;
    const p = feature.properties;
    byId("xai-station-name").textContent = metadataFor(station).Station_Name || station;
    byId("xai-station-id").textContent = station;
    byId("xai-detail-feature").textContent = I.featureLabel(p.Feature);
    byId("xai-detail-group").textContent = I.groupLabel(p.Feature_Group);
    byId("xai-detail-abs").textContent = number(p.MeanAbsSHAP, 8);
    byId("xai-detail-signed").textContent = number(p.MeanSignedSHAP, 8);
    byId("xai-detail-n").textContent = number(p.N);
    const summary = summaryFor(station);
    byId("xai-detail-pos").textContent = summary ? `${number(summary.TP)} / ${number(summary.FN)}` : "ไม่มีข้อมูล";
    byId("xai-detail-neg").textContent = summary ? `${number(summary.FP)} / ${number(summary.TN)}` : "ไม่มีข้อมูล";
    byId("xai-detail-recall").textContent = summary?.Recall == null ? "ไม่มีข้อมูล" : `${number(summary.Recall * 100, 1)}%`;
    const direction = Number(p.MeanSignedSHAP) < 0 ? "ลดคะแนนของโมเดลโดยเฉลี่ย" : Number(p.MeanSignedSHAP) > 0 ? "เพิ่มคะแนนของโมเดลโดยเฉลี่ย" : "ไม่แสดงทิศทางโดยเฉลี่ย";
    byId("xai-interpretation").textContent = `ความสำคัญเฉลี่ยบอกขนาดบทบาทของ ${I.featureLabel(p.Feature)}; ค่าทิศทางเฉลี่ย${direction}สำหรับโมเดลนี้ ไม่ใช่หลักฐานว่าตัวแปรเป็นสาเหตุของฝน`;
    setXaiPanel("inspection", true);
    renderXaiMarkersSelection();
  }

  function renderXaiMarkersSelection() {
    const layer = maps.xai.layers.stations;
    if (!layer) return;
    layer.eachLayer(marker => marker.setStyle({ weight: marker.getTooltip()?.getContent().startsWith(stationLabel(state.xaiStation)) ? 2.5 : byId("xai-signed-toggle").checked ? 1.5 : 1 }));
  }

  const SAMPLE_COLUMNS = {
    raw_canonical: ["Station", "Station_Name", "Datetime", "Rainfall", "Relative_Humidity", "Temperature"],
    feature_engineered: ["Station", "Datetime", "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "RHMin_6h", "Temp_t", "TempMax_6h", "q_850", "u_850", "v_850", "Elevation_m", "SUHI_5K", "BuiltUp_Binary"],
    model_ready: ["Station", "Datetime", "Target_Datetime", "Target_Def5_tPlus1", "Rain_t", "RainSum_3h", "RainSum_6h", "RH_t", "Temp_t", "q_850", "Elevation_m", "XGB_Final_Score", "Outcome_0p5"]
  };

  const SAMPLE_LABELS = {
    Station: "รหัสสถานี", Station_Name: "ชื่อสถานี", Datetime: "เวลา predictor t",
    Target_Datetime: "เวลาเป้าหมาย t+1", Rainfall: "ปริมาณฝน", Relative_Humidity: "ความชื้นสัมพัทธ์",
    Temperature: "อุณหภูมิ", Target_Def5_tPlus1: "Def5 ที่ t+1", XGB_Final_Score: "คะแนนจากแบบจำลอง",
    Outcome_0p5: "ผลที่เกณฑ์ 0.5"
  };

  function sampleHeader(column) {
    return I.features[column]?.displayNameTH || SAMPLE_LABELS[column] || column;
  }

  function sampleCell(value, column) {
    if (value === null || value === undefined) return "—";
    if (column === "Datetime" || column === "Target_Datetime") return rawTime(String(value));
    if (typeof value === "number") return Number(value).toLocaleString("en-US", { maximumFractionDigits: 6 });
    return String(value);
  }

  function renderSampleTable(stage = "raw_canonical") {
    const rows = state.data.lineage?.[stage] || [];
    const columns = SAMPLE_COLUMNS[stage];
    const root = byId("sample-table-wrap");
    root.replaceChildren();
    const table = document.createElement("table");
    table.className = "sample-table";
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    columns.forEach(column => {
      const th = document.createElement("th");
      const primary = document.createElement("span"); primary.textContent = sampleHeader(column);
      const technical = document.createElement("small"); technical.textContent = column;
      th.append(primary, technical); headerRow.append(th);
    });
    thead.append(headerRow); table.append(thead);
    const tbody = document.createElement("tbody");
    rows.forEach(row => {
      const tr = document.createElement("tr");
      columns.forEach(column => { const td = document.createElement("td"); td.textContent = sampleCell(row[column], column); tr.append(td); });
      tbody.append(tr);
    });
    table.append(tbody); root.append(table);
    $$("[data-sample-stage]").forEach(button => button.classList.toggle("active", button.dataset.sampleStage === stage));
  }

  function setupDataLineage() {
    $$("[data-sample-stage]").forEach(button => button.addEventListener("click", () => renderSampleTable(button.dataset.sampleStage)));
    renderSampleTable();
  }

  const SCENARIO_INPUTS = [
    { key: "Rain_t", label: "ปริมาณฝน ณ ชั่วโมงปัจจุบัน", code: "Rain_t", unit: "mm/h", step: .1 },
    { key: "Rain_tMinus1", label: "ปริมาณฝนย้อนหลัง 1 ชั่วโมง", code: "Rain t−1", unit: "mm/h", step: .1 },
    { key: "Rain_tMinus2", label: "ปริมาณฝนย้อนหลัง 2 ชั่วโมง", code: "Rain t−2", unit: "mm/h", step: .1 },
    { key: "Rain_tMinus3", label: "ปริมาณฝนย้อนหลัง 3 ชั่วโมง", code: "Rain t−3", unit: "mm/h", step: .1 },
    { key: "Rain_tMinus4", label: "ปริมาณฝนย้อนหลัง 4 ชั่วโมง", code: "Rain t−4", unit: "mm/h", step: .1 },
    { key: "Rain_tMinus5", label: "ปริมาณฝนย้อนหลัง 5 ชั่วโมง", code: "Rain t−5", unit: "mm/h", step: .1 },
    { key: "RH_t", label: I.features.RH_t.displayNameTH, code: "RH_t", unit: "%", step: .1 },
    { key: "Temp_t", label: I.features.Temp_t.displayNameTH, code: "Temp_t", unit: "°C", step: .1 },
    { key: "q_850", label: I.features.q_850.displayNameTH, code: "q_850", unit: "หน่วยตามข้อมูลต้นทาง", step: .0001 },
    { key: "u_850", label: I.features.u_850.displayNameTH, code: "u_850", unit: "หน่วยตามข้อมูลต้นทาง", step: .1 },
    { key: "v_850", label: I.features.v_850.displayNameTH, code: "v_850", unit: "หน่วยตามข้อมูลต้นทาง", step: .1 },
    { key: "Hour", label: "ชั่วโมงในข้อมูลที่จัดเก็บ", code: "Hour", unit: "0–23", step: 1 }
  ];

  function baselineInputValue(baseline, key) {
    if (key.startsWith("Rain_tMinus")) return baseline.rain_lags[key];
    if (key === "Hour") return Number(String(baseline.Datetime).slice(11, 13));
    return baseline.features[key];
  }

  function scenarioValue(key) {
    return Number(byId(`scenario-input-${key}`).value);
  }

  function scenarioFeatures() {
    const features = { ...state.scenarioBaseline.features };
    const rain = ["Rain_t", "Rain_tMinus1", "Rain_tMinus2", "Rain_tMinus3", "Rain_tMinus4", "Rain_tMinus5"].map(scenarioValue);
    features.Rain_t = rain[0];
    features.RainSum_3h = rain.slice(0, 3).reduce((sum, value) => sum + value, 0);
    features.RainSum_6h = rain.reduce((sum, value) => sum + value, 0);
    ["RH_t", "Temp_t", "q_850", "u_850", "v_850"].forEach(key => { features[key] = scenarioValue(key); });
    const hour = scenarioValue("Hour");
    features.Hour_sin = Math.sin(2 * Math.PI * hour / 24);
    features.Hour_cos = Math.cos(2 * Math.PI * hour / 24);
    return features;
  }

  function updateScenarioDerived() {
    if (!state.scenarioBaseline) return;
    const features = scenarioFeatures();
    byId("scenario-rain3").textContent = `${number(features.RainSum_3h, 2)} mm`;
    byId("scenario-rain6").textContent = `${number(features.RainSum_6h, 2)} mm`;
    byId("scenario-hour-cycle").textContent = `${number(features.Hour_sin, 6)} / ${number(features.Hour_cos, 6)}`;
    byId("scenario-score-new").textContent = "—";
    byId("scenario-score-delta").textContent = "—";
    byId("scenario-result-summary").textContent = "ปรับค่าในขั้นที่ 2 แล้วกดคำนวณเพื่อเปรียบเทียบ";
  }

  function renderScenarioMap() {
    maps.scenario = initializeMap("scenario-map-leaflet", { zoomControl: true, fitOptions: { padding: [16, 16], maxZoom: 9 } });
    renderScenarioMapMarkers();
  }

  function renderScenarioMapMarkers() {
    if (!maps.scenario) return;
    const counts = new Map();
    state.data.scenario.baselines.forEach(item => counts.set(item.Station, (counts.get(item.Station) || 0) + 1));
    const filteredStation = byId("scenario-station")?.value || "all";
    const highlighted = filteredStation !== "all" ? filteredStation : state.scenarioBaseline?.Station;
    const group = L.layerGroup();
    state.data.metadata.features.forEach(feature => {
      const station = feature.properties.Station;
      if (!counts.has(station) || !feature.geometry) return;
      const [lon, lat] = feature.geometry.coordinates;
      const selected = station === highlighted;
      L.circleMarker([lat, lon], {
        pane: "rainbomb-stations", radius: selected ? 8 : 5,
        color: selected ? "#fff1d6" : "#172120", weight: selected ? 3 : 2,
        fillColor: selected ? "#e1a64a" : "#78c3b8", fillOpacity: .94
      }).bindTooltip(`${stationLabel(station)} · ${counts.get(station)} กรณี`)
        .on("click", () => {
          byId("scenario-station").value = station;
          byId("scenario-search").value = "";
          refreshScenarioBaselines();
          maps.scenario.map.panTo([lat, lon]);
        }).addTo(group);
    });
    maps.scenario.replace("scenarioStations", group);
  }

  function renderBaselineShap(baseline) {
    const chart = byId("scenario-shap-chart");
    chart.replaceChildren();
    const rows = Object.entries(baseline.baseline_shap).map(([feature, value]) => ({ feature, value: Number(value) }));
    const selected = [
      ...rows.filter(row => row.value > 0).sort((a, b) => b.value - a.value).slice(0, 5),
      ...rows.filter(row => row.value < 0).sort((a, b) => a.value - b.value).slice(0, 5)
    ];
    const maxAbs = Math.max(...selected.map(row => Math.abs(row.value)), .000001);
    selected.forEach(row => {
      const item = document.createElement("div"); item.className = `shap-row ${row.value >= 0 ? "positive" : "negative"}`;
      const label = document.createElement("span"); label.textContent = I.featureLabel(row.feature);
      const track = document.createElement("i"); const bar = document.createElement("b"); bar.style.width = `${Math.abs(row.value) / maxAbs * 100}%`; track.append(bar);
      const value = document.createElement("strong"); value.textContent = `${row.value >= 0 ? "+" : ""}${number(row.value, 6)}`;
      item.append(label, track, value); chart.append(item);
    });
  }

  function renderScenarioBaseline() {
    const baseline = state.data.scenario.baselines.find(item => item.id === byId("scenario-baseline").value);
    if (!baseline) return;
    state.scenarioBaseline = baseline;
    byId("scenario-meta-station").textContent = `${baseline.Station} · ${baseline.Station_Name || "ไม่มีชื่อสถานี"}`;
    byId("scenario-meta-time").textContent = rawTime(baseline.Datetime);
    byId("scenario-meta-target-time").textContent = rawTime(baseline.Target_Datetime);
    byId("scenario-meta-target-rain").textContent = baseline.Target_Rainfall_tPlus1 == null
      ? "ไม่มีค่าใน frozen event metadata (กรณีไม่เกิด Def5)"
      : `${number(baseline.Target_Rainfall_tPlus1, 1)} mm/h`;
    byId("scenario-meta-actual").textContent = baseline.Target_Def5_tPlus1 ? "เกิด Def5 (1)" : "ไม่เกิด Def5 (0)";
    byId("scenario-meta-outcome").textContent = outcomeLabel(baseline.Outcome_0p5);
    byId("scenario-score-original").textContent = number(baseline.XGB_Final_Score, 6);
    SCENARIO_INPUTS.forEach(item => { byId(`scenario-input-${item.key}`).value = baselineInputValue(baseline, item.key); });
    byId("scenario-locked").innerHTML = `<div><span>${I.featureLabel("Elevation_m")}</span><strong>${number(baseline.features.Elevation_m, 2)}</strong></div><div><span>${I.featureLabel("SUHI_5K")}</span><strong>${number(baseline.features.SUHI_5K, 4)}</strong></div><div><span>${I.featureLabel("BuiltUp_Binary")}</span><strong>${number(baseline.features.BuiltUp_Binary)}</strong></div>`;
    updateScenarioDerived();
    renderBaselineShap(baseline);
    renderScenarioMapMarkers();
  }

  function refreshScenarioBaselines() {
    const station = byId("scenario-station").value;
    const outcome = byId("scenario-outcome").value;
    const query = byId("scenario-search").value.trim().toLocaleLowerCase();
    const select = byId("scenario-baseline");
    const previous = select.value;
    const rows = state.data.scenario.baselines.filter(item => {
      const matchesStation = station === "all" || item.Station === station;
      const matchesOutcome = outcome === "all" || item.Outcome_0p5 === outcome;
      const haystack = `${item.Station} ${item.Station_Name || ""} ${item.Datetime} ${item.Target_Datetime} ${item.Outcome_0p5}`.toLocaleLowerCase();
      return matchesStation && matchesOutcome && (!query || haystack.includes(query));
    });
    select.replaceChildren();
    byId("scenario-case-count").textContent = `${number(rows.length)} กรณี`;
    const list = byId("scenario-case-list");
    list.replaceChildren();
    if (!rows.length) {
      select.append(new Option("ไม่มีกรณีที่ตรงกับตัวกรอง", ""));
      const empty = document.createElement("p"); empty.className = "scenario-case-empty"; empty.textContent = "ไม่พบกรณีที่ตรงกับคำค้นและประเภทผล"; list.append(empty);
      state.scenarioBaseline = null;
      ["scenario-meta-station","scenario-meta-time","scenario-meta-target-time","scenario-meta-target-rain","scenario-meta-actual","scenario-meta-outcome","scenario-score-original","scenario-score-new","scenario-score-delta"].forEach(id => { byId(id).textContent = "—"; });
      byId("scenario-run").disabled = true;
      renderScenarioMapMarkers();
      return;
    }
    rows.forEach(item => {
      select.append(new Option(`${rawTime(item.Target_Datetime)} · ${item.Station} · ${item.Outcome_0p5}`, item.id));
      const button = document.createElement("button");
      button.type = "button";
      button.className = `scenario-case ${item.Outcome_0p5.toLowerCase()}`;
      button.dataset.baselineId = item.id;
      const title = document.createElement("strong"); title.textContent = `${item.Station} · ${item.Station_Name || "ไม่มีชื่อสถานี"}`;
      const meta = document.createElement("span"); meta.textContent = `${rawTime(item.Target_Datetime)} · ${item.Outcome_0p5} · score ${number(item.XGB_Final_Score, 4)}`;
      button.append(title, meta);
      button.addEventListener("click", () => {
        select.value = item.id;
        $$(".scenario-case").forEach(node => node.classList.toggle("active", node === button));
        renderScenarioBaseline();
      });
      list.append(button);
    });
    if (rows.some(item => item.id === previous)) select.value = previous;
    byId("scenario-run").disabled = state.scenarioValidation?.status !== "PASS";
    renderScenarioBaseline();
    const activeId = select.value;
    $$(".scenario-case").forEach(node => node.classList.toggle("active", node.dataset.baselineId === activeId));
  }

  function runScenario() {
    if (!state.scenarioRuntime || state.scenarioValidation?.status !== "PASS" || !state.scenarioBaseline) return;
    const score = state.scenarioRuntime.predict(scenarioFeatures());
    const delta = score - Number(state.scenarioBaseline.XGB_Final_Score);
    byId("scenario-score-new").textContent = number(score, 6);
    byId("scenario-score-delta").textContent = `${delta >= 0 ? "+" : ""}${number(delta, 6)}`;
    const direction = Math.abs(delta) < 0.000001 ? "แทบไม่เปลี่ยนจากกรณีต้นฉบับ" : delta > 0 ? "เพิ่มขึ้นจากกรณีต้นฉบับ" : "ลดลงจากกรณีต้นฉบับ";
    byId("scenario-result-summary").textContent = `คะแนนแบบจำลอง${direction} ${number(Math.abs(delta), 6)} หน่วย — ค่านี้ไม่ใช่ความน่าจะเป็น`;
    byId("scenario-result-note").textContent = "คำนวณจาก frozen XGBoost Compact20 แบบ exact tree traversal ในเบราว์เซอร์; คะแนนไม่ใช่ calibrated probability";
  }

  function applyScenarioPreset(name) {
    if (!state.scenarioBaseline) return;
    const set = (key, value) => {
      const input = byId(`scenario-input-${key}`);
      input.value = Math.min(Number(input.max), Math.max(Number(input.min), value));
    };
    if (name === "rain-now") set("Rain_t", scenarioValue("Rain_t") + 10);
    if (name === "rain-history") {
      set("Rain_tMinus1", scenarioValue("Rain_tMinus1") + 5);
      set("Rain_tMinus2", scenarioValue("Rain_tMinus2") + 5);
    }
    if (name === "humidity") set("RH_t", scenarioValue("RH_t") + 10);
    if (name === "night") set("Hour", 22);
    updateScenarioDerived();
  }

  function setupScenarioPage() {
    const stations = [...new Set(state.data.scenario.baselines.map(item => item.Station))].sort();
    const stationSelect = byId("scenario-station");
    stationSelect.replaceChildren();
    stationSelect.append(new Option("ทุกสถานี", "all"));
    stations.forEach(station => stationSelect.append(new Option(stationLabel(station), station)));
    SCENARIO_INPUTS.forEach(item => {
      const range = state.data.scenario.ranges[item.key];
      const label = document.createElement("label"); label.className = "scenario-input"; label.htmlFor = `scenario-input-${item.key}`;
      const heading = document.createElement("span"); heading.textContent = item.label;
      const code = document.createElement("small"); code.textContent = `${item.code} · ${item.unit}`;
      const input = document.createElement("input"); input.id = `scenario-input-${item.key}`; input.type = "number"; input.step = item.step; input.min = range.min; input.max = range.max;
      const guide = document.createElement("em"); guide.textContent = `ช่วงที่พบ p01–p99: ${number(range.p01, item.step < .001 ? 5 : 2)} – ${number(range.p99, item.step < .001 ? 5 : 2)}`;
      input.addEventListener("input", updateScenarioDerived);
      const advanced = ["q_850", "u_850", "v_850"].includes(item.key);
      label.append(heading, code, input, guide); byId(advanced ? "scenario-advanced-inputs" : "scenario-simple-inputs").append(label);
    });
    stationSelect.addEventListener("change", refreshScenarioBaselines);
    byId("scenario-outcome").addEventListener("change", refreshScenarioBaselines);
    byId("scenario-baseline").addEventListener("change", renderScenarioBaseline);
    byId("scenario-search").addEventListener("input", refreshScenarioBaselines);
    $$('[data-scenario-outcome]').forEach(button => button.addEventListener("click", () => {
      byId("scenario-outcome").value = button.dataset.scenarioOutcome;
      $$('[data-scenario-outcome]').forEach(node => node.classList.toggle("active", node === button));
      refreshScenarioBaselines();
    }));
    byId("scenario-reset").addEventListener("click", renderScenarioBaseline);
    byId("scenario-map-reset").addEventListener("click", () => {
      stationSelect.value = "all";
      byId("scenario-search").value = "";
      refreshScenarioBaselines();
      maps.scenario.fit(maps.scenario.layers.basin, maps.scenario.fitOptions);
    });
    $$('[data-scenario-preset]').forEach(button => button.addEventListener("click", () => applyScenarioPreset(button.dataset.scenarioPreset)));
    byId("scenario-run").addEventListener("click", runScenario);
    try {
      state.scenarioRuntime = RainBombXGB.create(state.data.frozenModel);
      state.scenarioValidation = RainBombXGB.validate(state.scenarioRuntime, state.data.modelValidation);
    } catch (error) {
      state.scenarioValidation = { status: "FAIL", rowCount: 0, maxAbsoluteError: NaN, tolerance: state.data.modelValidation?.tolerance };
      console.error(error);
    }
    const badge = byId("scenario-validation");
    const passed = state.scenarioValidation.status === "PASS";
    badge.className = `validation-badge ${passed ? "pass" : "fail"}`;
    badge.textContent = passed
      ? `PASS · ${number(state.scenarioValidation.rowCount)} rows · max error ${state.scenarioValidation.maxAbsoluteError.toExponential(2)}`
      : "FAIL · browser inference disabled";
    byId("scenario-run").disabled = !passed;
    if (!passed) byId("scenario-result-note").textContent = "Exact browser-side inference did not pass validation; scenario scores are disabled.";
    renderScenarioMap();
    refreshScenarioBaselines();
  }

  function registerInteractions() {
    $$('[data-route-target]').forEach(button => button.addEventListener("click", () => navigate(button.dataset.routeTarget)));
    byId("close-controls").addEventListener("click", () => setControls(false));
    byId("drawer-toggle").addEventListener("click", () => setControls(true));
    byId("close-inspection").addEventListener("click", closeInspection);
    byId("zoom-in").addEventListener("click", () => zoomMap(.2));
    byId("zoom-out").addEventListener("click", () => zoomMap(-.2));
    byId("reset-view").addEventListener("click", resetMap);
    $$("input[name='map-view-mode']").forEach(input => input.addEventListener("change", () => setMapMode(input.value)));
    $$("input[name='station-symbol']").forEach(input => input.addEventListener("change", () => { state.mapSymbol = input.value; renderMainStations(); updateMapLegend(); }));
    $$("input[name='map-thematic']").forEach(input => input.addEventListener("change", () => setMapThematic(input.value)));
    byId("basin-toggle").addEventListener("change", event => maps.map.visible("basin", event.target.checked));
    byId("admin-toggle").addEventListener("change", event => maps.map.visible("admin", event.target.checked));
    byId("river-toggle").addEventListener("change", event => maps.map.visible("rivers", event.target.checked));
    byId("gauges-toggle").addEventListener("change", renderMainStations);
    byId("events-toggle").addEventListener("change", renderMainEvents);
    byId("map-elevation-opacity").addEventListener("input", event => maps.map.opacity("elevation", Number(event.target.value) / 100));
    byId("timeline-slider").addEventListener("input", updateTimeline);
    byId("timeline-playback-mode").addEventListener("change", event => setPlaybackMode(event.target.value));
    byId("play-button").addEventListener("click", togglePlayback);
    byId("view-station-events").addEventListener("click", () => {
      if (!state.selectedStation) return;
      byId("event-station-select").value = state.selectedStation;
      refreshEventOptions(true);
      navigate("events");
    });

    const search = byId("station-search");
    search.addEventListener("keydown", event => {
      if (event.key !== "Enter") return;
      const value = search.value.trim().toLowerCase();
      const match = state.data.metadata.features.find(feature => {
        const p = feature.properties;
        return p.Station.toLowerCase().includes(value) || String(p.Station_Name || "").toLowerCase().includes(value);
      });
      if (match) { navigate("map"); selectMainStation(match.properties.Station); } else showToast("No station matched the verified station list");
    });
    document.addEventListener("keydown", event => {
      if (event.key === "/" && state.route === "map" && document.activeElement !== search) { event.preventDefault(); search.focus(); }
      if (event.key === "Escape") { closeInspection(); stopPlayback(); }
    });

    window.addEventListener("resize", () => {
      Object.values(maps).forEach(view => view.map.invalidateSize());
      if (state.route === "model") renderHourlyChart();
    });
  }

  function registerWebMCPTools() {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const register = tool => { try { void Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(() => {}); } catch (_) {} };
    register({ name: "open_research_page", title: "Open RainBomb Atlas page", description: "Open one of the six scientific explorer pages.", inputSchema: { type: "object", properties: { page: { type: "string", enum: ["map","events","model","xai","scenario","method"] } }, required: ["page"], additionalProperties: false }, annotations: { readOnlyHint: false, untrustedContentHint: false }, execute(input) { navigate(input.page); return { page: input.page }; } });
    register({ name: "select_station", title: "Select verified station", description: "Open the Map inspection drawer for a verified station ID.", inputSchema: { type: "object", properties: { stationId: { type: "string" } }, required: ["stationId"], additionalProperties: false }, annotations: { readOnlyHint: false, untrustedContentHint: false }, execute(input) { if (!metadataFor(input.stationId).Station) throw new Error("Unknown station ID"); navigate("map"); selectMainStation(input.stationId); return { stationId: input.stationId, inspection: "open" }; } });
    register({ name: "set_map_view_mode", title: "Set map view mode", description: "Switch between period summary and time slice on the Map page.", inputSchema: { type: "object", properties: { mode: { type: "string", enum: ["summary","slice"] } }, required: ["mode"], additionalProperties: false }, annotations: { readOnlyHint: false, untrustedContentHint: false }, execute(input) { const radio = $(`input[name='map-view-mode'][value='${input.mode}']`); radio.checked = true; setMapMode(input.mode); return { mode: input.mode }; } });
  }

  async function init() {
    try {
      await loadData();
      renderMapBase();
      setupEventsPage();
      setupModelPage();
      setupXaiPage();
      setupScenarioPage();
      setupDataLineage();
      registerInteractions();
      updateTimeline();
      const loading = byId("loading-screen");
      loading.classList.add("done");
      loading.setAttribute("aria-hidden", "true");
      window.setTimeout(() => { loading.hidden = true; }, 320);
      const initial = location.hash.slice(1);
      navigate(["map","events","model","xai","scenario","method"].includes(initial) ? initial : "map", false);
      registerWebMCPTools();
    } catch (error) {
      console.error(error);
      const loading = byId("loading-screen");
      loading.classList.add("error");
      loading.querySelector("p").textContent = location.protocol === "file:"
        ? "Verified data cannot be loaded from a file:// address. Open the local preview URL supplied with this project."
        : `Verified spatial outputs could not be loaded: ${error.message}`;
    }
  }

  init();
})();
