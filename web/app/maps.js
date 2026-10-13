(() => {
  "use strict";

  const BASEMAPS = {
    osm: {
      label: "OpenStreetMap",
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      options: {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }
    }
  };

  const PANE_Z = {
    elevation: 310,
    kde: 330,
    admin: 410,
    rivers: 420,
    basin: 430,
    events: 450,
    stations: 460
  };

  function assetPath(record) {
    return `../${record.derived_png.replace(/^dist\//, "")}`;
  }

  function imageBounds(record) {
    const [west, south, east, north] = record.web_bounds;
    return [[south, west], [north, east]];
  }

  function create(containerId, options = {}) {
    const map = L.map(containerId, {
      zoomControl: options.zoomControl !== false,
      attributionControl: true,
      preferCanvas: true,
      minZoom: 6,
      maxZoom: 18
    });
    Object.entries(PANE_Z).forEach(([name, zIndex]) => {
      map.createPane(`rainbomb-${name}`);
      map.getPane(`rainbomb-${name}`).style.zIndex = String(zIndex);
    });
    map.getPane("rainbomb-kde").style.pointerEvents = "none";
    map.getPane("rainbomb-elevation").style.pointerEvents = "none";

    const spec = BASEMAPS.osm;
    const base = L.tileLayer(spec.url, spec.options).addTo(map);
    const layers = { basemap: base };
    if (options.scale !== false) L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);

    function replace(name, layer) {
      if (layers[name] && map.hasLayer(layers[name])) map.removeLayer(layers[name]);
      layers[name] = layer;
      if (layer) layer.addTo(map);
      return layer;
    }

    function geo(name, data, layerOptions = {}) {
      return replace(name, L.geoJSON(data, layerOptions));
    }

    function image(name, record, opacity, pane) {
      if (!record) return replace(name, null);
      return replace(name, L.imageOverlay(assetPath(record), imageBounds(record), {
        opacity,
        pane: `rainbomb-${pane}`,
        alt: name === "elevation" ? "Colored terrain elevation" : "Weighted KDE concentration"
      }));
    }

    function visible(name, show) {
      const layer = layers[name];
      if (!layer) return;
      if (show && !map.hasLayer(layer)) layer.addTo(map);
      if (!show && map.hasLayer(layer)) map.removeLayer(layer);
    }

    function opacity(name, value) {
      const layer = layers[name];
      if (layer?.setOpacity) layer.setOpacity(Number(value));
    }

    function fit(layer = layers.basin, fitOptions = {}) {
      if (!layer?.getBounds || !layer.getBounds().isValid()) return;
      map.fitBounds(layer.getBounds(), { padding: [28, 28], maxZoom: 10, ...fitOptions });
    }

    return { map, layers, geo, image, replace, visible, opacity, fit };
  }

  function context(view, data) {
    view.geo("admin", data.admin, {
      pane: "rainbomb-admin",
      style: { color: "#8b7d69", weight: 1.1, opacity: .8, fillOpacity: 0, dashArray: "4 5" },
      interactive: false
    });
    view.geo("rivers", data.rivers, {
      pane: "rainbomb-rivers",
      style: feature => ({
        color: Number(feature.properties.MAIN_RIV) === 1 ? "#267da0" : "#559ab2",
        weight: Number(feature.properties.MAIN_RIV) === 1 ? 1.8 : .8,
        opacity: Number(feature.properties.MAIN_RIV) === 1 ? .9 : .65
      }),
      interactive: false
    });
    view.geo("basin", data.basin, {
      pane: "rainbomb-basin",
      style: { color: "#347f78", weight: 2.4, opacity: 1, fillColor: "#7bb6aa", fillOpacity: .035 },
      interactive: false
    });
    return view.layers.basin;
  }

  window.RainBombMaps = { BASEMAPS, create, context, assetPath, imageBounds };
})();
