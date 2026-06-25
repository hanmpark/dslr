const state = {
  summary: null,
  features: [],
  colors: {},
  trainSeries: {},
  accSeries: {},
  trainMeta: {
    completedHouses: 0,
    currentHouse: null,
    currentEpoch: 0,
    totalEpochs: 1000,
    latestLoss: null,
    latestAccuracy: null,
  },
};
window.appState = state;

const houseOrder = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"];

function $(selector) {
  return document.querySelector(selector);
}

function all(selector) {
  return Array.from(document.querySelectorAll(selector));
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toLocaleString("fr-FR", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(220, Math.floor(rect.width));
  const height = Math.max(160, Math.floor(rect.height || canvas.clientHeight || 300));
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function clear(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
}

function bounds(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value === null || value === undefined || Number.isNaN(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function drawAxes(ctx, width, height, margin, xLabel, yLabel) {
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, height - margin.bottom);
  ctx.lineTo(width - margin.right, height - margin.bottom);
  ctx.stroke();

  ctx.fillStyle = "#64748b";
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(xLabel || "", (width + margin.left - margin.right) / 2, height - 8);
  ctx.save();
  ctx.translate(13, (height + margin.top - margin.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel || "", 0, 0);
  ctx.restore();
}

function scale(value, min, max, a, b) {
  return a + ((value - min) / (max - min)) * (b - a);
}

function drawBarChart(canvas, labels, values, colors) {
  const { ctx, width, height } = fitCanvas(canvas);
  clear(ctx, width, height);
  const margin = { top: 18, right: 18, bottom: 42, left: 44 };
  const max = Math.max(1, ...values);
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const gap = 12;
  const barW = Math.max(12, (plotW - gap * (labels.length - 1)) / labels.length);

  drawAxes(ctx, width, height, margin, "", "count");
  labels.forEach((label, i) => {
    const x = margin.left + i * (barW + gap);
    const h = (values[i] / max) * plotH;
    ctx.fillStyle = colors[i];
    ctx.fillRect(x, height - margin.bottom - h, barW, h);
    ctx.fillStyle = "#334155";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(label.slice(0, 10), x + barW / 2, height - 21);
    ctx.fillText(values[i], x + barW / 2, height - margin.bottom - h - 5);
  });
}

function drawHistogram(canvas, data) {
  const { ctx, width, height } = fitCanvas(canvas);
  clear(ctx, width, height);
  const margin = { top: 18, right: 18, bottom: 42, left: 52 };
  const houses = data.houses;
  const maxCount = Math.max(1, ...houses.flatMap((h) => data.counts[h]));
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const groupW = plotW / data.bins.length;
  const barW = Math.max(1, groupW / houses.length);

  drawAxes(ctx, width, height, margin, data.feature, "frequency");
  houses.forEach((house, hi) => {
    const counts = data.counts[house];
    ctx.fillStyle = data.colors[house];
    counts.forEach((count, i) => {
      const h = (count / maxCount) * plotH;
      const x = margin.left + i * groupW + hi * barW;
      const y = height - margin.bottom - h;
      ctx.globalAlpha = 0.72;
      ctx.fillRect(x, y, barW, h);
      ctx.globalAlpha = 1;
    });
  });
  drawLegend(ctx, width, houses, data.colors);
}

function drawLegend(ctx, width, houses, colors) {
  const itemW = 102;
  const totalW = itemW * houses.length;
  let x = Math.max(12, width - totalW - 12);
  let y = 18;
  ctx.font = "12px system-ui";
  houses.forEach((house) => {
    if (x + itemW > width - 8) {
      x = 12;
      y += 18;
    }
    ctx.fillStyle = colors[house];
    ctx.fillRect(x, y - 9, 12, 12);
    ctx.fillStyle = "#334155";
    ctx.fillText(house, x + 16, y);
    x += itemW;
  });
}

function drawScatter(canvas, data) {
  const { ctx, width, height } = fitCanvas(canvas);
  clear(ctx, width, height);
  const margin = { top: 18, right: 18, bottom: 48, left: 58 };
  const xs = data.points.map((p) => p.x);
  const ys = data.points.map((p) => p.y);
  const [minX, maxX] = bounds(xs);
  const [minY, maxY] = bounds(ys);

  drawAxes(ctx, width, height, margin, data.xFeature, data.yFeature);
  data.points.forEach((point) => {
    const x = scale(point.x, minX, maxX, margin.left, width - margin.right);
    const y = scale(point.y, minY, maxY, height - margin.bottom, margin.top);
    ctx.fillStyle = data.colors[point.house];
    ctx.globalAlpha = 0.58;
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  drawLegend(ctx, width, data.houses, data.colors);
}

function drawLineChart(canvas, series, yLabel) {
  const { ctx, width, height } = fitCanvas(canvas);
  clear(ctx, width, height);
  const margin = { top: 18, right: 18, bottom: 42, left: 54 };
  const allPoints = Object.values(series).flat();
  const xs = allPoints.map((p) => p.epoch);
  const ys = allPoints.map((p) => p.value);
  const [minX, maxX] = bounds(xs.length ? xs : [0, 1]);
  const [minY, maxY] = bounds(ys.length ? ys : [0, 1]);
  drawAxes(ctx, width, height, margin, "epoch", yLabel);

  Object.entries(series).forEach(([house, points]) => {
    if (!points.length) return;
    ctx.strokeStyle = state.colors[house] || "#2563eb";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, i) => {
      const x = scale(point.epoch, minX, maxX, margin.left, width - margin.right);
      const y = scale(point.value, minY, maxY, height - margin.bottom, margin.top);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  drawLegend(ctx, width, houseOrder, state.colors);
}

function renderMetrics(summary) {
  const items = [
    ["Train samples", summary.trainRows],
    ["Test samples", summary.testRows],
    ["Features dataset", summary.features.length],
    ["Features model", summary.featuresForModel.length],
  ];
  $("#metrics").innerHTML = items.map(([label, value]) => `
    <div class="metric">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>
  `).join("");
}

function fillFeatureSelects(features) {
  const options = features.map((f) => `<option value="${f}">${f}</option>`).join("");
  $("#histFeature").innerHTML = options;
  $("#scatterX").innerHTML = options;
  $("#scatterY").innerHTML = options;
  if (features.length > 1) $("#scatterY").selectedIndex = 1;
}

async function loadDescribe() {
  const data = await api("/api/describe");
  const rows = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"];
  const head = `<tr><th>Stat</th>${data.features.map((f) => `<th>${f}</th>`).join("")}</tr>`;
  const body = rows.map((row) => `
    <tr>
      <td>${row}</td>
      ${data.features.map((f) => `<td>${formatNumber(data.stats[f][row], 4)}</td>`).join("")}
    </tr>
  `).join("");
  $("#describeTable").innerHTML = head + body;
}

async function loadHistogram() {
  const feature = encodeURIComponent($("#histFeature").value);
  const data = await api(`/api/histogram?feature=${feature}`);
  drawHistogram($("#histCanvas"), data);
}

async function loadScatter(auto = false) {
  let url = "/api/scatter";
  if (!auto) {
    const x = encodeURIComponent($("#scatterX").value);
    const y = encodeURIComponent($("#scatterY").value);
    url += `?x=${x}&y=${y}`;
  }
  const data = await api(url);
  $("#scatterTitle").textContent =
    `${data.xFeature} vs ${data.yFeature} | r = ${data.correlation.toFixed(4)}`
    + (data.auto ? " | selection automatique" : "");
  $("#scatterX").value = data.xFeature;
  $("#scatterY").value = data.yFeature;
  drawScatter($("#scatterCanvas"), data);
}

function miniBounds(samples, feature) {
  return bounds(samples.map((s) => s[feature]).filter((v) => v !== null));
}

function drawMiniScatter(canvas, samples, xFeature, yFeature, colors) {
  const { ctx, width, height } = fitCanvas(canvas);
  clear(ctx, width, height);
  const [minX, maxX] = miniBounds(samples, xFeature);
  const [minY, maxY] = miniBounds(samples, yFeature);
  samples.forEach((sample) => {
    if (sample[xFeature] === null || sample[yFeature] === null) return;
    const x = scale(sample[xFeature], minX, maxX, 6, width - 6);
    const y = scale(sample[yFeature], minY, maxY, height - 16, 6);
    ctx.fillStyle = colors[sample.house] || "#64748b";
    ctx.globalAlpha = 0.42;
    ctx.fillRect(x, y, 2, 2);
  });
  ctx.globalAlpha = 1;
}

function drawMiniHistogram(canvas, samples, feature, colors) {
  const { ctx, width, height } = fitCanvas(canvas);
  clear(ctx, width, height);
  const values = samples.map((s) => s[feature]).filter((v) => v !== null);
  const [min, max] = bounds(values);
  const bins = 14;
  const binW = (max - min) / bins || 1;
  const grouped = {};
  houseOrder.forEach((h) => grouped[h] = new Array(bins).fill(0));
  samples.forEach((sample) => {
    const value = sample[feature];
    if (value === null || !grouped[sample.house]) return;
    let idx = Math.floor((value - min) / binW);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    grouped[sample.house][idx] += 1;
  });
  const maxCount = Math.max(1, ...Object.values(grouped).flat());
  const groupW = width / bins;
  const barW = groupW / houseOrder.length;
  houseOrder.forEach((house, hi) => {
    ctx.fillStyle = colors[house];
    grouped[house].forEach((count, i) => {
      const h = (count / maxCount) * (height - 18);
      ctx.globalAlpha = 0.62;
      ctx.fillRect(i * groupW + hi * barW, height - 16 - h, barW, h);
    });
  });
  ctx.globalAlpha = 1;
}

async function loadPair() {
  const data = await api("/api/pair");
  const grid = $("#pairGrid");
  grid.className = "pair-wrap pair-grid";
  grid.style.gridTemplateColumns = `repeat(${data.features.length}, 118px)`;
  grid.innerHTML = "";
  data.features.forEach((rowFeature, i) => {
    data.features.forEach((colFeature, j) => {
      const cell = document.createElement("div");
      cell.className = "pair-cell" + (j > i ? " blank" : "");
      if (j <= i) {
        const canvas = document.createElement("canvas");
        cell.appendChild(canvas);
        const label = document.createElement("div");
        label.className = "pair-label";
        label.textContent = i === j ? rowFeature : `${colFeature} / ${rowFeature}`;
        cell.appendChild(label);
        requestAnimationFrame(() => {
          if (i === j) drawMiniHistogram(canvas, data.samples, rowFeature, data.colors);
          else drawMiniScatter(canvas, data.samples, colFeature, rowFeature, data.colors);
        });
      }
      grid.appendChild(cell);
    });
  });
}

function logLine(text) {
  const log = $("#trainLog");
  log.textContent += text + "\n";
  log.scrollTop = log.scrollHeight;
}

function setTrainState(text, mode = "idle") {
  const pill = $("#trainState");
  pill.textContent = text;
  pill.className = `pill ${mode}`;
}

function setProgress(selector, percent) {
  const node = $(selector);
  if (!node) return;
  const safe = Math.min(100, Math.max(0, percent));
  node.style.width = `${safe}%`;
}

function setTrainProgress(percent, label = null) {
  setProgress("#trainProgressBar", percent);
  $("#trainProgressText").textContent = label || `${Math.round(percent)}%`;
}

function renderHouseCards() {
  const cards = houseOrder.map((house) => `
    <div class="house-card" id="house-${house}" style="--house-color: ${state.colors[house] || "#2563eb"}">
      <div class="house-header">
        <div class="house-name">
          <span class="house-dot"></span>
          <span>${house}</span>
        </div>
        <span class="house-label" data-role="status">En attente</span>
      </div>
      <div class="house-values">
        <div class="house-metric">
          <span>Positifs</span>
          <strong data-role="positive">-</strong>
        </div>
        <div class="house-metric">
          <span>Negatifs</span>
          <strong data-role="negative">-</strong>
        </div>
        <div class="house-metric">
          <span>Loss</span>
          <strong data-role="loss">-</strong>
        </div>
        <div class="house-metric">
          <span>Accuracy</span>
          <strong data-role="accuracy">-</strong>
        </div>
      </div>
      <div class="progress">
        <div class="progress-bar" data-role="progress"></div>
      </div>
      <div class="house-foot">
        <span data-role="epoch">0 / ${state.trainMeta.totalEpochs}</span>
        <span data-role="bias">bias -</span>
      </div>
    </div>
  `).join("");
  $("#houseCards").innerHTML = cards;
}

function updateHouseCard(house, patch) {
  const card = $(`#house-${house}`);
  if (!card) return;
  if (patch.status) card.querySelector('[data-role="status"]').textContent = patch.status;
  if (patch.positive !== undefined) card.querySelector('[data-role="positive"]').textContent = patch.positive;
  if (patch.negative !== undefined) card.querySelector('[data-role="negative"]').textContent = patch.negative;
  if (patch.loss !== undefined) card.querySelector('[data-role="loss"]').textContent = patch.loss;
  if (patch.accuracy !== undefined) card.querySelector('[data-role="accuracy"]').textContent = patch.accuracy;
  if (patch.epoch !== undefined) card.querySelector('[data-role="epoch"]').textContent = patch.epoch;
  if (patch.bias !== undefined) card.querySelector('[data-role="bias"]').textContent = patch.bias;
  if (patch.progress !== undefined) {
    const safe = Math.min(100, Math.max(0, patch.progress));
    card.querySelector('[data-role="progress"]').style.width = `${safe}%`;
  }
  if (patch.className === "active") {
    card.classList.add("active");
    card.classList.remove("done");
  }
  if (patch.className === "done") {
    card.classList.remove("active");
    card.classList.add("done");
  }
  if (patch.className === "reset") {
    card.classList.remove("active", "done");
  }
}

function updateTrainOverview(event) {
  const meta = state.trainMeta;
  if (event.type === "preprocess") {
    meta.totalEpochs = event.epochs || meta.totalEpochs;
    $("#currentEpoch").textContent = `0 / ${meta.totalEpochs}`;
    renderHouseCards();
    return;
  }
  if (event.type === "house-start") {
    meta.currentHouse = event.house;
    meta.currentEpoch = 0;
    meta.latestLoss = null;
    meta.latestAccuracy = null;
  }
  if (event.type === "epoch") {
    meta.currentHouse = event.house;
    meta.currentEpoch = event.epoch;
    meta.totalEpochs = event.epochs;
    meta.latestLoss = event.loss;
    meta.latestAccuracy = event.accuracy;
  }
  $("#currentHouse").textContent = meta.currentHouse || "-";
  $("#currentEpoch").textContent = meta.currentHouse ? `${meta.currentEpoch} / ${meta.totalEpochs}` : "-";
  $("#currentLoss").textContent = meta.latestLoss === null ? "-" : meta.latestLoss.toFixed(6);
  $("#currentAccuracy").textContent = meta.latestAccuracy === null ? "-" : `${meta.latestAccuracy.toFixed(2)}%`;
}

function resetTrainingUi() {
  state.trainMeta = {
    completedHouses: 0,
    currentHouse: null,
    currentEpoch: 0,
    totalEpochs: 1000,
    latestLoss: null,
    latestAccuracy: null,
  };
  $("#currentHouse").textContent = "-";
  $("#currentEpoch").textContent = "-";
  $("#currentLoss").textContent = "-";
  $("#currentAccuracy").textContent = "-";
  setTrainProgress(0);
  renderHouseCards();
}

function resetTrainingCharts() {
  state.trainSeries = {};
  state.accSeries = {};
  houseOrder.forEach((house) => {
    state.trainSeries[house] = [];
    state.accSeries[house] = [];
  });
  drawLineChart($("#lossCanvas"), state.trainSeries, "loss");
  drawLineChart($("#accCanvas"), state.accSeries, "accuracy");
}

function startTraining() {
  $("#startTrain").disabled = true;
  setTrainState("En cours", "running");
  $("#trainLog").textContent = "";
  resetTrainingUi();
  resetTrainingCharts();

  const source = new EventSource("/api/train-stream");
  source.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "preprocess") {
      updateTrainOverview(event);
      logLine(`Loaded ${event.samples} samples, ${event.features.length} features`);
      logLine("Preprocessing: missing values -> mean, then z-score normalization");
    }
    if (event.type === "house-start") {
      all(".house-card").forEach((card) => card.classList.remove("active"));
      updateTrainOverview(event);
      updateHouseCard(event.house, {
        status: "En cours",
        positive: event.positive,
        negative: event.negative,
        progress: 0,
        epoch: `0 / ${state.trainMeta.totalEpochs}`,
        className: "active",
      });
      logLine(`Training ${event.house}: ${event.positive} positive / ${event.negative} negative`);
    }
    if (event.type === "epoch") {
      updateTrainOverview(event);
      state.trainSeries[event.house].push({ epoch: event.epoch, value: event.loss });
      state.accSeries[event.house].push({ epoch: event.epoch, value: event.accuracy });
      const houseProgress = (event.epoch / event.epochs) * 100;
      const overallProgress =
        ((state.trainMeta.completedHouses + event.epoch / event.epochs) / houseOrder.length) * 100;
      updateHouseCard(event.house, {
        status: "En cours",
        loss: event.loss.toFixed(6),
        accuracy: `${event.accuracy.toFixed(2)}%`,
        progress: houseProgress,
        epoch: `${event.epoch} / ${event.epochs}`,
      });
      setTrainProgress(overallProgress, `${Math.round(overallProgress)}%`);
      if (event.epoch === 1 || event.epoch % 100 === 0 || event.epoch === event.epochs) {
        logLine(`${event.house} epoch ${event.epoch}/${event.epochs} - loss ${event.loss.toFixed(6)} - acc ${event.accuracy.toFixed(2)}%`);
      }
      drawLineChart($("#lossCanvas"), state.trainSeries, "loss");
      drawLineChart($("#accCanvas"), state.accSeries, "accuracy");
    }
    if (event.type === "house-end") {
      state.trainMeta.completedHouses += 1;
      updateHouseCard(event.house, {
        status: "Termine",
        progress: 100,
        bias: `bias ${event.bias.toFixed(6)}`,
        className: "done",
      });
      const overallProgress = (state.trainMeta.completedHouses / houseOrder.length) * 100;
      setTrainProgress(overallProgress, `${Math.round(overallProgress)}%`);
      logLine(`Finished ${event.house}, learned bias ${event.bias.toFixed(6)}`);
    }
    if (event.type === "done") {
      setTrainProgress(100, "100%");
      logLine(`Training accuracy: ${event.correct}/${event.total} (${event.accuracy.toFixed(2)}%)`);
      logLine(`Model saved to ${event.weightsPath}`);
      setTrainState(`Termine - ${event.accuracy.toFixed(2)}%`, "done");
      $("#startTrain").disabled = false;
      source.close();
    }
  };
  source.onerror = () => {
    setTrainState("Erreur ou connexion fermee", "error");
    $("#startTrain").disabled = false;
    source.close();
  };
}

async function runPredict() {
  $("#predictState").textContent = "Prediction en cours...";
  const data = await api("/api/predict");
  if (data.error) {
    $("#predictState").textContent = data.error;
    return;
  }
  $("#predictState").textContent = `${data.rows} predictions -> ${data.output}`;
  drawBarChart(
    $("#predictChart"),
    houseOrder,
    houseOrder.map((h) => data.counts[h] || 0),
    houseOrder.map((h) => state.colors[h])
  );
  const head = "<tr><th>Index</th><th>House</th><th>Best probability</th></tr>";
  const body = data.preview.map((row) => `
    <tr>
      <td>${row.index}</td>
      <td>${row.house}</td>
      <td>${formatNumber(row.probability, 4)}</td>
    </tr>
  `).join("");
  $("#predictTable").innerHTML = head + body;
}

function setupNavigation() {
  all(".nav").forEach((button) => {
    button.addEventListener("click", () => {
      all(".nav").forEach((b) => b.classList.remove("active"));
      all(".view").forEach((view) => view.classList.remove("active"));
      button.classList.add("active");
      $(`#view-${button.dataset.view}`).classList.add("active");
      setTimeout(redrawVisible, 20);
    });
  });
}

function redrawVisible() {
  if ($("#view-overview").classList.contains("active") && state.summary) {
    drawBarChart(
      $("#houseChart"),
      houseOrder,
      houseOrder.map((h) => state.summary.houseCounts[h] || 0),
      houseOrder.map((h) => state.colors[h])
    );
  }
  if ($("#view-histogram").classList.contains("active")) loadHistogram();
  if ($("#view-scatter").classList.contains("active")) loadScatter(false);
  if ($("#view-train").classList.contains("active")) {
    drawLineChart($("#lossCanvas"), state.trainSeries, "loss");
    drawLineChart($("#accCanvas"), state.accSeries, "accuracy");
  }
}

async function init() {
  setupNavigation();
  state.summary = await api("/api/summary");
  state.features = state.summary.features;
  state.colors = state.summary.colors;
  $("#status").textContent = state.summary.hasWeights ? "Modele disponible" : "Modele a entrainer";
  renderMetrics(state.summary);
  fillFeatureSelects(state.features);
  drawBarChart(
    $("#houseChart"),
    houseOrder,
    houseOrder.map((h) => state.summary.houseCounts[h] || 0),
    houseOrder.map((h) => state.colors[h])
  );
  await loadDescribe();
  await loadHistogram();
  await loadScatter(true);
  await loadPair();
  resetTrainingUi();
  resetTrainingCharts();

  $("#histFeature").addEventListener("change", loadHistogram);
  $("#autoScatter").addEventListener("click", () => loadScatter(true));
  $("#showScatter").addEventListener("click", () => loadScatter(false));
  $("#startTrain").addEventListener("click", startTraining);
  $("#runPredict").addEventListener("click", runPredict);
  window.addEventListener("resize", () => setTimeout(redrawVisible, 80));
}

init().catch((error) => {
  $("#status").textContent = "Erreur";
  console.error(error);
});
