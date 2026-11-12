const $ = (s) => document.querySelector(s);
const fmt = (n) => Math.round(n).toLocaleString("en-US");

const TITLES = {
  dashboard: ["Demand overview", "Live multi-day ticket, occupancy, staffing and concession forecasts."],
  forecast: ["Forecast a showing", "Estimate tickets, uncertainty range and operations plan for a single show."],
  models: ["Model comparison", "Evaluation metrics and feature importances from your trained Random Forest."],
};

const STATE = {
  view: "dashboard",
  hiddenSeries: new Set(),
  forecastData: null,
  metricsData: null,
  chartIdx: null,
};

const SERIES_META = [
  { k: "tickets", n: "Random Forest", c: "#e0b155" },
  { k: "baseline", n: "Baseline", c: "#8aa0bd" },
];

function debounce(fn, ms = 200) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

async function apiGet(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function occStatus(occ) {
  if (occ >= 95) return { label: "Sell-out risk", cls: "down" };
  if (occ >= 75) return { label: "Near sell-out", cls: "warn" };
  if (occ >= 40) return { label: "Healthy", cls: "up" };
  return { label: "Under-utilised", cls: "warn" };
}

/* ── Safely read input values from DOM ───────────────────────────────────── */
function readInt(id, fallback = 1) {
  const el = document.getElementById(id);
  if (!el) return fallback;
  const v = parseInt(el.value, 10);
  return isNaN(v) ? fallback : v;
}

function readFloat(id, fallback = 1) {
  const el = document.getElementById(id);
  if (!el) return fallback;
  const v = parseFloat(el.value);
  return isNaN(v) ? fallback : v;
}

function readStr(id, fallback = "") {
  const el = document.getElementById(id);
  if (!el || !el.value) return fallback;
  return el.value;
}

/* ── Dynamic defaults loaded from model pipeline ────────────────────────── */
let DEFAULTS = {
  film_code: 1493,
  cinema_code: 524,
  show_time: 2,
  capacity: 532,
  ticket_price: 78280,
};

async function init() {
  const today = todayISO();
  const fdateEl = document.getElementById("fdate");
  if (fdateEl) fdateEl.value = today;

  const pdEl = document.getElementById("pd");
  if (pdEl) pdEl.value = today;

  // // 1. Health check & model version
  // try {
  //   const health = await apiGet("/health");
  //   $("#apiStatus").innerHTML = `<span class="up">Live</span> · ${health.model_version} (${health.n_estimators} trees)`;
  // } catch {
  //   $("#apiStatus").innerHTML = `<span class="down">● Offline</span> · Cannot reach API`;
  // }

  // 2. Fetch options directly from trained model categories
  try {
    const opts = await apiGet("/options");
    DEFAULTS = opts.defaults;

    fillSelect("fm", opts.movies, "Movie #", opts.defaults.film_code);
    fillSelect("fc", opts.screens, "Screen #", opts.defaults.cinema_code);
    fillSelect("fst", opts.showtimes, "Slot ", opts.defaults.show_time);

    fillSelect("pm", opts.movies, "Movie #", opts.defaults.film_code, true);
    fillSelect("pc", opts.screens, "Screen #", opts.defaults.cinema_code);
    fillSelect("pt", opts.showtimes, "Slot ", opts.defaults.show_time);

    setInputVal("fcap", opts.defaults.capacity);
    setInputVal("fprice", opts.defaults.ticket_price);

    setInputVal("ps", opts.defaults.capacity);
    setInputVal("pp", opts.defaults.ticket_price);
  } catch (err) {
    const kpis = document.getElementById("kpis");
    if (kpis) kpis.innerHTML = `<div class="card down">Failed to load model options: ${err.message}</div>`;
  }

  // 3. Bind real-time reactive updates on Dashboard filter change
  const debouncedDashboard = debounce(loadDashboard, 150);
  ["fm", "fc", "fst", "fdate", "fh", "fcap", "fprice"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", () => loadDashboard());
      el.addEventListener("input", debouncedDashboard);
    }
  });

  // 4. "New movie" toggle for single show forecast
  const pmEl = document.getElementById("pm");
  if (pmEl) {
    pmEl.addEventListener("change", () => {
      const wrap = document.getElementById("customMovieWrap");
      if (wrap) wrap.classList.toggle("hidden", pmEl.value !== "new");
      handleForecastSubmit();
    });
  }

  // 5. Bind real-time reactive updates on Forecast form inputs
  const debouncedForecast = debounce(handleForecastSubmit, 150);
  ["pm", "pmCustom", "pc", "pd", "pt", "ps", "pp"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", () => handleForecastSubmit());
      el.addEventListener("input", debouncedForecast);
    }
  });

  const form = document.getElementById("form");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      handleForecastSubmit();
    });
  }

  const goBtn = document.getElementById("go");
  if (goBtn) {
    goBtn.addEventListener("click", (e) => {
      e.preventDefault();
      handleForecastSubmit();
    });
  }

  route();

  // Trigger initial real-time predictions right away
  loadDashboard();
  handleForecastSubmit();
}

function setInputVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function fillSelect(id, items, prefix, selectedVal, includeNew = false) {
  const el = document.getElementById(id);
  if (!el) return;
  let html = items
    .map((v) => `<option value="${v}" ${Number(v) === Number(selectedVal) ? "selected" : ""}>${prefix}${v}</option>`)
    .join("");
  if (includeNew) {
    html += `<option value="new">+ New movie (cold-start)</option>`;
  }
  el.innerHTML = html;
}

/* ── Routing ────────────────────────────────────────────────────────────── */
function route() {
  const hash = (location.hash || "#dashboard").slice(1);
  STATE.view = TITLES[hash] ? hash : "dashboard";

  document.querySelectorAll("nav a").forEach((a) => {
    if (a.getAttribute("href") === "#" + STATE.view) {
      a.setAttribute("aria-current", "page");
    } else {
      a.removeAttribute("aria-current");
    }
  });

  document.querySelectorAll(".vis").forEach((sec) => {
    sec.classList.toggle("on", sec.id === "v-" + STATE.view);
  });

  const titleEl = document.getElementById("title");
  const subtitleEl = document.getElementById("subtitle");
  const filtersEl = document.getElementById("filters");

  if (titleEl) titleEl.textContent = TITLES[STATE.view][0];
  if (subtitleEl) subtitleEl.textContent = TITLES[STATE.view][1];
  if (filtersEl) filtersEl.style.display = STATE.view === "dashboard" ? "flex" : "none";

  if (STATE.view === "dashboard") loadDashboard();
  if (STATE.view === "forecast") handleForecastSubmit();
  if (STATE.view === "models") loadModels();
}
addEventListener("hashchange", route);

/* ── Dashboard (Multi-day rolling forecast) ──────────────────────────────── */
async function loadDashboard() {
  const payload = {
    film_code: readInt("fm", DEFAULTS.film_code),
    cinema_code: readInt("fc", DEFAULTS.cinema_code),
    show_time: readInt("fst", DEFAULTS.show_time),
    capacity: Math.max(10, readFloat("fcap", DEFAULTS.capacity)),
    ticket_price: Math.max(1, readFloat("fprice", DEFAULTS.ticket_price)),
    forecast_date: readStr("fdate", todayISO()),
    days: readInt("fh", 7),
  };

  const kpis = document.getElementById("kpis");
  if (kpis && !STATE.forecastData) {
    kpis.innerHTML = `<div class="skel"></div><div class="skel"></div><div class="skel"></div><div class="skel"></div>`;
  }

  try {
    STATE.forecastData = await apiPost("/forecast", payload);
    renderDashboard(STATE.forecastData, payload.days);
  } catch (err) {
    if (kpis) kpis.innerHTML = `<div class="card down">Live forecast error: ${err.message}</div>`;
  }
}

function renderDashboard(data, days) {
  const tf = data.total_food;
  const occBadge = occStatus(data.avg_occ);

  const kpis = document.getElementById("kpis");
  if (kpis) {
    kpis.innerHTML = [
      [
        `Tickets, next ${days} days`,
        fmt(data.total_tickets),
        `<span class="chip up">Peak: ${data.peak.day.slice(0, 3)} (${fmt(data.peak.tickets)})</span>`,
      ],
      [
        "Projected occupancy",
        `${Math.round(data.avg_occ)}%`,
        `<span class="chip ${occBadge.cls}">${occBadge.label}</span> across period`,
      ],
      [
        "Projected box office",
        fmt(data.total_revenue),
        `Estimated revenue at current ticket price`,
      ],
      [
        "Concessions to stock",
        fmt(tf.popcorn + tf.drinks + tf.snacks),
        `🍿 ${fmt(tf.popcorn)} · 🥤 ${fmt(tf.drinks)} · 🍬 ${fmt(tf.snacks)}`,
      ],
    ]
      .map(
        ([lab, val, note]) =>
          `<div class="card kpi"><div class="lab">${lab}</div><div class="val">${val}</div><div class="note">${note}</div></div>`
      )
      .join("");
  }

  renderLegend();
  renderDemandChart(data.forecast);
  renderWeekdayOccupancy(data.forecast);
  renderOpsTable(data.forecast);
}

function renderLegend() {
  const legendEl = document.getElementById("legend");
  if (!legendEl) return;
  legendEl.innerHTML = SERIES_META.map(
    (m) =>
      `<button type="button" aria-pressed="${!STATE.hiddenSeries.has(m.k)}" data-k="${m.k}"><i style="background:${m.c}"></i>${m.n}</button>`
  ).join("");

  document.querySelectorAll("#legend button").forEach((btn) => {
    btn.onclick = () => {
      const k = btn.dataset.k;
      if (STATE.hiddenSeries.has(k)) STATE.hiddenSeries.delete(k);
      else STATE.hiddenSeries.add(k);
      renderLegend();
      if (STATE.forecastData) renderDemandChart(STATE.forecastData.forecast);
    };
  });
}

function renderDemandChart(rows) {
  const cwEl = document.getElementById("cw");
  if (!cwEl) return;

  const W = 760, H = 290, p = { l: 48, r: 16, t: 16, b: 28 };
  const vis = SERIES_META.filter((m) => !STATE.hiddenSeries.has(m.k));
  const maxVal = Math.max(
    10,
    ...rows.map((r) => Math.max(r.interval_high || r.tickets, r.baseline || 0))
  ) * 1.12;
  const step = Math.pow(10, Math.floor(Math.log10(Math.max(1, maxVal / 4))));
  const ty = Math.ceil(maxVal / 4 / step) * step;
  const top = ty * 4;

  const X = (i) => p.l + (rows.length === 1 ? (W - p.l - p.r) / 2 : (i * (W - p.l - p.r)) / (rows.length - 1));
  const Y = (v) => p.t + (1 - v / top) * (H - p.t - p.b);

  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const y = Y(ty * i);
    grid += `<line x1="${p.l}" x2="${W - p.r}" y1="${y}" y2="${y}" stroke="#3a2832" stroke-dasharray="3 4"/>`;
    grid += `<text x="${p.l - 8}" y="${y + 4}" text-anchor="end">${fmt(ty * i)}</text>`;
  }

  const labelEvery = Math.max(1, Math.ceil(rows.length / 8));
  rows.forEach((r, i) => {
    if (i % labelEvery === 0) {
      grid += `<text x="${X(i)}" y="${H - 8}" text-anchor="middle">${r.day.slice(0, 3)} ${r.date.slice(5)}</text>`;
    }
  });

  // Shaded 10–90% uncertainty band from all 200 trees
  let bandPath = "";
  if (!STATE.hiddenSeries.has("tickets") && rows.length > 1) {
    const upper = rows.map((r, i) => `${i === 0 ? "M" : "L"}${X(i).toFixed(1)} ${Y(r.interval_high).toFixed(1)}`).join(" ");
    const lower = rows
      .slice()
      .reverse()
      .map((r, idx) => `L${X(rows.length - 1 - idx).toFixed(1)} ${Y(r.interval_low).toFixed(1)}`)
      .join(" ");
    bandPath = `<path d="${upper} ${lower} Z" fill="#e0b155" opacity="0.14"/>`;
  }

  const paths = vis
    .map((m) => {
      const d = rows
        .map((r, i) => `${i === 0 ? "M" : "L"}${X(i).toFixed(1)} ${Y(r[m.k]).toFixed(1)}`)
        .join(" ");
      return `<path d="${d}" fill="none" stroke="${m.c}" stroke-width="${m.k === "tickets" ? 2.6 : 2}" ${
        m.k === "baseline" ? 'stroke-dasharray="6 4"' : ""
      }/>`;
    })
    .join("");

  cwEl.innerHTML = `
    <svg id="svg" viewBox="0 0 ${W} ${H}" tabindex="0" role="img" aria-label="Ticket demand forecast chart. Use arrow keys to inspect values.">
      <g class="axis">${grid}</g>
      ${bandPath}
      ${paths}
      <line id="cur" y1="${p.t}" y2="${H - p.b}" stroke="#f2e9da" opacity="0"/>
      <g id="dots"></g>
    </svg>
    <div class="tip" id="tip"></div>
  `;

  const svg = document.getElementById("svg");
  const tip = document.getElementById("tip");
  const liveEl = document.getElementById("live");

  function show(i) {
    i = Math.max(0, Math.min(rows.length - 1, i));
    const r = rows[i];
    const x = X(i);
    const curEl = document.getElementById("cur");
    const dotsEl = document.getElementById("dots");
    if (curEl) { curEl.setAttribute("x1", x); curEl.setAttribute("x2", x); curEl.setAttribute("opacity", 0.35); }
    if (dotsEl) dotsEl.innerHTML = vis.map((m) => `<circle cx="${x}" cy="${Y(r[m.k])}" r="4" fill="${m.c}"/>`).join("");
    if (tip) {
      tip.innerHTML = `<b>${r.day.slice(0, 3)} ${r.date.slice(5)}</b>
        <div><span style="color:#e0b155">RF Forecast</span><b>${fmt(r.tickets)}</b></div>
        <div><span style="color:#a99a97">Likely Range</span><b>${fmt(r.interval_low)}–${fmt(r.interval_high)}</b></div>
        <div><span style="color:#8aa0bd">Baseline</span><b>${fmt(r.baseline)}</b></div>`;
      tip.style.left = (x / W) * 100 + "%";
      tip.style.top = (Y(r.tickets) / H) * 100 + "%";
      tip.style.opacity = 1;
    }
    if (liveEl) liveEl.textContent = `${r.date} (${r.day}): ${fmt(r.tickets)} tickets (range ${fmt(r.interval_low)}–${fmt(r.interval_high)}), ${Math.round(r.occ_pct)}% occupancy.`;
    STATE.chartIdx = i;
  }

  if (svg) {
    svg.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * W;
      const idx = Math.round(((x - p.l) / Math.max(1, W - p.l - p.r)) * (rows.length - 1));
      show(idx);
    });
    svg.addEventListener("pointerleave", () => {
      if (tip) tip.style.opacity = 0;
      const curEl = document.getElementById("cur");
      const dotsEl = document.getElementById("dots");
      if (curEl) curEl.setAttribute("opacity", 0);
      if (dotsEl) dotsEl.innerHTML = "";
    });
    svg.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        show((STATE.chartIdx ?? 0) + (e.key === "ArrowRight" ? 1 : -1));
      }
    });
  }
}

function renderWeekdayOccupancy(rows) {
  const occEl = document.getElementById("occ");
  if (!occEl) return;

  const order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const sums = Object.fromEntries(order.map((d) => [d, { total: 0, n: 0 }]));
  rows.forEach((r) => {
    if (sums[r.day]) { sums[r.day].total += r.occ_pct; sums[r.day].n += 1; }
  });

  const vals = order.map((d) => (sums[d].n ? sums[d].total / sums[d].n : 0));
  const top = Math.max(20, Math.ceil(Math.max(...vals) / 10) * 10);
  const W = 420, H = 290, p = { l: 38, r: 8, t: 16, b: 28 };
  const Y = (v) => p.t + (1 - v / top) * (H - p.t - p.b);
  const bw = (W - p.l - p.r) / 7;

  let s = "";
  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i;
    s += `<line x1="${p.l}" x2="${W - p.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#3a2832" stroke-dasharray="3 4"/>`;
    s += `<text x="${p.l - 6}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v)}%</text>`;
  }

  order.forEach((d, i) => {
    const x = p.l + i * bw + bw * 0.2;
    const w = bw * 0.6;
    const v = vals[i];
    const fill = i >= 5 ? "#e0b155" : "#8aa0bd";
    s += `<text x="${p.l + i * bw + bw / 2}" y="${H - 8}" text-anchor="middle">${d.slice(0, 3)}</text>`;
    s += `<rect x="${x}" y="${Y(v)}" width="${w}" height="${Math.max(0, H - p.b - Y(v))}" rx="4" fill="${fill}"><title>${d}: ${v.toFixed(1)}%</title></rect>`;
  });

  occEl.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Projected occupancy by weekday"><g class="axis">${s}</g></svg>`;
}

function renderOpsTable(rows) {
  const tableEl = document.getElementById("opsTable");
  if (!tableEl) return;

  tableEl.innerHTML = `
    <thead>
      <tr>
        <th>Date</th><th>Day</th><th>Tickets (Range)</th><th>Occupancy</th>
        <th>Shows</th><th>Staffing</th><th>🍿</th><th>🥤</th><th>🍬</th><th>🎁</th><th>Recommended Action</th>
      </tr>
    </thead>
    <tbody>
      ${rows
        .map((r) => {
          const st = occStatus(r.occ_pct);
          return `
            <tr>
              <td>${r.date}</td>
              <td>${r.day.slice(0, 3)}</td>
              <td><b>${fmt(r.tickets)}</b> <span style="color:var(--mute)">(${fmt(r.interval_low)}–${fmt(r.interval_high)})</span></td>
              <td class="${st.cls}">${Math.round(r.occ_pct)}%</td>
              <td>${r.schedule.shows}</td>
              <td>${r.schedule.staff}</td>
              <td>${r.food.popcorn}</td>
              <td>${r.food.drinks}</td>
              <td>${r.food.snacks}</td>
              <td>${r.food.combos}</td>
              <td>${r.schedule.action}</td>
            </tr>`;
        })
        .join("")}
    </tbody>`;
}

/* ── Single Showing Forecast ────────────────────────────────────────────── */
async function handleForecastSubmit() {
  const pmEl = document.getElementById("pm");
  const filmIsNew = pmEl && pmEl.value === "new";
  const filmCode = filmIsNew
    ? readInt("pmCustom", 99999)
    : readInt("pm", DEFAULTS.film_code);

  const payload = {
    film_code: filmCode,
    cinema_code: readInt("pc", DEFAULTS.cinema_code),
    show_time: readInt("pt", DEFAULTS.show_time),
    capacity: Math.max(10, readFloat("ps", DEFAULTS.capacity)),
    ticket_price: Math.max(1, readFloat("pp", DEFAULTS.ticket_price)),
    forecast_date: readStr("pd", todayISO()),
  };

  const out = document.getElementById("out");
  const goBtn = document.getElementById("go");

  if (goBtn) goBtn.disabled = true;

  try {
    const res = await apiPost("/predict", payload);
    renderSingleForecast(res, payload);
  } catch (err) {
    if (out) out.innerHTML = `<div class="card placeholder down">Live prediction failed: ${err.message}</div>`;
  } finally {
    if (goBtn) goBtn.disabled = false;
  }
}

function renderSingleForecast(r, req) {
  const out = document.getElementById("out");
  if (!out) return;

  const st = occStatus(r.predicted_occupancy);
  const lowPct = Math.min(100, (r.interval_low / r.capacity) * 100);
  const highPct = Math.min(100, (r.interval_high / r.capacity) * 100);
  const occPct = Math.min(100, r.predicted_occupancy);
  const delta = r.predicted_tickets - r.baseline_prediction;
  const maxAbsContrib = Math.max(1, ...r.top_features.map((f) => Math.abs(f.contribution)));

  const coldBanner = r.cold_start
    ? `<div class="banner"><strong>Cold-start movie/screen:</strong> Limited history for this code in training data. Treat as low-confidence.</div>`
    : "";

  const contribHtml = r.top_features
    .map((f) => {
      const w = Math.max(3, (Math.abs(f.contribution) / maxAbsContrib) * 100);
      const color = f.contribution >= 0 ? "var(--ok)" : "var(--bad)";
      const sign = f.contribution > 0 ? "+" : "";
      return `
        <div class="bar">
          <span>${f.name}</span>
          <span class="t"><i style="width:${w.toFixed(0)}%;background:${color}"></i></span>
          <b class="${f.contribution >= 0 ? "up" : "down"}">${sign}${f.contribution}</b>
        </div>`;
    })
    .join("");

  const movieLabel = req.film_code === 99999 ? "New (cold-start)" : `Movie #${req.film_code}`;
  const nowTime = new Date().toLocaleTimeString();

  out.innerHTML = `
    ${coldBanner}
    <div style="display:flex;justify-content:space-between;align-items:center;margin:0 0 12px;font-size:13px;color:var(--mute)">
      <span>Live forecast for <b style="color:var(--text)">${movieLabel}</b> · Screen #${req.cinema_code} · Slot ${req.show_time} · ${req.forecast_date} (${r.day}) · ${fmt(req.capacity)} seats</span>
      <span class="chip up">🟢 Updated ${nowTime} (${r.latency_ms} ms)</span>
    </div>
    <div class="results">
      <div class="card res rec">
        <div class="tag">Random Forest · ${r.model_version}</div>
        <div class="big">${fmt(r.predicted_tickets)} <span style="font:15px var(--sans);color:var(--mute)">tickets</span></div>
        <div>Likely range: <b>${fmt(r.interval_low)}–${fmt(r.interval_high)}</b></div>
        <div class="${st.cls}" style="margin-top:6px">${Math.round(r.predicted_occupancy)}% occupancy · ${st.label}</div>
        <div class="gauge">
          <div class="gauge-track" aria-label="Occupancy gauge ${Math.round(r.predicted_occupancy)} percent">
            <div class="gauge-band" style="left:${lowPct}%;width:${Math.max(2, highPct - lowPct)}%"></div>
            <div class="gauge-fill" style="width:${occPct}%"></div>
          </div>
        </div>
      </div>

      <div class="card res">
        <div class="tag" style="color:var(--slate)">Simple Baseline Comparison</div>
        <div class="big">${fmt(r.baseline_prediction)} <span style="font:15px var(--sans);color:var(--mute)">tickets</span></div>
        <div>Difference vs baseline: <b class="${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "+" : ""}${fmt(delta)}</b></div>
        <div style="color:var(--mute);font-size:13px;margin-top:8px">Baseline scales median show attendance to ${fmt(req.capacity)} hall seats.</div>
      </div>

      <div class="card res">
        <div class="tag">Operations &amp; Concessions · Human Review Required</div>
        <div style="margin-top:8px"><b>Shows:</b> ${r.schedule.shows} &nbsp;|&nbsp; <b>Staff:</b> ${r.schedule.staff}</div>
        <div style="margin-top:6px">🍿 Popcorn ${r.food.popcorn} · 🥤 Drinks ${r.food.drinks} · 🍬 Snacks ${r.food.snacks} · 🎁 Combos ${r.food.combos}</div>
        <div style="color:var(--mute);font-size:13px;margin-top:8px">${r.suggested_action}</div>
      </div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="head"><h2>Why this forecast?</h2><span style="font-size:13px;color:var(--mute)">Computed across all 200 trees · ${r.latency_ms} ms</span></div>
      <p style="margin:0 0 12px">${r.explanation}</p>
      <div class="bars">${contribHtml}</div>
    </div>

    <details class="card tech">
      <summary>Technical details (real-time payload from model artifact)</summary>
      <pre>${JSON.stringify({ request: req, response: r }, null, 2)}</pre>
    </details>
  `;
}

/* ── Model Comparison ───────────────────────────────────────────────────── */
async function loadModels() {
  if (STATE.metricsData) return;
  try {
    STATE.metricsData = await apiGet("/metrics");
    renderModels(STATE.metricsData);
  } catch (err) {
    const mt = document.getElementById("mt");
    if (mt) mt.innerHTML = `<caption>Failed to load metrics: ${err.message}</caption>`;
  }
}

function renderModels(data) {
  const mt = document.getElementById("mt");
  const mb = document.getElementById("mb");
  const mn = document.getElementById("mn");
  if (!mt) return;

  const bestMae = Math.min(...data.models.map((m) => m.mae));
  const bestWape = Math.min(...data.models.map((m) => m.wape));

  mt.innerHTML = `
    <caption>Lower is better for MAE, RMSE, and WAPE. ★ marks the deployed model.</caption>
    <thead>
      <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>WAPE</th></tr>
    </thead>
    <tbody>
      ${data.models
        .map(
          (m) => `
          <tr class="${m.is_deployed ? "best" : ""}">
            <td>${m.model}${m.is_deployed ? " ★" : ""}</td>
            <td>${m.mae.toFixed(2)}${m.mae === bestMae ? " (best)" : ""}</td>
            <td>${m.rmse.toFixed(2)}</td>
            <td>${m.wape.toFixed(2)}%${m.wape === bestWape ? " (best)" : ""}</td>
          </tr>`
        )
        .join("")}
    </tbody>`;

  if (mb) {
    const maxImp = Math.max(1, ...data.feature_importances.map((f) => f.importance_pct));
    mb.innerHTML = data.feature_importances
      .slice(0, 7)
      .map(
        (f, idx) => `
        <div class="bar">
          <span>${f.feature}</span>
          <span class="t"><i style="width:${((f.importance_pct / maxImp) * 100).toFixed(0)}%;background:${idx === 0 ? "var(--brass)" : "var(--slate)"}"></i></span>
          <b>${f.importance_pct.toFixed(1)}%</b>
        </div>`
      )
      .join("");
  }

  if (mn) {
    const cs = data.cold_start_info;
    mn.textContent = `Trained vocabulary: ${cs.known_movies_count} movies, ${cs.known_screens_count} cinemas, ${cs.known_showtimes_count} showtime slots.`;
  }
}

init();
