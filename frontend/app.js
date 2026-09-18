/* Интерфейс весовой, две роли:
   Весовщик — чат с ИИ (снимок с камеры, вес с весов по COM-порту, водитель и груз из путевого листа → заезд/выезд);
   Руководитель — машины и карта складов (рейсы, тревоги, события с камер). */
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { vehicles: [], warehouses: [], attachment: null, busy: false };

// ---------- утилиты ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(⚠[^\n]*)/g, '<span class="alert">$1</span>');
const t = (v) => (v == null ? "—" : `${Number(v).toFixed(2)} т`);
const dash = (v) => (v == null || v === "" ? "—" : esc(v));
// 1 машина, 3 машины, 6 машин, 21 машина, 12 машин
const plural = (n, one, few, many) => {
  const d = n % 10, dd = n % 100;
  return d === 1 && dd !== 11 ? one : d >= 2 && d <= 4 && (dd < 12 || dd > 14) ? few : many;
};
const api = async (url, opts = {}) => {
  const res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
};
const lightbox = (src) => { const lb = $("#lightbox"); lb.querySelector("img").src = src; lb.classList.remove("hidden"); };
$("#lightbox").onclick = () => $("#lightbox").classList.add("hidden");
document.addEventListener("click", (e) => { if (e.target.matches("img.zoom, img.shot, img.photo, .event img, .hero img")) lightbox(e.target.dataset.full || e.target.src); });

// ---------- роли: весовщик (чат) / руководитель (карта и машины) ----------
const ROLES = { weigher: "Весовщик", manager: "Руководитель" };

function getRole() {
  try { return localStorage.getItem("role") || ""; } catch { return ""; }
}

function applyRole(role) {
  if (!ROLES[role]) { $("#role-screen").classList.remove("hidden"); return; }
  try { localStorage.setItem("role", role); } catch { /* приватный режим — роль живёт до перезагрузки */ }
  document.body.classList.remove("role-weigher", "role-manager");
  document.body.classList.add(`role-${role}`);
  $("#role-label").textContent = ROLES[role];
  $("#role-screen").classList.add("hidden");
  // Leaflet должен пересчитать размер контейнера после смены раскладки (или создаться, если карты ещё нет).
  setTimeout(() => {
    if (map.obj) { map.obj.invalidateSize(); map.fitted = false; }
    renderWarehouses();
  }, 60);
}
document.querySelectorAll(".role-option").forEach((b) => (b.onclick = () => applyRole(b.dataset.role)));
$("#btn-role").onclick = () => $("#role-screen").classList.remove("hidden");

// ---------- состояние ----------
async function refreshState() {
  const s = await api("/api/state");
  Object.assign(state, { vehicles: s.vehicles, warehouses: s.warehouses, scale: s.scale });
  renderVehicles();
  renderWarehouses();
  renderWarehouseSelect();
  if (s.scale) $("#scale-port").textContent = s.scale.port;
}

// ---------- весы на COM-порту (ИМИТАЦИЯ: реального терминала нет, показания придумываются) ----------
$("#scale-badge").onclick = () => {
  const sc = state.scale || { port: "COM3", baud: 9600, frame: "8N1" };
  const last = sc.last;
  openModal(`
    <h2>⚖ Весы · ${esc(sc.port)}</h2>
    <div class="muted">Весовой терминал подключён к компьютеру по COM-порту. Вес машины снимается автоматически,
      когда она стоит на платформе и показание стабилизировалось, — весовщику вводить вес не нужно.</div>
    <div class="kv" style="margin-top:12px">
      <div>Порт</div><div><b>${esc(sc.port)}</b> · ${esc(sc.baud)} бод, ${esc(sc.frame)}</div>
      <div>Состояние</div><div><span class="dot green"></span>подключены, данные идут</div>
      <div>Последнее показание</div><div>${last ? `<b>${t(last.weight_t)}</b> · ${last.stable ? "стабильно" : "нестабильно"} · ${esc(last.at)} · ${last.kind === "exit" ? "выезд (тара)" : "заезд (брутто)"}${last.vehicle_id ? ` · машина ${esc(vehicleLabel(last.vehicle_id))}` : ""}` : "ещё не было"}</div>
    </div>
    <div class="scale-note">⚠ Имитация для демо: настоящего весового терминала нет, показания придумываются правдоподобно по типу техники
      (у каждой машины своя тара, брутто зависит от груза). В отчётах вес подписан «весы ${esc(sc.port)} (имитация показаний)».</div>`);
};

// ---------- машины ----------
// Кириллица, похожая на латиницу (в номерах пишут и так, и так), верхний регистр, без пробелов.
const LOOKALIKE = { А: "A", В: "B", Е: "E", К: "K", М: "M", Н: "H", О: "O", Р: "P", С: "C", Т: "T", Х: "X", У: "Y" };
const norm = (s) => String(s ?? "").toUpperCase().replace(/\s+/g, "").replace(/[АВЕКМНОРСТХУ]/g, (c) => LOOKALIKE[c]);
const vehicleLabel = (id, fallback) => state.vehicles.find((v) => String(v.id) === String(id))?.label || fallback || `№${id}`;

function renderVehicles() {
  const q = $("#search").value.trim();
  const byId = q.match(/^№\s*(\d+)$/);
  const nq = norm(q);
  const list = state.vehicles.filter((v) => {
    if (!q) return true;
    if (byId) return String(v.id) === byId[1];  // «№12» — ровно машина 12, а не 1, 12, 112…
    return String(v.id) === q || [v.plate, v.plate_formatted, v.label].some((x) => norm(x).includes(nq));
  });
  $("#vehicle-count").textContent = `${state.vehicles.length} ед.`;
  if (!list.length) { $("#vehicle-list").innerHTML = '<div class="empty">Пока нет машин. Весовщик загружает фото машины в чат.</div>'; return; }
  $("#vehicle-list").innerHTML = list.map((v) => `
    <div class="vehicle" data-id="${esc(v.id)}">
      <div>
        <div class="vid ${v.is_temporary ? "tmp" : ""}"><span class="no">№${esc(v.id)}</span> ${esc((v.label || "").replace(/^№\S+ · /, ""))}</div>
        <div class="sub">${esc(v.equipment_type || "")}${v.plate_formatted ? " · " + esc(v.plate_formatted) : " · номер не прочитан"}</div>
        <div class="sub"><span class="dot ${v.on_territory ? "green" : "gray"}"></span>${v.on_territory ? "на территории · " + esc(v.current_warehouse) : "выехала"} · ${esc(v.last_seen || "")}</div>
      </div>
      <img src="${esc(v.photo || "")}" alt="" onerror="this.style.visibility='hidden'">
    </div>`).join("");
  document.querySelectorAll(".vehicle").forEach((el) => (el.onclick = () => openVehicle(el.dataset.id)));
}
$("#search").oninput = renderVehicles;
// У руководителя панель «Техника» лежит поверх карты — её можно свернуть и смотреть карту целиком
$("#btn-panel").onclick = () => {
  const collapsed = $(".panel.left").classList.toggle("collapsed");
  $("#btn-panel").textContent = collapsed ? "›" : "‹";
  $("#btn-panel").title = collapsed ? "Показать список техники" : "Свернуть список техники";
};

// Ошибка сети/сервера не должна молча глотаться при клике по карточке.
const openVehicle = (id) => loadVehicleModal(id).catch((e) => alert("Не удалось открыть: " + e.message));
async function loadVehicleModal(id) {
  const d = await api(`/api/vehicles/${encodeURIComponent(id)}`);
  const v = d.vehicle, a = v.appearance || {};
  const trips = d.trips.map((tr) => `
    <tr><td>${tr.id}</td><td>${esc(tr.warehouse)}</td><td>${esc(tr.entry_time)}<br>${t(tr.entry_weight)}</td>
      <td>${dash(tr.exit_time)}<br>${tr.exit_weight != null ? t(tr.exit_weight) : ""}</td><td><b>${t(tr.net_weight)}</b></td>
      <td>${dash(tr.driver)}<br>${dash(tr.crop)}</td>
      <td><span class="pill ${tr.status}">${tr.status === "open" ? "открыт" : "закрыт"}</span>${(tr.alerts || []).map((x) => `<br><span class="pill red">${esc(x)}</span>`).join("")}</td>
      <td>${tr.waybill ? `<button type="button" class="wb-open ghost small" data-waybill="${esc(waybillKey(tr.waybill, { warehouse: tr.warehouse }))}">📄 №${esc(tr.waybill.number)}</button>` : "—"}</td></tr>`).join("");
  const events = d.messages.filter((m) => m.role === "ai").map((m) => `
    <div class="event"><img src="${esc(m.image || "")}" alt=""><div><div class="small muted">${esc(m.created_at)}</div>${isReport(m) ? reportCompact(m) : md(m.text).split("\n").slice(0, 3).join("<br>")}</div></div>`).join("");
  openModal(`
    <div class="hero">
      <img src="${esc(v.photo || "")}" alt="">
      <div>
        <h2>${esc(v.label)} ${v.is_temporary ? '<span class="pill red">без номера</span>' : ""}</h2>
        <div class="kv">
          <div>Госномер</div><div><b>${dash(v.plate_formatted)}</b></div>
          <div>Производитель / модель</div><div>${dash(v.manufacturer)} / ${dash(v.model)}${v.year ? ` · ${esc(v.year)} г.` : ""}</div>
          <div>Тип техники</div><div>${dash(v.equipment_type)}</div>
          <div>Цвет</div><div>${dash(v.color || a.color)}</div>
          <div>Статус</div><div>${v.on_territory ? '<span class="pill open">на территории</span>' : '<span class="pill">выехала</span>'}</div>
          <div>Визитов</div><div>${v.visits} · первый ${esc(v.first_seen)} · последний ${esc(v.last_seen)}</div>
        </div>
      </div>
    </div>
    ${a.description ? `<h3>Описание внешности (VLM)</h3><div>${esc(a.description)}</div>` : ""}
    <h3>Рейсы</h3>
    <table><tr><th>Рейс</th><th>Склад</th><th>Заезд · брутто</th><th>Выезд · тара</th><th>Нетто</th><th>Водитель · культура</th><th>Статус</th><th>Путевой лист</th></tr>${trips || "<tr><td colspan=8>нет</td></tr>"}</table>
    <h3>События с камер</h3><div class="events">${events || '<div class="muted">нет</div>'}</div>`);
}

// ---------- склады ----------
const map = { obj: null, layer: null, markers: {} };
// Панель «Техника» лежит поверх карты слева: склады не должны прятаться под ней
const PANEL_PAD = () => ($(".panel.left").classList.contains("collapsed") ? 40 : $(".panel.left").offsetWidth + 40);

function ensureMap() {
  if (map.obj || typeof L === "undefined") return map.obj;
  // Карту создаём только для руководителя: у весовщика панель скрыта, тайлы качать незачем.
  if (!document.body.classList.contains("role-manager")) return null;
  map.obj = L.map("map", { zoomControl: false, attributionControl: true });
  L.control.zoom({ position: "bottomright" }).addTo(map.obj);  // слева сверху его закрыла бы панель
  // Карта во весь экран меняет размер вместе с раскладкой (смена роли, окно браузера) — Leaflet сам
  // этого не замечает и рисует плитки по старому размеру: сверху оставалась пустая полоса.
  new ResizeObserver(() => map.obj.invalidateSize()).observe($("#map"));
  const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" });
  const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 18, attribution: "Esri World Imagery" });
  // Подсказку «карта не загрузилась» убираем, только когда реально пришли тайлы.
  [osm, sat].forEach((layer) => layer.once("load", () => $("#map").querySelector(".map-offline")?.remove()));
  sat.addTo(map.obj);
  L.control.layers({ "Спутник (поля)": sat, "Схема": osm }, null, { collapsed: false, position: "topright" }).addTo(map.obj);
  map.layer = L.layerGroup().addTo(map.obj);
  L.DomUtil.create("div", "map-gps", $("#map")).textContent = "Костанайская обл. · склады в полях, 20–30 км от Костаная · координаты условные";
  return map.obj;
}

// Окошко склада на карте: главное одним взглядом, подробности — кнопкой (большое окно с рейсами)
function warehousePopup(w) {
  const el = document.createElement("div");
  el.className = "wh-popup";
  el.innerHTML = `
    <div class="row"><b>${esc(w.name)}</b><span class="badge">${w.count_now} ед. сейчас</span></div>
    <div class="small muted">${esc(w.scale)}</div>
    <div class="small">Посл. заезд: ${w.last_entry ? `<b>${esc(w.last_entry.vehicle?.label)}</b> · ${esc(w.last_entry.entry_time)}` : "—"}</div>
    <div class="small">Посл. выезд: ${w.last_exit ? `<b>${esc(w.last_exit.vehicle?.label)}</b> · ${esc(w.last_exit.exit_time)}` : "—"}</div>
    <div class="small">Выгружено: <b>${t(w.received_t)}</b> · загружено: <b>${t(w.shipped_t)}</b></div>
    <div class="row small"><span>Заполнение: ${t(w.received_t)} из ${w.capacity_t} т</span><span>${w.load_pct}%</span></div>
    <div class="bar"><i style="width:${w.load_pct}%"></i></div>
    ${w.alerts ? `<div class="small alert-line">⚠ тревог: ${w.alerts}</div>` : ""}
    <button class="primary small more">Подробнее: сводка, рейсы, события</button>`;
  el.querySelector(".more").onclick = () => openWarehouse(w.id);
  return el;
}

function renderWarehouses() {
  const mapOk = !!ensureMap();
  if (mapOk) {
    const pts = [];
    state.warehouses.forEach((w) => {
      // iconSize: null — размер по содержимому; по умолчанию Leaflet даёт значку 12×12 px,
      // и надпись склада вылезала из узкой рамки (на спутнике её было не прочитать)
      const icon = L.divIcon({ className: "", iconSize: null, html: `<div class="wh-marker ${w.alerts ? "alert" : ""}" data-id="${w.id}"><b>${esc(w.name)}</b>${esc(w.scale)}<span class="cnt ${w.count_now ? "" : "zero"}">${w.count_now}</span></div>`, iconAnchor: [70, 22] });
      const m = map.markers[w.id];
      // Данные обновляются каждые несколько секунд: маркер и окошко меняем на месте,
      // иначе открытое окошко склада закрывалось бы само при каждом обновлении.
      if (m) { m.setIcon(icon); m.setPopupContent(warehousePopup(w)); }
      else {
        map.markers[w.id] = L.marker([w.lat, w.lon], { icon }).addTo(map.layer)
          .bindPopup(warehousePopup(w), { maxWidth: 320, minWidth: 260, autoPanPaddingTopLeft: [PANEL_PAD(), 20] });
      }
      pts.push([w.lat, w.lon]);
    });
    if (pts.length && !map.fitted) {
      map.obj.fitBounds(pts, { paddingTopLeft: [PANEL_PAD(), 70], paddingBottomRight: [70, 50] });
      map.fitted = true;
    }
  }
  // Список складов — только если карты нет совсем: иначе до складов было бы не добраться
  $("#warehouse-list").classList.toggle("hidden", mapOk);
  document.body.classList.toggle("no-map", !mapOk);
  if (mapOk) return;
  $("#warehouse-list").innerHTML = state.warehouses.map((w) => `
    <div class="wcard" data-id="${w.id}">
      <div class="row"><b>${esc(w.name)}</b><span class="badge">${w.count_now} ед. сейчас</span></div>
      <div class="small muted">Посл. заезд: ${w.last_entry ? esc(w.last_entry.vehicle?.label) + " · " + esc(w.last_entry.entry_time) : "—"}</div>
      <div class="small muted">Посл. выезд: ${w.last_exit ? esc(w.last_exit.vehicle?.label) + " · " + esc(w.last_exit.exit_time) : "—"}</div>
      <div class="row small"><span>Принято: ${t(w.load_t)} / ${w.capacity_t} т</span><span>${w.load_pct}%</span></div>
      <div class="bar"><i style="width:${w.load_pct}%"></i></div>
    </div>`).join("");
  document.querySelectorAll(".wcard").forEach((el) => (el.onclick = () => openWarehouse(el.dataset.id)));
}

const openWarehouse = (id) => loadWarehouseModal(id).catch((e) => alert("Не удалось открыть: " + e.message));
async function loadWarehouseModal(id) {
  const w = await api(`/api/warehouses/${id}`);
  const trips = w.trips.map((tr) => `
    <tr><td>${tr.id}</td><td><b>${esc(tr.vehicle?.label)}</b><br>${dash(tr.vehicle?.plate_formatted)}</td><td>${esc(tr.entry_time)}<br>${t(tr.entry_weight)}</td>
      <td>${dash(tr.exit_time)}<br>${tr.exit_weight != null ? t(tr.exit_weight) : ""}</td><td><b>${t(tr.net_weight)}</b></td><td>${dash(tr.driver)}<br>${dash(tr.crop)}</td>
      <td><span class="pill ${tr.status}">${tr.status === "open" ? "открыт" : "закрыт"}</span>${(tr.alerts || []).map((x) => `<br><span class="pill red">${esc(x)}</span>`).join("")}</td>
      <td>${tr.waybill ? `<button type="button" class="wb-open ghost small" data-waybill="${esc(waybillKey(tr.waybill, { warehouse: w.name }))}">📄 №${esc(tr.waybill.number)}</button>` : "—"}</td></tr>`).join("");
  const events = w.messages.slice().reverse().map((m) => `
    <div class="event"><img src="${esc(m.image || "")}" alt=""><div><div class="small muted">${esc(m.created_at)}${m.vehicle_id ? ` · <a class="vehicle-link" data-vehicle="${esc(m.vehicle_id)}">${esc(vehicleLabel(m.vehicle_id))}</a>` : ""}</div>${isReport(m) ? reportCompact(m) : md(m.text).split("\n").slice(0, 2).join("<br>")}</div></div>`).join("");
  // Сверху — главное для руководителя (сколько принято и отгружено, кто сейчас на складе),
  // ниже — полный журнал рейсов и событий с камер, свёрнутый: раскрывается по нажатию.
  const kpi = (label, value, sub = "", cls = "") =>
    `<div class="kpi ${cls}"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div>${sub ? `<div class="kpi-sub">${sub}</div>` : ""}</div>`;
  const tiles = [
    kpi("Выгружено на склад", t(w.received_t), `приёмка · закрытых рейсов: ${w.trips_closed}`, "in"),
    kpi("Загружено со склада", t(w.shipped_t), "отгрузка: машина уехала тяжелее", "out"),
    kpi("Заполнение склада", `${w.load_pct}%`, `${t(w.received_t)} из ${w.capacity_t} т<div class="bar"><i style="width:${w.load_pct}%"></i></div>`),
    kpi("Рейсов", `${w.trips_total}`, `открыто ${w.trips_open} · закрыто ${w.trips_closed}`),
  ].join("");
  const onSite = (w.on_site || []).map((n) => `<tr>
      <td><a class="vehicle-link" data-vehicle="${esc(n.vehicle?.id)}">${esc(n.vehicle?.label)}</a></td>
      <td>${dash(n.vehicle?.plate_formatted)}</td><td>${esc(n.entry_time)}</td><td>${t(n.entry_weight)}</td>
      <td>${esc([n.driver, n.crop].filter(Boolean).join(" · ") || "—")}</td>
      <td>${n.alerts.length
        ? `<span class="pill red">⚠ ${esc(n.status)}</span><div class="alert-text">${esc(n.alerts[0])}${n.alerts.length > 1 ? ` <b>+${n.alerts.length - 1}</b>` : ""}</div>`
        : `<span class="pill open">${esc(n.status)}</span>`}</td></tr>`).join("");
  // Техника на складе — заметная сводка; сам список машин скрыт и раскрывается по нажатию
  const flagged = (w.on_site || []).filter((n) => n.alerts.length).length;
  const waiting = w.count_now - flagged;
  const nowBlock = w.count_now ? `
    <details class="wh-now">
      <summary>
        <div class="now-head"><span class="now-count">${w.count_now}</span><span class="now-title">${plural(w.count_now, "машина", "машины", "машин")} на складе сейчас</span></div>
        <div class="now-stats">
          ${waiting ? `<span class="pill open">${waiting} ${plural(waiting, "ждёт", "ждут", "ждут")} выезда</span>` : ""}
          ${flagged ? `<span class="pill red">⚠ ${flagged} с замечаниями</span>` : ""}
          <span class="muted">брутто на заезде ${t(w.awaiting_gross_t)}</span>
        </div>
        <span class="now-toggle"><span class="show">Показать список ▸</span><span class="hide">Скрыть список ▾</span></span>
      </summary>
      <table class="wh-table"><tr><th>Машина</th><th>Госномер</th><th>Заехала</th><th>Брутто</th><th>Водитель · культура</th><th>Статус</th></tr>${onSite}</table>
    </details>`
    : '<div class="wh-now empty"><span class="now-count">0</span><span class="now-title">машин на складе сейчас</span></div>';
  const crops = Object.entries(w.by_crop || {}).map(([c, v]) =>
    `<tr><td>${esc(c)}</td><td>${v.received_t ? t(v.received_t) : "—"}</td><td>${v.shipped_t ? t(v.shipped_t) : "—"}</td><td>${v.trips}</td></tr>`).join("");
  const alertKinds = Object.entries(w.alert_kinds || {}).map(([k, n]) => `<span class="pill red">${esc(k)} ×${n}</span>`).join(" ");
  const minutes = (m) => (m >= 60 ? `${Math.floor(m / 60)} ч ${m % 60} мин` : `${m} мин`);
  openModal(`
    <h2>${esc(w.name)}</h2><div class="muted">${esc(w.scale)} · ${w.lat.toFixed(4)}° N, ${w.lon.toFixed(4)}° E · склад и координаты условные, события — с камеры</div>
    <div class="kpis">${tiles}</div>
    ${nowBlock}
    <div class="wh-two">
      <div><h3>По культурам · закрытые рейсы</h3>
        ${crops ? `<table class="wh-table"><tr><th>Культура</th><th>Выгружено</th><th>Загружено</th><th>Рейсов</th></tr>${crops}</table>`
                : '<div class="muted">Закрытых рейсов пока нет — тоннаж появится после выезда машин.</div>'}</div>
      <div><h3>Итоги</h3><div class="kv">
        <div>Время на весовой</div><div>${w.avg_turnaround_min != null ? `в среднем ${minutes(w.avg_turnaround_min)} от заезда до выезда` : "—"}</div>
        <div>Последний заезд</div><div>${w.last_entry ? `${esc(w.last_entry.vehicle?.label)} · ${esc(w.last_entry.entry_time)}` : "—"}</div>
        <div>Последний выезд</div><div>${w.last_exit ? `${esc(w.last_exit.vehicle?.label)} · ${esc(w.last_exit.exit_time)}` : "—"}</div>
        <div>Тревоги</div><div>${w.alerts ? alertKinds : "нет"}</div>
      </div></div>
    </div>
    <details class="wh-more"><summary>Все рейсы · ${w.trips_total}</summary>
      <table><tr><th>Рейс</th><th>Машина</th><th>Заезд · брутто</th><th>Выезд · тара</th><th>Нетто</th><th>Водитель · культура</th><th>Статус</th><th>Путевой лист</th></tr>${trips || "<tr><td colspan=8>нет</td></tr>"}</table>
    </details>
    <details class="wh-more"><summary>Информация с камер · ${w.messages.length}</summary>
      <div class="events">${events || '<div class="muted">нет</div>'}</div>
    </details>`);
}

function openModal(html) { $("#modal-body").innerHTML = html; $("#modal").classList.remove("hidden"); }
$("#modal-close").onclick = () => $("#modal").classList.add("hidden");
$("#modal").onclick = (e) => { if (e.target === $("#modal")) $("#modal").classList.add("hidden"); };

// ---------- фото ----------
// Снимок приходит с камеры весовой (кнопка «Подключить камеру» — имитация: камера сама снимает
// заехавшую машину, кадр весовщик не выбирает) или загружается с компьютера вручную.
function renderWarehouseSelect() {
  const wsel = $("#f-warehouse");
  if (!wsel.options.length) wsel.innerHTML = state.warehouses.map((w) => `<option value="${w.id}">${esc(w.name)}</option>`).join("");
}

function setAttachment(a) {
  state.attachment = a;
  $("#attachment").classList.toggle("hidden", !a);
  $("#btn-send").disabled = !a || state.busy;
  if (a) { $("#attachment-img").src = a.url; $("#attachment-name").textContent = a.name || "Фото с компьютера"; }
}
$("#attachment-remove").onclick = () => setAttachment(null);

async function uploadPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return alert("Нужна фотография (jpg или png)");
  try {
    const fd = new FormData(); fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    setAttachment({ ...(await res.json()), name: file.name });
    $("#btn-send").focus();
  } catch (err) {
    alert("Не удалось загрузить фото: " + err.message);
  }
}
$("#file-input").onchange = async (e) => {
  await uploadPhoto(e.target.files[0]);
  e.target.value = "";  // тот же файл можно выбрать повторно
};
// Фото можно просто перетащить из папки в чат
const dropZone = $(".panel.center");
dropZone.addEventListener("dragover", (e) => { if (e.dataTransfer?.types.includes("Files")) { e.preventDefault(); dropZone.classList.add("dragover"); } });
dropZone.addEventListener("dragleave", (e) => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove("dragover"); });
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  uploadPhoto(e.dataTransfer?.files?.[0]);
});

// Камера весовой (имитация): окно «подключение → снимок» и снимок сам прикрепляется в чат.
// Паузы — чтобы окно было видно и читалось как работа камеры, а не мгновенная подмена файла.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function cameraSteps(steps) {
  const icon = { wait: '<span class="icon"><span class="spinner"></span></span>', ok: '<span class="icon ok">✓</span>', err: '<span class="icon err">✕</span>' };
  $("#camera-steps").innerHTML = steps.map(([kind, text]) => `<div class="step ${kind}">${icon[kind]}${esc(text)}</div>`).join("");
}
async function takeCameraShot() {
  const dlg = $("#camera-dialog"), shot = $("#camera-shot"), btn = $("#btn-camera");
  btn.disabled = true;
  shot.removeAttribute("src");
  $("#camera-close").classList.add("hidden");
  dlg.classList.remove("hidden", "shot-done");
  cameraSteps([["wait", "Подключение к камере весовой…"]]);
  try {
    await sleep(900);
    cameraSteps([["ok", "Камера подключена"], ["wait", "Машина на весах — делаю снимок…"]]);
    const [a] = await Promise.all([api("/api/camera/capture", { method: "POST", body: JSON.stringify({ random: true }) }), sleep(900)]);
    await new Promise((ok) => { shot.onload = shot.onerror = ok; shot.src = a.url; });
    dlg.classList.add("shot-done");   // вспышка и сам снимок
    const port = state.scale?.port || "COM3";
    cameraSteps([["ok", "Камера подключена"], ["ok", "Снимок получен"], ["wait", `Весы ${port}: машина на платформе, жду стабильного веса…`]]);
    await sleep(800);
    cameraSteps([["ok", "Камера подключена"], ["ok", "Снимок получен"], ["ok", `Весы ${port}: вес стабилен — запишется при отправке`]]);
    await sleep(900);
    const time = new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setAttachment({ ...a, name: `Снимок с камеры весовой · ${time}` });
    dlg.classList.add("hidden");
    $("#btn-send").focus();
  } catch (err) {
    cameraSteps([["err", "Камера не ответила: " + err.message]]);
    $("#camera-close").classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
}
$("#btn-camera").onclick = takeCameraShot;
$("#camera-close").onclick = () => $("#camera-dialog").classList.add("hidden");

// ---------- чат ----------
function renderMessage(m) {
  const files = m.files || {}, p = m.payload || {};
  if (m.role === "guard") {
    return `<div class="msg guard"><div class="who"><span>Охранник</span><span>${esc(m.created_at)}</span></div>
      ${m.image ? `<img class="shot" src="${esc(m.image)}" alt="">` : ""}<div class="text">${esc(m.text)}</div></div>`;
  }
  const who = `<div class="who"><span>ИИ-ассистент${p.processing_seconds ? ` · ${p.processing_seconds} c` : ""}</span><span>${esc(m.created_at)}</span></div>`;
  const image = m.image ? `<img class="shot" src="${esc(m.image)}" alt="">` : "";
  const links = [["annotated", "кадр"], ["plate_zoom", "номер"], ["vehicle_crop", "вырезка"], ["json", "JSON"], ["csv", "CSV"]]
    .filter(([k]) => files[k]).map(([k, n]) => `<a href="${esc(files[k])}" target="_blank" download>${n}</a>`);
  const filesLine = links.length ? `<span class="files">Файлы: ${links.join(" · ")}</span>` : "";
  if (isReport(m)) {
    const s = p.report_summary;
    const seen = s ? `<span>На кадре: техники ${s.vehicles_or_equipment}, людей ${s.people}, номеров прочитано ${s.plates_read}</span>` : "";
    return `<div class="msg ai report-msg${p.undone ? " undone" : ""}">${who}${image}${reportCard(m)}
      ${filesLine || seen ? `<div class="msg-foot">${filesLine}${seen}</div>` : ""}</div>`;
  }
  // Отмена, правка, ошибка — короткий текст, как и раньше
  const foot = [];
  if (p.error) foot.push('<span class="chip warn">Ошибка обработки</span>');
  if (filesLine) foot.push(filesLine);
  return `<div class="msg ai">${who}${image}<div class="text">${md(m.text)}</div>
    ${foot.length ? `<div class="msg-foot">${foot.join("")}</div>` : ""}</div>`;
}

// ---------- ответ по снимку — таблицей ----------
// Ответ собирается из данных, а не из текста: правка события (номер, вес, водитель) обновляет
// vehicle/trip в сообщении, и таблица сразу показывает новые значения.
const isReport = (m) => m.role === "ai" && ["entry", "exit"].includes(m.payload?.event_kind) && !!m.payload?.vehicle;
const EVENT_RU = { entry: "Заезд", exit: "Выезд" };

function row(label, value, note = "") {
  if (value == null || value === "") return "";
  return `<tr><th>${esc(label)}</th><td>${value}${note ? `<div class="note">${esc(note)}</div>` : ""}</td></tr>`;
}

function reportParts(m) {
  const p = m.payload || {}, d = p.details || {}, v = p.vehicle || {}, tr = p.trip || {};
  const exit = p.event_kind === "exit";
  const plate = p.plate?.formatted || v.plate_formatted;
  return {
    p, d, v, tr, exit, plate,
    warehouse: d.warehouse || state.warehouses.find((w) => w.id === m.warehouse_id)?.name,
    time: d.event_time || (exit ? tr.exit_time : tr.entry_time),
    make: [d.manufacturer ?? v.manufacturer, (d.country ?? null) && `(${d.country})`].filter(Boolean).join(" "),
    model: [d.model ?? (v.model !== "не определена" ? v.model : null), (d.year ?? v.year) && `${d.year ?? v.year} г.`].filter(Boolean).join(", "),
  };
}

function reportCard(m) {
  const { p, d, v, tr, exit, plate, warehouse, time, make, model } = reportParts(m);
  const files = m.files || {};
  const zoom = files.plate_zoom ? `<img class="plate-zoom zoom" src="${esc(files.plate_zoom)}" alt="номер">` : "";
  const plateCell = plate ? `<b class="plate">${esc(plate)}</b>${zoom}` : `<span class="bad">не прочитан</span>${zoom}`;
  const plateNote = plate ? p.plate?.source : (d.plate_guess ? `распознаватель предполагает ${d.plate_guess}` : "");
  const machine = [
    row("Тип техники", esc(d.equipment_type ?? v.equipment_type)),
    row("Производитель", esc(make || "не определён")),
    row("Модель", esc(model || "не определена")),
    row("Как определено", esc(d.identified_by)),
    row("Госномер", plateCell, plateNote),
    row("Реестр data.egov.kz", esc(d.registry)),
    row("Узнана", esc(d.recognized || d.plateless_match)),
  ].join("");
  const net = tr.net_weight;
  const weighing = [
    row("Событие", `<span class="chip ${p.event_kind}">${EVENT_RU[p.event_kind]}</span>`),
    exit ? row("Тара", t(p.weight), p.weight_source) : row("Брутто", `<b>${t(p.weight)}</b>`, p.weight_source),
    exit ? row("Брутто на заезде", t(tr.entry_weight)) : "",
    exit ? row("Нетто", `<b class="${net != null && net < 0 ? "bad" : ""}">${t(net)}</b>`) : "",
    row("Рейс", tr.id ? `№${esc(tr.id)} · ${exit ? "закрыт" : "открыт, ждём выезда"}` : ""),
    row("Склад", esc(warehouse)),
    row("Время", esc(time), d.time_note),
    row("Водитель", esc(tr.driver), d.driver_source ? `по: ${d.driver_source}` : ""),
    row("Культура", esc(tr.crop), d.driver_source ? `по: ${d.driver_source}` : ""),
    row("Заметка", esc(d.note)),
  ].join("");
  // Путевой лист — коротко под «Машиной и номером», целиком — по кнопке
  const wb = d.waybill || tr.waybill;
  const wbMini = wb ? `<div class="wb-mini">
      <div class="wb-mini-head">📄 Путевой лист №${esc(wb.number)} <span class="muted">от ${esc(wb.date)}</span></div>
      <div>Водитель: <b>${esc(wb.driver.name)}</b> · ${esc(wb.driver.category)}</div>
      <div>Груз: <b>${esc(wb.task.cargo)}</b> · ${esc(wb.task.from)} → ${esc(warehouse)}</div>
      <button type="button" class="wb-open" data-waybill="${esc(waybillKey(wb, { warehouse }))}">Открыть путевой лист</button>
    </div>` : "";
  const alerts = (p.alerts || []).map((a) => `<li>${esc(a)}</li>`).join("");
  const badges = [p.edited ? '<span class="chip exit">исправлено весовщиком</span>' : "",
                  p.undone ? '<span class="chip">отменено</span>' : ""].join("");
  return `<div class="report">
    <div class="report-title"><a class="vehicle-link" data-vehicle="${esc(v.id)}" title="Открыть карточку машины">${esc(vehicleLabel(v.id, v.label))}</a>${badges}</div>
    ${alerts ? `<ul class="report-alerts">${alerts}</ul>` : ""}
    <div class="report-grid">
      <div class="report-col"><table class="kv-table"><caption>Машина и номер</caption>${machine}</table>${wbMini}</div>
      <table class="kv-table"><caption>Взвешивание</caption>${weighing}</table>
    </div>
    ${d.appearance ?? v.appearance?.description ? `<div class="report-extra"><b>Внешность:</b> ${esc(d.appearance ?? v.appearance?.description)}</div>` : ""}
  </div>`;
}

// ---------- путевой лист (ИМИТАЦИЯ: реальных листов нет, backend/waybill.py) ----------
// Кнопки «Открыть путевой лист» есть в чате, в карточке машины и в окне склада: лист кладём в реестр
// по номеру, кнопка хранит только номер.
const WAYBILLS = new Map();
function waybillKey(wb, extra = {}) {
  WAYBILLS.set(wb.number, { wb, ...extra });
  return wb.number;
}
function openWaybill(key) {
  const entry = WAYBILLS.get(key);
  if (!entry) return;
  const { wb, warehouse } = entry, veh = wb.vehicle || {}, drv = wb.driver || {}, task = wb.task || {}, dep = wb.departure || {}, mk = wb.marks || {};
  const kv = (pairs) => `<div class="kv">${pairs.filter(([, v]) => v != null && v !== "").map(([k, v]) => `<div>${esc(k)}</div><div>${esc(v)}</div>`).join("")}</div>`;
  openModal(`
    <div class="waybill-doc">
      <div class="wb-org">${esc(wb.organization)}</div>
      <h2 class="wb-title">${esc(wb.title.toUpperCase())} № ${esc(wb.number)}</h2>
      <div class="wb-date">от ${esc(wb.date)}</div>
      <div class="wb-grid">
        <section><h4>${wb.kind === "tractor" ? "Трактор" : "Автомобиль"}</h4>${kv([["Марка, модель", veh.make_model], ["Тип", veh.type], ["Госномер", veh.plate], ["Гаражный №", veh.garage_no], ["Прицеп", veh.trailer]])}</section>
        <section><h4>${wb.kind === "tractor" ? "Тракторист" : "Водитель"}</h4>${kv([["ФИО", drv.name], ["Табельный №", drv.personnel_no], ["Удостоверение", drv.license], ["Категория", drv.category]])}</section>
        <section><h4>Задание</h4>${kv([["Груз", task.cargo], ["Откуда", task.from], ["Куда", warehouse || "склад, указанный весовщиком"], ["Рейсов по заданию", task.trips_planned]])}</section>
        <section><h4>Выезд</h4>${kv([["Время выезда", dep.time], ["Показания одометра", dep.odometer_km != null ? `${dep.odometer_km.toLocaleString("ru-RU")} км` : null], ["Моточасы", dep.engine_hours != null ? `${dep.engine_hours.toLocaleString("ru-RU")} ч` : null]])}</section>
      </div>
      <section class="wb-marks"><h4>Отметки</h4>${kv([["Предрейсовый медосмотр", mk.medic], ["Выпуск на линию (механик)", mk.mechanic], ["Диспетчер", mk.dispatcher]])}</section>
      <div class="wb-foot">Путевой лист выписывает диспетчер до выезда; весовая берёт из него водителя и груз, весовщику вводить их не нужно.</div>
    </div>`);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest(".wb-open[data-waybill]");
  if (b) openWaybill(b.dataset.waybill);
});

// Коротко для карточек руководителя (машина, склад): событие, вес, водитель, культура, тревоги
function reportCompact(m) {
  const { p, tr, exit, plate, warehouse, time } = reportParts(m);
  const alerts = (p.alerts || []).map((a) => `<li>${esc(a)}</li>`).join("");
  return `<table class="kv-table compact">
      ${row("Событие", `<span class="chip ${p.event_kind}">${EVENT_RU[p.event_kind]}</span> ${exit ? `нетто <b>${t(tr.net_weight)}</b>` : `брутто <b>${t(p.weight)}</b>`}`)}
      ${row("Госномер", esc(plate || "не прочитан"))}
      ${row("Водитель · культура", esc([tr.driver, tr.crop].filter(Boolean).join(" · ")))}
      ${row("Склад · время", esc([warehouse, time].filter(Boolean).join(" · ")))}
    </table>${alerts ? `<ul class="report-alerts">${alerts}</ul>` : ""}`;
}

document.addEventListener("click", (e) => {
  const link = e.target.closest(".vehicle-link[data-vehicle]");
  if (link) { $("#modal").classList.add("hidden"); openVehicle(link.dataset.vehicle); }
});

let lastMessageKey = "";
async function loadMessages() {
  const msgs = await api("/api/messages");
  const key = msgs.length ? `${msgs.length}:${msgs[msgs.length - 1].id}` : "0";
  const pending = msgs.length && msgs[msgs.length - 1].role === "guard";
  if (key === lastMessageKey && !!$("#pending") === !!pending) return;
  lastMessageKey = key;
  const box = $("#messages");
  box.innerHTML = msgs.length ? msgs.map(renderMessage).join("") : '<div class="empty">Чат пуст. Загрузите фото машины с компьютера (кнопкой или перетащите в чат), заполните фамилию и культуру и отправьте.</div>';
  // Сервер ещё считает: последний ответ — от охранника, ответа ИИ пока нет (например, после перезагрузки страницы).
  if (pending && !state.busy) box.insertAdjacentHTML("beforeend", pendingHtml());
  box.scrollTop = box.scrollHeight;
}
const pendingHtml = () => `<div class="msg ai" id="pending"><div class="who"><span>ИИ-ассистент</span></div><span class="spinner"></span>Анализирую снимок: техника → номер → марка/модель → рейс… (обычно 10–30 с)</div>`;

// Enter в полях формы переводит к следующему полю; отправка — только из заметки (или кнопкой).
// Весовщик вводит только склад, номер (если ИИ ошибся) и заметку: вес — с весов на COM-порту,
// водитель и груз — из путевого листа (обе вещи — имитация, реальных данных нет).
const FIELDS = ["#f-plate", "#f-note"];
FIELDS.slice(0, -1).forEach((s, i) => $(s).addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.isComposing) return;
  e.preventDefault();
  $(FIELDS[i + 1]).focus();
}));

function showHint(text, field) {
  const hint = $("#composer-hint");
  hint.textContent = text || "";
  hint.classList.toggle("hidden", !text);
  FIELDS.forEach((s) => $(s).classList.toggle("invalid", $(s) === field));
  if (field) field.focus();
}
FIELDS.forEach((s) => $(s).addEventListener("input", () => { if ($(s).classList.contains("invalid")) showHint(""); }));

$("#composer").onsubmit = async (e) => {
  e.preventDefault();
  if (!state.attachment || state.busy) return;
  showHint("");
  state.busy = true; $("#btn-send").disabled = true;
  const snapshot = Object.fromEntries(FIELDS.map((s) => [s, $(s).value]));
  const body = {
    capture_id: state.attachment.capture_id, frame: state.attachment.frame,
    warehouse_id: $("#f-warehouse").value,
    plate_override: $("#f-plate").value.trim() || null, note: $("#f-note").value.trim(),
  };
  const box = $("#messages");
  box.querySelector(".empty")?.remove();
  box.insertAdjacentHTML("beforeend", `<div class="msg guard"><div class="who"><span>Охранник</span></div><img class="shot" src="${esc(state.attachment.url)}"><div class="text">${esc([body.plate_override && "номер: " + body.plate_override, body.note].filter(Boolean).join(", ") || "Машина на весах.")}</div></div>` + pendingHtml());
  box.scrollTop = box.scrollHeight;
  const saved = state.attachment;
  setAttachment(null);
  try {
    await api("/api/report", { method: "POST", body: JSON.stringify(body) });
    // Пока ИИ думал, весовщик мог начать вводить следующую машину — такие поля не трогаем.
    FIELDS.forEach((s) => { if ($(s).value === snapshot[s]) $(s).value = ""; });
  } catch (err) {
    if (!state.attachment) setAttachment(saved);  // снимок не теряем — можно отправить повторно
    alert("Ошибка обработки: " + err.message);
  } finally {
    state.busy = false;
    $("#btn-send").disabled = !state.attachment;
    await Promise.allSettled([loadMessages(), refreshState()]);
  }
};

$("#btn-reset").onclick = async () => {
  if (!confirm("Очистить машины, рейсы и чат для нового демо?")) return;
  try {
    await api("/api/reset", { method: "POST" });
    await Promise.all([loadMessages(), refreshState()]);
  } catch (e) { alert("Сброс не удался: " + e.message); }
};

// ---------- старт ----------
(async () => {
  applyRole(getRole());
  // Первая загрузка может не удаться (сервер перезапускается) — опрос всё равно запускаем.
  // Сначала машины и склады: таблицы ответов в чате берут из них подписи (название склада, машины)
  await refreshState().catch(() => {});
  await loadMessages().catch(() => {});
  setInterval(() => refreshState().catch(() => {}), 20000);
  setInterval(() => { if (!state.busy) loadMessages().catch(() => {}); }, 5000);
})();
