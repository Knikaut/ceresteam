/* Интерфейс весовой, две роли:
   Весовщик — чат с ИИ (снимок с камеры + фамилия, культура, вес → заезд/выезд);
   Руководитель — машины и карта складов (рейсы, тревоги, события с камер). */
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { vehicles: [], warehouses: [], camera: { frames: [], used: [], next: null }, attachment: null, busy: false };

// ---------- утилиты ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(⚠[^\n]*)/g, '<span class="alert">$1</span>');
const t = (v) => (v == null ? "—" : `${Number(v).toFixed(2)} т`);
const dash = (v) => (v == null || v === "" ? "—" : esc(v));
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
  Object.assign(state, { vehicles: s.vehicles, warehouses: s.warehouses, camera: s.camera });
  renderVehicles();
  renderWarehouses();
  renderCamera();
  $("#status-territory").textContent = `На территории: ${s.on_territory} ед.`;
  // textContent — строки сервера (модель, ошибка) не интерпретируются как HTML.
  const vlm = s.vlm || { enabled: s.vlm_enabled, model: s.vlm_model };
  let vlmText = "VLM: выкл (марка/модель — только по реестру и надписям)";
  if (vlm.enabled) {
    vlmText = `VLM: ${[vlm.provider, vlm.model].filter(Boolean).join(" · ") || "подключена"}`;
    if (vlm.fallback) vlmText += ` (резерв ${vlm.fallback})`;
    if (vlm.last_error) vlmText += ` · ⚠ ${vlm.last_error}`;
  }
  $("#status-vlm").textContent = vlmText;
  $("#status-time").textContent = s.server_time;
  $("#status-camera").textContent = `Камера: папка кадров (${s.camera.frames.length} шт.)`;
}

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
  if (!list.length) { $("#vehicle-list").innerHTML = '<div class="empty">Пока нет машин. Сделайте снимок с камеры.</div>'; return; }
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

// Ошибка сети/сервера не должна молча глотаться при клике по карточке.
const openVehicle = (id) => loadVehicleModal(id).catch((e) => alert("Не удалось открыть: " + e.message));
async function loadVehicleModal(id) {
  const d = await api(`/api/vehicles/${encodeURIComponent(id)}`);
  const v = d.vehicle, a = v.appearance || {};
  const trips = d.trips.map((tr) => `
    <tr><td>${tr.id}</td><td>${esc(tr.warehouse)}</td><td>${esc(tr.entry_time)}<br>${t(tr.entry_weight)}</td>
      <td>${dash(tr.exit_time)}<br>${tr.exit_weight != null ? t(tr.exit_weight) : ""}</td><td><b>${t(tr.net_weight)}</b></td>
      <td>${dash(tr.driver)}<br>${dash(tr.crop)}</td>
      <td><span class="pill ${tr.status}">${tr.status === "open" ? "открыт" : "закрыт"}</span>${(tr.alerts || []).map((x) => `<br><span class="pill red">${esc(x)}</span>`).join("")}</td></tr>`).join("");
  const events = d.messages.filter((m) => m.role === "ai").map((m) => `
    <div class="event"><img src="${esc(m.image || "")}" alt=""><div><div class="small muted">${esc(m.created_at)} · ${m.event_kind === "entry" ? "заезд" : m.event_kind === "exit" ? "выезд" : ""}</div>${md(m.text).split("\n").slice(0, 3).join("<br>")}</div></div>`).join("");
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
    <table><tr><th>Рейс</th><th>Склад</th><th>Заезд · брутто</th><th>Выезд · тара</th><th>Нетто</th><th>Водитель · культура</th><th>Статус</th></tr>${trips || "<tr><td colspan=7>нет</td></tr>"}</table>
    <h3>События с камер</h3><div class="events">${events || '<div class="muted">нет</div>'}</div>`);
}

// ---------- склады ----------
const map = { obj: null, layer: null };

function ensureMap() {
  if (map.obj || typeof L === "undefined") return map.obj;
  // Карту создаём только для руководителя: у весовщика панель скрыта, тайлы качать незачем.
  if (!document.body.classList.contains("role-manager")) return null;
  map.obj = L.map("map", { zoomControl: true, attributionControl: true });
  const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" });
  const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 18, attribution: "Esri World Imagery" });
  // Подсказку «карта не загрузилась» убираем, только когда реально пришли тайлы.
  [osm, sat].forEach((layer) => layer.once("load", () => $("#map").querySelector(".map-offline")?.remove()));
  sat.addTo(map.obj);
  L.control.layers({ "Спутник (поля)": sat, "Схема": osm }, null, { collapsed: false, position: "topright" }).addTo(map.obj);
  map.layer = L.layerGroup().addTo(map.obj);
  L.DomUtil.create("div", "map-gps", $("#map")).textContent = "GPS: 53.2205° N, 63.6280° E · Костанайская обл. · координаты складов условные";
  return map.obj;
}

function renderWarehouses() {
  if (ensureMap()) {
    map.layer.clearLayers();
    const pts = [];
    state.warehouses.forEach((w) => {
      const icon = L.divIcon({ className: "", html: `<div class="wh-marker ${w.alerts ? "alert" : ""}" data-id="${w.id}"><b>${esc(w.name)}</b>${esc(w.scale)}<span class="cnt ${w.count_now ? "" : "zero"}">${w.count_now}</span></div>`, iconAnchor: [70, 22] });
      L.marker([w.lat, w.lon], { icon }).addTo(map.layer).on("click", () => openWarehouse(w.id));
      pts.push([w.lat, w.lon]);
    });
    if (pts.length && !map.fitted) { map.obj.fitBounds(pts, { padding: [40, 40] }); map.fitted = true; }
  }
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
  const now = (w.vehicles_now || []).filter(Boolean).map((v) => `<li><b>${esc(v.label)}</b> · ${dash(v.plate_formatted)}</li>`).join("");
  const trips = w.trips.map((tr) => `
    <tr><td>${tr.id}</td><td><b>${esc(tr.vehicle?.label)}</b><br>${dash(tr.vehicle?.plate_formatted)}</td><td>${esc(tr.entry_time)}<br>${t(tr.entry_weight)}</td>
      <td>${dash(tr.exit_time)}<br>${tr.exit_weight != null ? t(tr.exit_weight) : ""}</td><td><b>${t(tr.net_weight)}</b></td><td>${dash(tr.driver)}<br>${dash(tr.crop)}</td>
      <td><span class="pill ${tr.status}">${tr.status === "open" ? "открыт" : "закрыт"}</span>${(tr.alerts || []).map((x) => `<br><span class="pill red">${esc(x)}</span>`).join("")}</td></tr>`).join("");
  const events = w.messages.slice().reverse().map((m) => `
    <div class="event"><img src="${esc(m.image || "")}" alt=""><div><div class="small muted">${esc(m.created_at)} · ${m.event_kind === "entry" ? "заезд" : m.event_kind === "exit" ? "выезд" : "без рейса"} · ${m.vehicle_id ? esc(vehicleLabel(m.vehicle_id)) : ""}</div>${md(m.text).split("\n").slice(0, 2).join("<br>")}</div></div>`).join("");
  openModal(`
    <h2>${esc(w.name)}</h2><div class="muted">${esc(w.scale)} · ${w.lat.toFixed(4)}° N, ${w.lon.toFixed(4)}° E · склад и координаты условные, события — с камеры</div>
    <div class="kv" style="margin-top:12px">
      <div>Машин сейчас</div><div><b>${w.count_now}</b> <ul style="margin:4px 0 0 16px;padding:0">${now}</ul></div>
      <div>Последняя заехала</div><div>${w.last_entry ? `<b>${esc(w.last_entry.vehicle?.label)}</b> · ${esc(w.last_entry.entry_time)} · брутто ${t(w.last_entry.entry_weight)}` : "—"}</div>
      <div>Последняя выехала</div><div>${w.last_exit ? `<b>${esc(w.last_exit.vehicle?.label)}</b> · ${esc(w.last_exit.exit_time)} · нетто ${t(w.last_exit.net_weight)}` : "—"}</div>
      <div>Принято за смену</div><div>${t(w.load_t)} из ${w.capacity_t} т (${w.load_pct}%)</div>
      <div>Рейсов / тревог</div><div>${w.trips_total} / ${w.alerts}</div>
    </div>
    <h3>Рейсы</h3>
    <table><tr><th>Рейс</th><th>Машина</th><th>Заезд · брутто</th><th>Выезд · тара</th><th>Нетто</th><th>Водитель · культура</th><th>Статус</th></tr>${trips || "<tr><td colspan=7>нет</td></tr>"}</table>
    <h3>Информация с камер</h3><div class="events">${events || '<div class="muted">нет</div>'}</div>`);
}

function openModal(html) { $("#modal-body").innerHTML = html; $("#modal").classList.remove("hidden"); }
$("#modal-close").onclick = () => $("#modal").classList.add("hidden");
$("#modal").onclick = (e) => { if (e.target === $("#modal")) $("#modal").classList.add("hidden"); };

// ---------- камера ----------
function renderCamera() {
  const c = state.camera;
  $("#camera-info").textContent = c.frames.length ? `папка · ${c.used.length}/${c.frames.length} кадров обработано · следующий: ${c.next || "—"}` : "в папке нет кадров";
  const sel = $("#frame-select");
  const chosen = sel.value;  // не сбрасываем выбор весовщика при фоновом обновлении
  sel.innerHTML = '<option value="">следующий кадр</option>' + c.frames.map((f) => `<option value="${esc(f)}" ${c.used.includes(f) ? "disabled" : ""}>${esc(f)}${c.used.includes(f) ? " ✓" : ""}</option>`).join("");
  if (chosen && c.frames.includes(chosen) && !c.used.includes(chosen)) sel.value = chosen;
  const wsel = $("#f-warehouse");
  if (!wsel.options.length) wsel.innerHTML = state.warehouses.map((w) => `<option value="${w.id}">${esc(w.name)}</option>`).join("");
}

function setAttachment(a) {
  state.attachment = a;
  $("#attachment").classList.toggle("hidden", !a);
  $("#btn-send").disabled = !a || state.busy;
  if (a) { $("#attachment-img").src = a.url; $("#attachment-name").textContent = a.frame ? `Кадр камеры ${a.frame}` : "Загруженное фото"; }
}
$("#attachment-remove").onclick = () => setAttachment(null);

$("#btn-capture").onclick = async () => {
  try {
    const frame = $("#frame-select").value || null;
    const a = await api("/api/camera/capture", { method: "POST", body: JSON.stringify({ frame }) });
    setAttachment(a);
    $("#f-driver").focus();
  } catch (e) { alert("Камера: " + e.message); }
};
$("#file-input").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const fd = new FormData(); fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    setAttachment(await res.json());
    $("#f-driver").focus();
  } catch (err) {
    alert("Не удалось загрузить фото: " + err.message);
  } finally {
    e.target.value = "";  // тот же файл можно выбрать повторно
  }
};

// ---------- чат ----------
function renderMessage(m) {
  const files = m.files || {}, p = m.payload || {};
  if (m.role === "guard") {
    return `<div class="msg guard"><div class="who"><span>Охранник</span><span>${esc(m.created_at)}</span></div>
      ${m.image ? `<img class="shot" src="${esc(m.image)}" alt="">` : ""}<div class="text">${esc(m.text)}</div></div>`;
  }
  const chips = [];
  if (p.error) chips.push('<span class="chip warn">Ошибка обработки</span>');  // подробности уже в тексте ответа
  if (p.event_kind === "entry") chips.push('<span class="chip entry">Заезд · брутто ' + t(p.weight) + "</span>");
  if (p.event_kind === "exit") chips.push('<span class="chip exit">Выезд · тара ' + t(p.weight) + " · нетто " + t(p.trip?.net_weight) + "</span>");
  if (p.plate?.formatted) chips.push(`<span class="chip ok">Номер ${esc(p.plate.formatted)}</span>`);
  else if (p.vehicle) chips.push('<span class="chip warn">Номер не прочитан</span>');
  // Подпись берём актуальную из списка машин (марку могли уточнить позже), иначе — из ответа.
  if (p.vehicle) chips.push(`<span class="chip link" data-vehicle="${esc(p.vehicle.id)}" title="Открыть карточку машины">${esc(vehicleLabel(p.vehicle.id, p.vehicle.label))}</span>`);
  if (p.vehicle?.year) chips.push(`<span class="chip">${esc(p.vehicle.year)} г.</span>`);
  if (p.vehicle && p.vlm_used === false) chips.push('<span class="chip">VLM не использована</span>');
  if (p.recognized_by_appearance) chips.push('<span class="chip ok">Узнана по внешности</span>');
  (p.alerts || []).forEach((a) => chips.push(`<span class="chip warn">⚠ ${esc(a)}</span>`));
  const links = [["annotated", "Кадр с рамками"], ["plate_zoom", "Номер крупно"], ["vehicle_crop", "Вырезка машины"], ["json", "detections.json"], ["csv", "detections.csv"]]
    .filter(([k]) => files[k]).map(([k, n]) => `<a href="${esc(files[k])}" target="_blank" download>${n}</a>`).join("");
  return `<div class="msg ai"><div class="who"><span>ИИ-ассистент${p.processing_seconds ? ` · ${p.processing_seconds} c` : ""}</span><span>${esc(m.created_at)}</span></div>
    ${m.image ? `<img class="shot" src="${esc(m.image)}" alt="">` : ""}
    <div class="text">${md(m.text)}${files.plate_zoom ? `<img class="plate-zoom zoom" src="${esc(files.plate_zoom)}" alt="номер">` : ""}</div>
    <div class="chips">${chips.join("")}</div><div class="files">${links}</div></div>`;
}

$("#messages").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip.link[data-vehicle]");
  if (chip) openVehicle(chip.dataset.vehicle);
});

let lastMessageKey = "";
async function loadMessages() {
  const msgs = await api("/api/messages");
  const key = msgs.length ? `${msgs.length}:${msgs[msgs.length - 1].id}` : "0";
  const pending = msgs.length && msgs[msgs.length - 1].role === "guard";
  if (key === lastMessageKey && !!$("#pending") === !!pending) return;
  lastMessageKey = key;
  const box = $("#messages");
  box.innerHTML = msgs.length ? msgs.map(renderMessage).join("") : '<div class="empty">Чат пуст. Нажмите «Снимок с камеры», заполните фамилию и культуру и отправьте.</div>';
  // Сервер ещё считает: последний ответ — от охранника, ответа ИИ пока нет (например, после перезагрузки страницы).
  if (pending && !state.busy) box.insertAdjacentHTML("beforeend", pendingHtml());
  box.scrollTop = box.scrollHeight;
}
const pendingHtml = () => `<div class="msg ai" id="pending"><div class="who"><span>ИИ-ассистент</span></div><span class="spinner"></span>Анализирую снимок: техника → номер → марка/модель → рейс… (10–30 с на CPU)</div>`;

// Enter в полях формы переводит к следующему полю; отправка — только из заметки (или кнопкой).
const FIELDS = ["#f-driver", "#f-crop", "#f-weight", "#f-plate", "#f-note"];
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
  // Фамилия и культура обязательны: без них рейс не оформить.
  if (!$("#f-driver").value.trim()) return showHint("Укажите фамилию водителя", $("#f-driver"));
  if (!$("#f-crop").value.trim()) return showHint("Укажите культуру", $("#f-crop"));
  showHint("");
  state.busy = true; $("#btn-send").disabled = true;
  const snapshot = Object.fromEntries(FIELDS.map((s) => [s, $(s).value]));
  const body = {
    capture_id: state.attachment.capture_id, frame: state.attachment.frame,
    driver: $("#f-driver").value.trim(), crop: $("#f-crop").value.trim(), warehouse_id: $("#f-warehouse").value,
    weight: $("#f-weight").value.trim() || null, plate_override: $("#f-plate").value.trim() || null, note: $("#f-note").value.trim(),
  };
  const box = $("#messages");
  box.querySelector(".empty")?.remove();
  box.insertAdjacentHTML("beforeend", `<div class="msg guard"><div class="who"><span>Охранник</span></div><img class="shot" src="${esc(state.attachment.url)}"><div class="text">${esc([body.driver && "водитель: " + body.driver, body.crop && "культура: " + body.crop, body.note].filter(Boolean).join(", ") || "Машина на весах.")}</div></div>` + pendingHtml());
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
  await Promise.allSettled([refreshState(), loadMessages()]);
  setInterval(() => refreshState().catch(() => {}), 20000);
  setInterval(() => { if (!state.busy) loadMessages().catch(() => {}); }, 5000);
})();
