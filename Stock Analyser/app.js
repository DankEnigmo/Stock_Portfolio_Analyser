lucide.createIcons();

const CONFIG = {
  API_BASE_URL: "http://localhost:5000",
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
  CONNECTION_TIMEOUT: 10000,
};

let connectionStatus = "unknown";
let myStockChart = null;

const formatCurrency = v =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(v);

const formatPercentage = v =>
  new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(v / 100);

const safeFixed = (v, d = 2) =>
  typeof v === "number" && !isNaN(v) ? v.toFixed(d) : "N/A";

const debounce = (fn, wait) => {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), wait);
  };
};

function updateConnectionStatus(status, msg = "") {
  connectionStatus = status;
  const el = document.getElementById("connection-status");
  el.className = "fixed top-24 left-4 right-4 z-40 status-indicator";
  if (status === "connected") {
    el.classList.add("status-success", "hidden");
  } else if (status === "connecting") {
    el.classList.remove("hidden");
    el.classList.add("status-warning");
    el.innerHTML = `<div class="flex items-center space-x-2"><div class="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-400"></div><span>${msg}</span></div>`;
  } else if (status === "error") {
    el.classList.remove("hidden");
    el.classList.add("status-error");
    el.innerHTML = `<div class="flex items-center space-x-2"><i data-lucide="alert-circle" class="h-4 w-4"></i><span>${msg}</span></div>`;
    lucide.createIcons();
  } else if (status === "offline") {
    el.classList.remove("hidden");
    el.classList.add("status-error");
    el.innerHTML = `<div class="flex items-center space-x-2"><i data-lucide="wifi-off" class="h-4 w-4"></i><span>Offline</span></div>`;
    lucide.createIcons();
  }
}

async function fetchAPI(endpoint, opts = {}, retry = 0) {
  const url = `${CONFIG.API_BASE_URL}${endpoint}`;
  if (retry === 0 && connectionStatus !== "connected") {
    updateConnectionStatus("connecting");
  }
  try {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), CONFIG.CONNECTION_TIMEOUT);
    const res = await fetch(url, {
      ...opts,
      signal: ctrl.signal,
      headers: { "Content-Type": "application/json", ...opts.headers },
    });
    clearTimeout(id);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    updateConnectionStatus("connected");
    return data;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("Timeout");
    if (!navigator.onLine) {
      updateConnectionStatus("offline");
      throw new Error("No internet");
    }
    if (
      e.message.includes("Failed to fetch") ||
      e.message.includes("Network request failed")
    ) {
      if (retry < CONFIG.RETRY_ATTEMPTS) {
        await new Promise(r => setTimeout(r, CONFIG.RETRY_DELAY));
        return fetchAPI(endpoint, opts, retry + 1);
      } else {
        updateConnectionStatus("error", "Backend unreachable");
      }
    } else {
      updateConnectionStatus("error", e.message);
    }
    throw e;
  }
}

function showLoading(sec, msg = "") {
  const l = document.getElementById(`${sec}-loading`);
  const e = document.getElementById(`${sec}-error`);
  const r = document.getElementById(sec === "ai" ? "ai-results" : `${sec}-details`);
  l && l.classList.remove("hidden");
  e && e.classList.add("hidden");
  r && r.classList.add("hidden");
  const bt = document.getElementById(`${sec === "ai" ? "analyze" : sec}-btn-text`);
  const bl = document.getElementById(`${sec === "ai" ? "analyze" : sec}-btn-loading`);
  bt && bt.classList.add("hidden");
  bl && bl.classList.remove("hidden");
}

function showError(sec, msg) {
  const l = document.getElementById(`${sec}-loading`);
  const e = document.getElementById(`${sec}-error`);
  l && l.classList.add("hidden");
  e && (e.textContent = msg) && e.classList.remove("hidden");
  resetButtonState(sec);
}

function showResults(sec) {
  const l = document.getElementById(`${sec}-loading`);
  const r = document.getElementById(sec === "ai" ? "ai-results" : `${sec}-details`);
  l && l.classList.add("hidden");
  r && r.classList.remove("hidden");
  resetButtonState(sec);
}

function resetButtonState(sec) {
  const bt = document.getElementById(`${sec === "ai" ? "analyze" : sec}-btn-text`);
  const bl = document.getElementById(`${sec === "ai" ? "analyze" : sec}-btn-loading`);
  bt && bt.classList.remove("hidden");
  bl && bl.classList.add("hidden");
}

function addHoldingRow(t = "", s = "", p = "") {
  const div = document.getElementById("wallet-holdings");
  const row = document.createElement("div");
  row.className = "holding-row flex items-center space-x-2";
  row.innerHTML = `
    <input type="text" class="w-1/4 p-3 rounded-lg form-input text-white" placeholder="Ticker" value="${t}" required pattern="[A-Za-z]{1,5}">
    <input type="number" class="w-1/4 p-3 rounded-lg form-input text-white" placeholder="Shares" value="${s}" required min="0.001" step="0.001">
    <input type="number" class="w-1/4 p-3 rounded-lg form-input text-white" placeholder="Purchase Price" value="${p}" required min="0.01" step="0.01">
    <button type="button" onclick="removeHoldingRow(this)" class="w-1/4 text-red-500 hover:text-red-400 text-center p-2 rounded-lg hover:bg-red-500/10">
      <i data-lucide="trash-2" class="w-4 h-4 mx-auto"></i>
    </button>`;
  div.appendChild(row);
  lucide.createIcons();
  row.querySelector("input[type=text]").addEventListener("input", e => {
    e.target.value = e.target.value.toUpperCase();
  });
}

function removeHoldingRow(btn) {
  btn.closest(".holding-row").remove();
}

function validatePortfolioForm() {
  const cash = parseFloat(document.getElementById("ai-cash").value);
  const holdings = [];
  document.querySelectorAll("#wallet-holdings .holding-row").forEach(r => {
    const i = r.querySelectorAll("input");
    holdings.push({
      ticker: i[0].value.trim().toUpperCase(),
      shares: parseFloat(i[1].value),
      purchase_price: parseFloat(i[2].value),
    });
  });
  if (isNaN(cash) || cash < 0) throw new Error("Valid cash amount required.");
  if (!holdings.length) throw new Error("Add at least one holding.");
  holdings.forEach((h, idx) => {
    if (!h.ticker || h.ticker.length < 1 || h.ticker.length > 5)
      throw new Error(`Row ${idx + 1}: invalid ticker.`);
    if (isNaN(h.shares) || h.shares <= 0)
      throw new Error(`Row ${idx + 1}: invalid shares.`);
    if (isNaN(h.purchase_price) || h.purchase_price <= 0)
      throw new Error(`Row ${idx + 1}: invalid purchase price.`);
  });
  return { cash, holdings };
}

async function analyzePortfolio(e) {
  e.preventDefault();
  try {
    const { cash, holdings } = validatePortfolioForm();
    showLoading("ai");
    const res = await fetchAPI("/analyze_portfolio", {
      method: "POST",
      body: JSON.stringify({ cash, wallet: holdings }),
    });
    showResults("ai");
    renderPortfolioResults(res);
  } catch (err) {
    showError("ai", `Error: ${err.message}`);
  }
}

function renderPortfolioResults(r) {
    const div = document.getElementById("ai-results");
    const s = r.portfolio_summary;
    const rec = r.recommendations;
    const allocDetails = r.allocation_details || {};
    
    div.innerHTML = `
      <div class="space-y-6">
        <!-- Portfolio Summary Section (unchanged) -->
        <div class="glass-card p-6 rounded-xl">
          <h4 class="text-2xl text-white mb-4 flex items-center">
            <i data-lucide="pie-chart" class="w-6 h-6 mr-2 text-blue-400"></i>Portfolio Summary
          </h4>
          <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div class="text-center p-4 bg-gray-800/30 rounded-lg">
              <div class="text-2xl font-bold text-green-400">${formatCurrency(s.total_value)}</div>
              <div class="text-gray-400">Total Value</div>
            </div>
            <div class="text-center p-4 bg-gray-800/30 rounded-lg">
              <div class="text-2xl font-bold text-blue-400">${formatCurrency(s.available_cash || 0)}</div>
              <div class="text-gray-400">Available Cash</div>
            </div>
            <div class="text-center p-4 bg-gray-800/30 rounded-lg">
              <div class="text-2xl font-bold text-purple-400">${s.holdings.length}</div>
              <div class="text-gray-400">Holdings</div>
            </div>
            <div class="text-center p-4 bg-gray-800/30 rounded-lg">
              <div class="text-2xl font-bold text-yellow-400">${formatCurrency(
                s.holdings.reduce((a, h) => a + h.unrealized_gain, 0)
              )}</div>
              <div class="text-gray-400">Total P&L</div>
            </div>
          </div>
          <div class="space-y-3">
            ${s.holdings
              .map(
                h => `
              <div class="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
                <div class="flex items-center space-x-4">
                  <div class="font-bold text-lg text-white">${h.ticker}</div>
                  <div class="text-gray-400">${h.shares} shares</div>
                </div>
                <div class="text-right">
                  <div class="font-semibold text-white">${formatCurrency(h.current_value)}</div>
                  <div class="text-sm ${
                    h.unrealized_gain >= 0 ? "text-green-400" : "text-red-400"
                  }">${h.unrealized_gain >= 0 ? "+" : ""}${formatCurrency(
                  h.unrealized_gain
                )} (${formatPercentage(h.allocation)})</div>
                </div>
              </div>`
              )
              .join("")}
          </div>
        </div>
  
        <!-- Enhanced Suggested Allocation Section -->
        <div class="glass-card p-6 rounded-xl">
          <h4 class="text-2xl text-white mb-4 flex items-center">
            <i data-lucide="layers" class="w-6 h-6 mr-2 text-teal-400"></i>Detailed Investment Strategy
          </h4>
          
          ${Object.keys(allocDetails).length > 0 ? `
            <div class="mb-6">
              <h5 class="text-lg font-semibold text-gray-300 mb-3">Recommended Allocations</h5>
              <div class="overflow-x-auto">
                <table class="w-full text-left text-gray-300">
                  <thead>
                    <tr class="text-white border-b border-gray-600">
                      <th class="p-3 text-left">Stock</th>
                      <th class="p-3 text-right">Allocation %</th>
                      <th class="p-3 text-right">Investment Amount</th>
                      <th class="p-3 text-right">Shares to Buy</th>
                      <th class="p-3 text-right">Current Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${Object.entries(allocDetails)
                      .sort(([,a], [,b]) => b.percentage - a.percentage)
                      .map(([ticker, details]) => `
                      <tr class="border-b border-gray-700/50 hover:bg-gray-800/30">
                        <td class="p-3 font-semibold text-white">${ticker}</td>
                        <td class="p-3 text-right text-blue-400 font-semibold">${safeFixed(details.percentage, 1)}%</td>
                        <td class="p-3 text-right text-green-400 font-semibold">${formatCurrency(details.investment_amount)}</td>
                        <td class="p-3 text-right text-purple-400">${safeFixed(details.shares_to_buy, 4)}</td>
                        <td class="p-3 text-right text-gray-300">${formatCurrency(details.current_price)}</td>
                      </tr>
                    `).join("")}
                  </tbody>
                </table>
              </div>
              
              <div class="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <div class="flex items-center space-x-2 mb-2">
                  <i data-lucide="info" class="w-4 h-4 text-blue-400"></i>
                  <span class="text-blue-400 font-semibold">Investment Summary</span>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span class="text-gray-400">Total to Invest: </span>
                    <span class="text-white font-semibold">${formatCurrency(
                      Object.values(allocDetails).reduce((sum, d) => sum + d.investment_amount, 0)
                    )}</span>
                  </div>
                  <div>
                    <span class="text-gray-400">Total Stocks: </span>
                    <span class="text-white font-semibold">${Object.keys(allocDetails).length}</span>
                  </div>
                  <div>
                    <span class="text-gray-400">Remaining Cash: </span>
                    <span class="text-white font-semibold">${formatCurrency(
                      s.total_value - Object.values(allocDetails).reduce((sum, d) => sum + d.investment_amount, 0)
                    )}</span>
                  </div>
                </div>
              </div>
            </div>
          ` : `
            <div class="text-center text-gray-400 py-8">
              <i data-lucide="alert-circle" class="w-12 h-12 mx-auto mb-4 text-gray-500"></i>
              <p>No positive return stocks available for allocation</p>
            </div>
          `}
          
          <!-- Simple percentage allocation as backup -->
          <div class="mt-6">
            <h5 class="text-lg font-semibold text-gray-300 mb-3">All Stock Allocations</h5>
            <div class="overflow-x-auto">
              <table class="w-full text-left text-gray-300">
                <thead>
                  <tr class="text-white">
                    <th class="p-2">Ticker</th>
                    <th class="p-2 text-right">Allocation %</th>
                  </tr>
                </thead>
                <tbody>
                  ${Object.entries(r.allocations || {})
                    .sort(([, a], [, b]) => b - a)
                    .map(([t, p]) => `
                    <tr class="${p > 0 ? 'text-white' : 'text-gray-500'}">
                      <td class="p-2 font-semibold">${t}</td>
                      <td class="p-2 text-right">${safeFixed(p)}%</td>
                    </tr>`
                    ).join("")}
                </tbody>
              </table>
            </div>
          </div>
        </div>
  
        <!-- AI Recommendations Section (unchanged) -->
        <div class="glass-card p-6 rounded-xl">
          <h4 class="text-2xl text-white mb-4 flex items-center">
            <i data-lucide="brain" class="w-6 h-6 mr-2 text-purple-400"></i>AI Recommendations
          </h4>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
              <h5 class="font-bold text-green-400 mb-2 flex items-center">
                <i data-lucide="trending-up" class="w-4 h-4 mr-1"></i>Buy
              </h5>
              ${(rec.buy || [])
                .map(b => `
                  <div class="text-sm text-gray-300 mb-1">
                    <span class="font-semibold">${b.ticker}</span> 
                    <span class="text-green-400">${formatPercentage(b.predicted_return)}</span>
                    <div class="text-xs text-gray-400">@ ${formatCurrency(b.current_price)}</div>
                  </div>`
                ).join("") || '<div class="text-gray-500 text-sm">None</div>'}
            </div>
            <div class="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
              <h5 class="font-bold text-yellow-400 mb-2 flex items-center">
                <i data-lucide="minus" class="w-4 h-4 mr-1"></i>Hold
              </h5>
              ${(rec.hold || [])
                .map(h => `
                  <div class="text-sm text-gray-300 mb-1">
                    <span class="font-semibold">${h.ticker}</span> 
                    <span class="text-yellow-400">${formatPercentage(h.predicted_return)}</span>
                    <div class="text-xs text-gray-400">@ ${formatCurrency(h.current_price)}</div>
                  </div>`
                ).join("") || '<div class="text-gray-500 text-sm">None</div>'}
            </div>
            <div class="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
              <h5 class="font-bold text-red-400 mb-2 flex items-center">
                <i data-lucide="trending-down" class="w-4 h-4 mr-1"></i>Sell
              </h5>
              ${(rec.sell || [])
                .map(s => `
                  <div class="text-sm text-gray-300 mb-1">
                    <span class="font-semibold">${s.ticker}</span> 
                    <span class="text-red-400">${formatPercentage(s.predicted_return)}</span>
                    <div class="text-xs text-gray-400">@ ${formatCurrency(s.current_price)}</div>
                  </div>`
                ).join("") || '<div class="text-gray-500 text-sm">None</div>'}
            </div>
          </div>
        </div>
  
        <!-- Top Stocks Section (unchanged) -->
        <div class="glass-card p-6 rounded-xl">
          <h4 class="text-2xl text-white mb-4 flex items-center">
            <i data-lucide="star" class="w-6 h-6 mr-2 text-yellow-400"></i>Top Stocks
          </h4>
          <div class="space-y-2">
            ${r.top_stocks
              .slice(0, 5)
              .map((s, idx) => `
              <div class="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                <div class="flex items-center space-x-3">
                  <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold text-sm">${idx + 1}</div>
                  <div>
                    <div class="font-semibold text-white">${s.ticker}</div>
                    <div class="text-xs text-gray-400">${formatCurrency(s.current_price)}</div>
                  </div>
                </div>
                <div class="text-right">
                  <div class="text-green-400 font-semibold">${formatPercentage(s.predicted_return)}</div>
                  <div class="text-xs text-gray-400">→ ${formatCurrency(s.predicted_price)}</div>
                </div>
              </div>`
              ).join("")}
          </div>
        </div>
      </div>`;
    lucide.createIcons();
  }

async function loadTop5() {
  const c = document.getElementById("top5-container");
  c.innerHTML =
    '<div class="col-span-full text-center text-blue-400">Loading…</div>';
  try {
    const r = await fetchAPI("/analyze_portfolio", {
      method: "POST",
      body: JSON.stringify({ cash: 0, wallet: [] }),
    });
    c.innerHTML = r.top_stocks
      .slice(0, 5)
      .map(
        s => `
      <div class="glass-card p-6 rounded-2xl">
        <h3 class="text-2xl font-bold text-white mb-2">${s.ticker}</h3>
        <p class="text-gray-400 mb-3">${formatCurrency(
          s.current_price
        )} → <span class="text-green-400">${formatCurrency(
          s.predicted_price
        )}</span></p>
        <p class="text-lg font-bold ${
          s.predicted_return >= 0 ? "text-green-400" : "text-red-400"
        }">${s.predicted_return >= 0 ? "+" : ""}${formatPercentage(
          s.predicted_return
        )}</p>
      </div>`
      )
      .join("");
  } catch {
    c.innerHTML =
      '<div class="col-span-full text-center text-red-400">Failed to load forecasts.</div>';
  }
}

function validateStockForm() {
  const t = document.getElementById("stock-ticker").value.trim().toUpperCase();
  if (!t || t.length < 1 || t.length > 5 || !/^[A-Z]+$/.test(t))
    throw new Error("Invalid ticker");
  return t;
}

async function searchStock(e) {
  e.preventDefault();
  try {
    const t = validateStockForm();
    showLoading("stock");
    const res = await fetchAPI(`/stock_details/${t}`);
    showResults("stock");
    renderStockResults(res);
  } catch (err) {
    showError("stock", `Error: ${err.message}`);
  }
}

function renderStockResults(r) {
  const c = document.getElementById("stock-analysis-content");
  c.innerHTML = `
    <div class="space-y-6">
      <div class="glass-card p-6 rounded-xl">
        <div class="flex flex-wrap items-center justify-between mb-6">
          <div>
            <h3 class="text-3xl font-bold text-white">${r.ticker}</h3>
            <p class="text-gray-400">${r.sector} • Market Cap ${formatCurrency(
    r.market_cap / 1e9
  )}B</p>
          </div>
          <div class="text-right">
            <div class="text-2xl font-bold text-white">${formatCurrency(
              r.current_price
            )}</div>
            <div class="text-sm text-gray-400">Current</div>
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="text-center p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <div class="text-xl font-bold text-blue-400">${formatCurrency(
              r.predicted_price
            )}</div>
            <div class="text-gray-400">Predicted</div>
          </div>
          <div class="text-center p-4 bg-${
            r.predicted_return >= 0 ? "green" : "red"
          }-500/10 border border-${
    r.predicted_return >= 0 ? "green" : "red"
  }-500/30 rounded-lg">
            <div class="text-xl font-bold text-${
              r.predicted_return >= 0 ? "green" : "red"
            }-400">${r.predicted_return >= 0 ? "+" : ""}${formatPercentage(
    r.predicted_return
  )}</div>
            <div class="text-gray-400">Return</div>
          </div>
        </div>
      </div>
    </div>`;
  if (r.historical_data && r.historical_data.length) {
    renderStockChart(r.historical_data, r.ticker);
  }
  lucide.createIcons();
}

function renderStockChart(data, ticker) {
  const ctx = document.getElementById("stock-chart").getContext("2d");
  if (myStockChart) myStockChart.destroy();
  const labels = data.map(d =>
    new Date(d.date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    })
  );
  const prices = data.map(d => d.price);
  myStockChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `${ticker} Price`,
          data: prices,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,0.1)",
          fill: true,
          tension: 0.4,
          pointRadius: 2,
          pointHoverRadius: 6,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      scales: {
        x: {
          ticks: { color: "#9ca3af", maxTicksLimit: 10 },
          grid: { color: "rgba(255,255,255,0.1)" },
        },
        y: {
          ticks: { color: "#9ca3af", callback: v => formatCurrency(v) },
          grid: { color: "rgba(255,255,255,0.1)" },
        },
      },
      plugins: {
        legend: { labels: { color: "#e5e7eb" } },
        tooltip: {
          backgroundColor: "rgba(0,0,0,0.8)",
          titleColor: "#fff",
          bodyColor: "#e5e7eb",
          borderColor: "#3b82f6",
          borderWidth: 1,
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${formatCurrency(ctx.parsed.y)}`,
          },
        },
      },
    },
  });
}

function openLoginModal() {
  document.getElementById("loginModal").classList.remove("hidden");
}
function openMemberModal(m) {
  const info = {
    prince: { name: "Prince Yadav", role: "CEO & AI Architect", color: "blue" },
    dakshita: { name: "Dakshita Jindal", role: "Head of Product", color: "purple" },
    anshu: { name: "Anshu", role: "Lead Data Scientist", color: "green" },
    sparsh: { name: "Sparsh", role: "Head of UX/UI Design", color: "pink" },
  };
  const i = info[m];
  if (!i) return;
  const c = document.getElementById("member-modal-content");
  c.innerHTML = `
    <div class="text-center">
      <img src="https://placehold.co/120x120/${
        i.color === "blue"
          ? "3b82f6"
          : i.color === "purple"
          ? "8b5cf6"
          : i.color === "green"
          ? "34d399"
          : "f472b6"
      }/ffffff?text=${i.name
        .split(" ")
        .map(n => n[0])
        .join("")}" class="mx-auto rounded-full mb-4" alt="${i.name}">
      <h3 class="text-2xl font-bold text-white mb-2">${i.name}</h3>
      <p class="text-${i.color}-400 mb-4">${i.role}</p>
      <p class="text-gray-300 leading-relaxed">Fictional team member.</p>
    </div>`;
  document.getElementById("memberModal").classList.remove("hidden");
}
function closeModal(id) {
  document.getElementById(id).classList.add("hidden");
}

function initScrollAnimations() {
  const obs = new IntersectionObserver(
    e => e.forEach(i => i.isIntersecting && i.target.classList.add("visible")),
    { threshold: 0.1, rootMargin: "0px 0px -50px 0px" }
  );
  document.querySelectorAll(".section-fade-in").forEach(s => obs.observe(s));
}

function initTickerInputs() {
  document.getElementById("stock-ticker").addEventListener("input", e => {
    e.target.value = e.target.value.toUpperCase();
  });
}

async function testConnection() {
  try {
    updateConnectionStatus("connecting");
    await fetchAPI("/status");
    updateConnectionStatus("connected");
  } catch {}
}

document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  addHoldingRow("AAPL", 10, 200);
  initScrollAnimations();
  initTickerInputs();
  testConnection();
  loadTop5();
  window.addEventListener(
    "resize",
    debounce(() => myStockChart && myStockChart.resize(), 250)
  );
});

document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    ["loginModal", "memberModal"].forEach(id => {
      const m = document.getElementById(id);
      if (!m.classList.contains("hidden")) closeModal(id);
    });
  }
});

window.addEventListener("error", e => console.error(e.error));
window.addEventListener("unhandledrejection", e => console.error(e.reason));