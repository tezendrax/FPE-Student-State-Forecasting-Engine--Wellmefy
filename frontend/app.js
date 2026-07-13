// Wellmate Future Prediction Engine Dashboard Controller

const API_BASE = "http://localhost:8003/api/v1/predictions";
const API_TOKEN = "wellmate-secure-token";

// Chart.js instance variable
let forecastChart = null;
let currentForecastData = null;

// Global state variables
let isServerOnline = false;

// Color mapping for different target dimensions to enhance visual aesthetics
const DIMENSION_THEMES = {
    burnout: { color: "#f43f5e", fill: "rgba(244, 63, 94, 0.15)", title: "Burnout Risk Forecast" },
    stress: { color: "#eab308", fill: "rgba(234, 179, 8, 0.15)", title: "Stress Level Forecast" },
    anxiety: { color: "#a855f7", fill: "rgba(168, 85, 247, 0.15)", title: "Anxiety Index Forecast" },
    fatigue: { color: "#f97316", fill: "rgba(249, 115, 22, 0.15)", title: "Fatigue Indicator Forecast" },
    sleep: { color: "#06b6d4", fill: "rgba(6, 182, 212, 0.15)", title: "Sleep Quality Forecast" },
    mood: { color: "#10b981", fill: "rgba(16, 185, 129, 0.15)", title: "Mood Trajectory Forecast" },
    focus: { color: "#3b82f6", fill: "rgba(59, 130, 246, 0.15)", title: "Focus Efficiency Forecast" },
    social: { color: "#ec4899", fill: "rgba(236, 72, 153, 0.15)", title: "Social Activity Forecast" },
    academic: { color: "#14b8a6", fill: "rgba(20, 184, 166, 0.15)", title: "Academic Dedication Forecast" },
    resilience: { color: "#6366f1", fill: "rgba(99, 102, 241, 0.15)", title: "Resilience Coefficient Forecast" }
};

document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    checkEngineHealth();
    
    // Automatically trigger initial forecast generation
    setTimeout(() => {
        generateForecast();
    }, 500);
});

// Setup DOM event listeners
function initEventListeners() {
    document.getElementById("btn-fetch").addEventListener("click", () => {
        generateForecast(true);
    });
    
    document.getElementById("dimension-select").addEventListener("change", () => {
        updateChartAndMatrix();
    });
    
    const studentSelect = document.getElementById("student-select");
    const customGroup = document.getElementById("custom-student-group");
    const customInput = document.getElementById("student-custom-input");
    
    // Show/hide custom input group and trigger forecasts
    studentSelect.addEventListener("change", () => {
        if (studentSelect.value === "custom") {
            customGroup.classList.remove("form-group-hidden");
            customInput.focus();
        } else {
            customGroup.classList.add("form-group-hidden");
            generateForecast(true);
        }
    });
    
    // Allow pressing enter in custom input
    customInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            generateForecast(true);
        }
    });
}

// Connect to health endpoint to check backend server availability
async function checkEngineHealth() {
    const statusDot = document.querySelector(".status-dot");
    const statusLabel = document.querySelector(".status-label");
    
    try {
        const response = await fetch(`${API_BASE}/health`, {
            headers: { "X-Access-Token": API_TOKEN }
        });
        
        if (response.ok) {
            const data = await response.json();
            isServerOnline = true;
            statusDot.className = "status-dot status-online";
            statusLabel.textContent = "FPE Backend Online";
            document.getElementById("cache-type").textContent = data.cache_type;
        } else {
            throw new Error("Degraded health status response");
        }
    } catch (e) {
        console.warn("Could not connect to FPE backend server. Standalone simulation mode active.");
        isServerOnline = false;
        statusDot.className = "status-dot status-offline";
        statusLabel.textContent = "Offline (Simulation Mode)";
        document.getElementById("cache-type").textContent = "LocalMemory";
    }
}

// Generate the 7-day forecast
async function generateForecast(forceRefresh = false) {
    const studentSelect = document.getElementById("student-select");
    let studentId = studentSelect.value;
    if (studentId === "custom") {
        studentId = document.getElementById("student-custom-input").value.trim();
        if (!studentId) {
            return; // Exit silently or do nothing
        }
    }
    const btn = document.getElementById("btn-fetch");
    const originalText = btn.innerHTML;
    
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Processing...`;
    
    try {
        if (isServerOnline) {
            // Fetch from FastAPI server
            const response = await fetch(`${API_BASE}/forecast?student_id=${studentId}&force_refresh=${forceRefresh}`, {
                headers: { "X-Access-Token": API_TOKEN }
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Server error generating forecast");
            }
            
            currentForecastData = await response.json();
            displayServerDiagnostics(currentForecastData);
        } else {
            // Offline Mode: Simulate model inference sequence
            currentForecastData = simulateForecastData(studentId);
            displayServerDiagnostics(currentForecastData);
        }
        
        updateChartAndMatrix();
        
    } catch (err) {
        alert(`Error: ${err.message}`);
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// Display diagnostics from API response
function displayServerDiagnostics(data) {
    document.getElementById("diag-latency").textContent = `${Math.round(data.latency_ms)} ms`;
    
    const fallbackBadge = document.getElementById("badge-fallback");
    const anomalyBadge = document.getElementById("badge-anomaly");
    
    if (data.fallback_used) {
        fallbackBadge.classList.remove("badge-hidden");
    } else {
        fallbackBadge.classList.add("badge-hidden");
    }
    
    if (data.anomaly_warning) {
        anomalyBadge.classList.remove("badge-hidden");
    } else {
        anomalyBadge.classList.add("badge-hidden");
    }
}

// Update the chart display and details table
function updateChartAndMatrix() {
    if (!currentForecastData) return;
    
    const dimension = document.getElementById("dimension-select").value;
    const theme = DIMENSION_THEMES[dimension];
    
    // Update chart title
    document.getElementById("chart-display-title").textContent = theme.title;
    
    // Extract series for selected dimension
    const labels = currentForecastData.forecast.map(item => `Day ${item.day}`);
    const p10_vals = currentForecastData.forecast.map(item => item[`${dimension}_p10`]);
    const p50_vals = currentForecastData.forecast.map(item => item[`${dimension}_p50`]);
    const p90_vals = currentForecastData.forecast.map(item => item[`${dimension}_p90`]);
    
    // Render Chart
    renderChart(labels, p10_vals, p50_vals, p90_vals, theme);
    
    // Render Matrix Table
    renderMatrixTable(p10_vals, p50_vals, p90_vals);
}

// Draw the line chart with median and dashed confidence bounds
function renderChart(labels, p10, p50, p90, theme) {
    const ctx = document.getElementById("forecastChart").getContext("2d");
    
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Upper Bound (p90)',
                    data: p90,
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointStyle: 'none',
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Median Prediction (p50)',
                    data: p50,
                    borderColor: theme.color,
                    borderWidth: 3.5,
                    backgroundColor: theme.fill,
                    fill: false,
                    pointBackgroundColor: theme.color,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.2
                },
                {
                    label: 'Lower Bound (p10)',
                    data: p10,
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointStyle: 'none',
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#90a0b8',
                        font: { family: 'Inter', size: 12 },
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: '#121624',
                    titleColor: '#fff',
                    bodyColor: '#90a0b8',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            const val = context.raw.toFixed(3);
                            return `${context.dataset.label}: ${val}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#90a0b8', font: { family: 'Inter' } }
                },
                y: {
                    min: 0.0,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#90a0b8', font: { family: 'Inter' } }
                }
            }
        }
    });
}

// Populate the trajectory matrix table
function renderMatrixTable(p10, p50, p90) {
    const tbody = document.querySelector("#forecast-table tbody");
    tbody.innerHTML = "";
    
    for (let i = 0; i < p50.length; i++) {
        const tr = document.createElement("tr");
        
        // Calculate day-to-day trend
        let trendHtml = "";
        if (i === 0) {
            trendHtml = `<span class="trend-stable"><i class="fa-solid fa-minus"></i> Starting</span>`;
        } else {
            const diff = p50[i] - p50[i-1];
            if (Math.abs(diff) < 0.015) {
                trendHtml = `<span class="trend-stable"><i class="fa-solid fa-minus"></i> Stable</span>`;
            } else if (diff > 0) {
                trendHtml = `<span class="trend-up"><i class="fa-solid fa-arrow-trend-up"></i> Rising (+${Math.round(diff * 100)}%)</span>`;
            } else {
                trendHtml = `<span class="trend-down"><i class="fa-solid fa-arrow-trend-down"></i> Declining (${Math.round(diff * 100)}%)</span>`;
            }
        }
        
        tr.innerHTML = `
            <td>Day ${i + 1}</td>
            <td>${p10[i].toFixed(3)}</td>
            <td><strong>${p50[i].toFixed(3)}</strong></td>
            <td>${p90[i].toFixed(3)}</td>
            <td>${trendHtml}</td>
        `;
        
        tbody.appendChild(tr);
    }
}

// Simulate student forecast data for client-side offline mode
function simulateForecastData(studentId) {
    console.log(`Simulating trajectory for offline student ${studentId}`);
    
    const forecastList = [];
    
    // Generate simulated curves for all dimensions
    const seed = getStudentHash(studentId);
    
    for (let day = 1; day <= 7; day++) {
        const day_rec = { day: day };
        
        Object.keys(DIMENSION_THEMES).forEach((dim) => {
            let base = 0.4 + 0.2 * Math.sin(seed + day * 0.5);
            let noise = 0.05 * Math.cos(seed * day);
            
            // Adjust specific shapes for realistic wellness paths
            if (dim === "burnout") {
                base = 0.15 + 0.06 * day; // steadily rising
            } else if (dim === "stress") {
                base = 0.3 + 0.1 * Math.sin(day * 0.8) + 0.08 * day; // exam workload peak
            } else if (dim === "sleep") {
                base = 0.78 - 0.05 * day; // sleep drop
            } else if (dim === "mood") {
                base = 0.7 - 0.04 * day; // mood decline
            }
            
            let p50 = Math.max(0.05, Math.min(0.95, base + noise));
            let p10 = Math.max(0.01, p50 - 0.08 - 0.02 * day); // boundary dilates over time
            let p90 = Math.min(0.99, p50 + 0.08 + 0.02 * day);
            
            day_rec[`${dim}_p10`] = p10;
            day_rec[`${dim}_p50`] = p50;
            day_rec[`${dim}_p90`] = p90;
        });
        
        forecastList.append(day_rec);
    }
    
    return {
        student_id: studentId,
        horizon_days: 7,
        forecast_epoch: Math.round(Date.now() / 1000),
        fallback_used: false,
        anomaly_warning: false,
        latency_ms: 12 + Math.random() * 8, // simulated local CPU time
        forecast: forecastList
    }
}

// Simple hash generator helper
function getStudentHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    return Math.abs(hash) % 100;
}

// Polyfill for Array.append since it was used in simulateForecastData
if (!Array.prototype.append) {
    Array.prototype.append = Array.prototype.push;
}
