let chart;

/**
 * Main function to fetch data and update the dashboard.
 */
async function loadData() {
  const selectElement = document.getElementById("regressionSelect");
  const type = selectElement.value;

  try {
    // 1. Fetch Data from Flask backend
    const res = await fetch("/get_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type }),
    });

    if (!res.ok) {
      throw new Error(`Server returned status: ${res.status}`);
    }

    const data = await res.json();

    // 2. Update Description
    document.getElementById("description").innerHTML = data.description;

    // 3. Display Metrics
    updateMetrics(data.metrics);

    // 4. Render Chart
    renderChart(data);
  } catch (error) {
    console.error("Error loading data:", error);
    document.getElementById(
      "description"
    ).innerHTML = `<div style="color: red; padding: 10px; border: 1px solid red; border-radius: 4px;">
                Error fetching data: ${error.message}. Check Flask server.
            </div>`;
  }
}

/**
 * Updates the metrics list with key-value pairs.
 * @param {object} metrics - Object containing model metrics.
 */
function updateMetrics(metrics) {
  const metricsList = document.getElementById("metrics");
  metricsList.innerHTML = "";

  for (const [key, value] of Object.entries(metrics)) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${key}:</span> <strong>${value}</strong>`;
    metricsList.appendChild(li);
  }
}

/**
 * Renders the Chart.js visualization.
 * @param {object} data - The data object returned from the backend.
 */
function renderChart(data) {
  const ctx = document.getElementById("regressionChart").getContext("2d");
  if (chart) chart.destroy();

  const type = document.getElementById("regressionSelect").value;
  const isClassification = type === "logistic" || type === "knn";
  const fitLabel = isClassification
    ? "Probability / Prediction"
    : "Regression Fit";
  const fitColor = isClassification ? "green" : "red";

  // Configuration for the Chart
  chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: isClassification ? "Actual Class (0/1)" : "Data Points",
          data: data.x.map((x, i) => ({ x: x, y: data.y[i] })),
          backgroundColor: "blue",
          pointRadius: 6,
          pointStyle: isClassification ? "rect" : "circle",
        },
        {
          label: fitLabel,
          // Use the smooth 'x_fit' and 'fit' arrays for the line/curve
          data: data.x_fit.map((x, i) => ({ x: x, y: data.fit[i] })),
          type: "line",
          borderColor: fitColor,
          borderWidth: 3,
          fill: false,
          showLine: true,
          pointRadius: 0,
          tension:
            type === "polynomial" || type === "logistic" || type === "knn"
              ? 0.4
              : 0,
        },
      ],
    },
    options: {
      responsive: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: { display: true, text: "Feature (X)" },
        },
        y: {
          title: {
            display: true,
            text: isClassification ? "Probability / Value" : "Y",
          },
          ...(isClassification ? { min: -0.1, max: 1.1 } : {}),
        },
      },
    },
  });
}

// Attach the loadData function to the button click, select change, and initial load
document.addEventListener("DOMContentLoaded", () => {
  const selectElement = document.getElementById("regressionSelect");

  // 1. Trigger loadData when the dropdown selection changes
  selectElement.addEventListener("change", loadData);

  // 2. Initial load (for the default Linear Regression model)
  loadData();

  // NOTE: The button already has onclick="loadData()" in the HTML, so no need to re-attach it here.
});
