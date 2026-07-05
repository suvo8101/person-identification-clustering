/**
 * Contact Sheet frontend.
 * Talks to the FastAPI backend to: create a session, upload images,
 * trigger clustering, and render the results as a filmstrip contact sheet.
 *
 * Configure API_BASE if the backend isn't on localhost:8000.
 */
const API_BASE = window.API_BASE || "http://localhost:8000";

const els = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  stagedList: document.getElementById("stagedList"),
  developBtn: document.getElementById("developBtn"),
  statusLine: document.getElementById("statusLine"),
  epsSlider: document.getElementById("epsSlider"),
  epsValue: document.getElementById("epsValue"),
  loaderPanel: document.getElementById("loaderPanel"),
  resultsPanel: document.getElementById("resultsPanel"),
  resultsHeading: document.getElementById("resultsHeading"),
  resultsSummary: document.getElementById("resultsSummary"),
  clusterGrid: document.getElementById("clusterGrid"),
  unmatchedPanel: document.getElementById("unmatchedPanel"),
  unmatchedList: document.getElementById("unmatchedList"),
  rollCounter: document.getElementById("rollCounter"),
  resetBtn: document.getElementById("resetBtn"),
};

let stagedFiles = [];
let sessionId = null;

function setStatus(message, state) {
  els.statusLine.textContent = message;
  if (state) els.statusLine.setAttribute("data-state", state);
  else els.statusLine.removeAttribute("data-state");
}

function renderStagedList() {
  els.stagedList.innerHTML = "";
  stagedFiles.forEach((file) => {
    const chip = document.createElement("span");
    chip.className = "staged-chip";
    chip.textContent = file.name;
    els.stagedList.appendChild(chip);
  });
  els.developBtn.disabled = stagedFiles.length === 0;
  els.rollCounter.textContent = `ROLL ${String(stagedFiles.length).padStart(3, "0")} · ${stagedFiles.length} FRAMES`;
}

function addFiles(fileList) {
  const incoming = Array.from(fileList).filter((f) =>
    /\.(jpe?g|png|bmp|webp)$/i.test(f.name)
  );
  const existingNames = new Set(stagedFiles.map((f) => f.name));
  incoming.forEach((f) => {
    if (!existingNames.has(f.name)) stagedFiles.push(f);
  });
  renderStagedList();
}

els.fileInput.addEventListener("change", (e) => addFiles(e.target.files));

["dragenter", "dragover"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.style.borderColor = "var(--safelight)";
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.style.borderColor = "";
  })
);
els.dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

els.epsSlider.addEventListener("input", () => {
  els.epsValue.textContent = Number(els.epsSlider.value).toFixed(2);
});

async function createSession() {
  const res = await fetch(`${API_BASE}/api/sessions`, { method: "POST" });
  if (!res.ok) throw new Error("Could not start a session on the backend.");
  const data = await res.json();
  return data.session_id;
}

async function uploadFiles(id) {
  const form = new FormData();
  stagedFiles.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_BASE}/api/sessions/${id}/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Upload failed.");
  return res.json();
}

async function processSession(id, eps) {
  const url = `${API_BASE}/api/sessions/${id}/process?eps=${eps}&min_samples=1`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Clustering failed.");
  }
  return res.json();
}

function confidenceClass(score) {
  if (score >= 75) return "stamp--high";
  if (score >= 50) return "stamp--mid";
  return "stamp--low";
}

function renderResults(data) {
  els.resultsHeading.textContent = `${data.num_clusters} identit${data.num_clusters === 1 ? "y" : "ies"} found`;
  els.resultsSummary.textContent =
    `${data.total_images} IMAGES · ${data.total_faces_detected} FACES DETECTED · ` +
    `EPS=${data.eps} · ${data.images_with_no_face.length} IMAGE(S) WITH NO FACE`;

  els.clusterGrid.innerHTML = "";

  if (data.clusters.length === 0) {
    els.clusterGrid.innerHTML = `<p class="empty-state">No faces detected in this batch.</p>`;
  }

  data.clusters.forEach((cluster) => {
    const strip = document.createElement("article");
    strip.className = "cluster-strip";

    const header = document.createElement("div");
    header.className = "cluster-strip__header";
    header.innerHTML = `
      <div class="cluster-strip__title">
        <h3>${cluster.label}</h3>
        <span class="cluster-strip__count">${cluster.size} frame${cluster.size === 1 ? "" : "s"}</span>
      </div>
      <span class="cluster-strip__avg mono">AVG CONF ${cluster.avg_confidence.toFixed(1)}%</span>
    `;
    strip.appendChild(header);

    const row = document.createElement("div");
    row.className = "frame-row";

    cluster.members.forEach((face, idx) => {
      const frame = document.createElement("div");
      frame.className = "frame" + (face.is_singleton ? " frame--singleton" : "");

      const img = document.createElement("img");
      img.src = `${API_BASE}${face.crop_url}`;
      img.alt = `Face crop from ${face.image_filename}`;
      img.loading = "lazy";
      frame.appendChild(img);

      const num = document.createElement("span");
      num.className = "frame__number";
      num.textContent = `#${String(idx + 1).padStart(2, "0")}`;
      frame.appendChild(num);

      const stamp = document.createElement("span");
      stamp.className = `stamp ${confidenceClass(face.confidence)}`;
      stamp.textContent = face.is_singleton ? "UNIQUE" : `${face.confidence.toFixed(0)}%`;
      frame.appendChild(stamp);

      frame.title = `${face.image_filename} — confidence ${face.confidence.toFixed(1)}%`;
      row.appendChild(frame);
    });

    strip.appendChild(row);
    els.clusterGrid.appendChild(strip);
  });

  if (data.images_with_no_face.length > 0) {
    els.unmatchedPanel.hidden = false;
    els.unmatchedList.innerHTML = "";
    data.images_with_no_face.forEach((name) => {
      const chip = document.createElement("span");
      chip.className = "staged-chip";
      chip.textContent = name;
      els.unmatchedList.appendChild(chip);
    });
  } else {
    els.unmatchedPanel.hidden = true;
  }

  els.loaderPanel.hidden = true;
  els.resultsPanel.hidden = false;
}

els.developBtn.addEventListener("click", async () => {
  if (stagedFiles.length === 0) return;
  els.developBtn.disabled = true;
  try {
    setStatus(`Starting session…`, "busy");
    sessionId = await createSession();

    setStatus(`Uploading ${stagedFiles.length} image(s)…`, "busy");
    await uploadFiles(sessionId);

    setStatus(`Detecting faces & clustering identities…`, "busy");
    const eps = Number(els.epsSlider.value);
    const results = await processSession(sessionId, eps);

    setStatus(`Done. ${results.num_clusters} identities from ${results.total_faces_detected} faces.`, "done");
    renderResults(results);
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Something went wrong.", "error");
  } finally {
    els.developBtn.disabled = false;
  }
});

els.resetBtn.addEventListener("click", () => {
  stagedFiles = [];
  sessionId = null;
  els.fileInput.value = "";
  renderStagedList();
  setStatus("");
  els.resultsPanel.hidden = true;
  els.loaderPanel.hidden = false;
});
