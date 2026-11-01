/**
 * Reusable Visualization Components & Interactive Views
 * for the MedSSL-BHSD Research Method Visualization System.
 * Uses semantic theme variables (--background, --card, --border, --foreground, --muted-foreground)
 * and renders interactive SVG/Canvas diagrams, live tensor badges, code drawers, and simulations.
 */

window.MedSSLViews = (function () {
  "use strict";

  const Core = window.MedSSLCore;
  const DATA = window.MEDSSL_BUNDLE;

  // Global Reactive State
  const state = {
    activeView: "pipeline", // 'pipeline' | 'tensors' | 'math' | 'schedules' | 'grad_ema' | 'forensics'
    method: "mean_teacher",
    activePreset: "mean_teacher_yaml",
    selectedNodeId: "student_net",
    currentStep: 1,
    isPlaying: false,
    playTimer: null,

    // Hyperparameters (initialized from configs/methods/mean_teacher.yaml)
    backbone: "nnunet3d",
    num_stages: 7,
    anisotropic_kernels: true,
    deep_supervision: true,
    consistency_multiscale: false,
    patch_z: 28,
    patch_y: 256,
    patch_x: 256,
    batch_size: 8,
    labeled_bs: 4,
    world_size: 2,
    labelnum: 154,
    num_classes: 6,

    max_iteration: 12000,
    iter_num: 4500,
    base_lr: 0.01,
    lr_scheduler: "poly",
    poly_power: 0.7,
    warmup_iters: 0,

    consistency: 0.05,
    consistency_rampup: 4000.0,
    consistency_rampdown_start: 7800,
    consistency_min_factor: 0.25,
    consistency_type: "hybrid",
    bg_consistency_weight: 0.05,
    confidence_thresh: 0.0,

    ema_decay: 0.999,
    ema_schedule: "cosine",
    perturb_student: true,
    student_noise_std: 0.02,
    student_intensity_scale: 0.05,

    // View 3 Math/Loss & Metric Paradox Sandbox state
    sim_pred_discrepancy: 0.18,
    sim_fg_overlap: 0.72,
    paradox_pos_dice_level: 1.0,
    paradox_fp_case_rate: 1.0,
    activeFormulaId: "hybrid_cons",

    // View 5 AMP & EMA Trajectory Lab state
    amp_simulate_overflow: false,
    traj_noise_scale: 0.35,

    // View 6 Empirical Run Forensics state
    selectedRunId: "o6s4pex3",
  };

  function applyPreset(presetKey) {
    state.activePreset = presetKey;
    if (presetKey === "mean_teacher_yaml") {
      Object.assign(state, {
        backbone: "nnunet3d",
        num_stages: 7,
        anisotropic_kernels: true,
        deep_supervision: true,
        consistency_multiscale: false,
        patch_z: 28,
        patch_y: 256,
        patch_x: 256,
        batch_size: 8,
        labeled_bs: 4,
        world_size: 2,
        labelnum: 154,
        max_iteration: 12000,
        base_lr: 0.01,
        lr_scheduler: "poly",
        poly_power: 0.7,
        consistency: 0.05,
        consistency_rampup: 4000.0,
        consistency_rampdown_start: 7800,
        consistency_min_factor: 0.25,
        consistency_type: "hybrid",
        bg_consistency_weight: 0.05,
        confidence_thresh: 0.0,
        ema_decay: 0.999,
        ema_schedule: "cosine",
        perturb_student: true,
        student_noise_std: 0.02,
        student_intensity_scale: 0.05,
      });
    } else if (presetKey === "mt_fold0_preset") {
      Object.assign(state, {
        backbone: "nnunet3d",
        num_stages: 7,
        anisotropic_kernels: true,
        deep_supervision: true,
        consistency_multiscale: false,
        patch_z: 32,
        patch_y: 192,
        patch_x: 192,
        batch_size: 4,
        labeled_bs: 2,
        world_size: 2,
        labelnum: 154,
        max_iteration: 12000,
        base_lr: 0.01,
        lr_scheduler: "poly",
        poly_power: 0.7,
        consistency: 1.0,
        consistency_rampup: 40.0, // <= 100 -> Triggers Epoch-based rampup!
        consistency_rampdown_start: 7800,
        consistency_min_factor: 0.25,
        consistency_type: "hybrid",
        bg_consistency_weight: 0.05,
        confidence_thresh: 0.0,
        ema_decay: 0.99, // Triggers flat cosine alpha = 0.99!
        ema_schedule: "cosine",
        perturb_student: true,
        student_noise_std: 0.02,
        student_intensity_scale: 0.05,
      });
    } else if (presetKey === "run_9tvq8qld") {
      Object.assign(state, {
        backbone: "nnunet3d",
        num_stages: 7,
        anisotropic_kernels: true,
        deep_supervision: true,
        consistency_multiscale: false,
        patch_z: 32,
        patch_y: 192,
        patch_x: 192,
        batch_size: 4,
        labeled_bs: 2,
        world_size: 2,
        labelnum: 154,
        max_iteration: 30000,
        iter_num: 14500,
        base_lr: 0.01,
        lr_scheduler: "poly",
        poly_power: 0.7,
        consistency: 0.05,
        consistency_rampup: 8000.0,
        consistency_rampdown_start: 19500,
        consistency_min_factor: 0.25,
        consistency_type: "hybrid",
        bg_consistency_weight: 0.05,
        confidence_thresh: 0.0,
        ema_decay: 0.999,
        ema_schedule: "cosine",
        perturb_student: true,
        student_noise_std: 0.02,
        student_intensity_scale: 0.05,
      });
    } else if (presetKey === "base_vnet") {
      Object.assign(state, {
        backbone: "vnet",
        num_stages: 5,
        anisotropic_kernels: false,
        deep_supervision: false,
        consistency_multiscale: false,
        patch_z: 32,
        patch_y: 160,
        patch_x: 160,
        batch_size: 4,
        labeled_bs: 2,
        world_size: 1,
        labelnum: 15,
        max_iteration: 12000,
        base_lr: 0.01,
        lr_scheduler: "poly",
        poly_power: 0.9,
        consistency: 0.1,
        consistency_rampup: 40.0,
        consistency_rampdown_start: 0,
        ema_decay: 0.99,
        ema_schedule: "step",
      });
    }
    if (state.iter_num > state.max_iteration) {
      state.iter_num = Math.round(state.max_iteration * 0.4);
    }
    renderApp();
  }

  function formatShape(s) {
    if (Array.isArray(s)) return `(${s.join(", ")})`;
    return String(s);
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ============================================================================
  // GLOBAL HEADER & CONTEXT BAR
  // ============================================================================
  function renderGlobalHeader(hierarchy, stepMetrics, methodSpec) {
    const methodsList = Object.values(DATA.methods);
    const tabs = [
      { id: "pipeline", label: "1. Pipeline & Step-by-Step", badge: `${methodSpec.execution_steps.length} Steps` },
      { id: "tensors", label: "2. 3D Tensors & Backbone", badge: `${hierarchy.numStages} Stages` },
      { id: "math", label: "3. Math, Loss & Metric Lab", badge: "Equations" },
      { id: "schedules", label: "4. Hyperparameter Time-Machine", badge: "Schedules" },
      { id: "grad_ema", label: "5. Gradient, AMP & EMA Lab", badge: "Dynamics" },
      { id: "forensics", label: "6. Config Diff & Run Forensics", badge: "2 Real Runs" },
    ];

    return `
      <header class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 mb-4 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[var(--border)]">
          <div>
            <div class="flex items-center gap-2">
              <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/15 text-blue-400 border border-blue-500/30">
                MedSSL-BHSD Interactive Learning System
              </span>
              <span class="px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-500/15 text-purple-400 border border-purple-500/30">
                ${escapeHtml(methodSpec.badge)}
              </span>
            </div>
            <h1 class="text-xl font-bold mt-1 text-[var(--foreground)]">
              ${escapeHtml(methodSpec.name)} — <span class="text-sm font-normal text-[var(--muted-foreground)]">Codebase-Verified Interactive Explorer (<code>${escapeHtml(methodSpec.file)}</code>)</span>
            </h1>
          </div>

          <!-- Method & Config Preset Selectors -->
          <div class="flex flex-wrap items-center gap-2">
            <div class="flex flex-col">
              <label class="text-xs text-[var(--muted-foreground)] font-medium mb-0.5">SSL Method (Method-Agnostic Engine)</label>
              <select id="select-method" class="bg-[var(--background)] text-[var(--foreground)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-xs font-semibold">
                ${methodsList
                  .map(
                    (m) =>
                      `<option value="${m.id}" ${m.id === state.method ? "selected" : ""}>${escapeHtml(
                        m.short_name
                      )} (${escapeHtml(m.file)})</option>`
                  )
                  .join("")}
              </select>
            </div>

            <div class="flex flex-col">
              <label class="text-xs text-[var(--muted-foreground)] font-medium mb-0.5">Configuration Preset</label>
              <select id="select-preset" class="bg-[var(--background)] text-[var(--foreground)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-xs font-semibold">
                <option value="mean_teacher_yaml" ${state.activePreset === "mean_teacher_yaml" ? "selected" : ""}>configs/methods/mean_teacher.yaml (28×256×256, BS=8, α=0.999)</option>
                <option value="mt_fold0_preset" ${state.activePreset === "mt_fold0_preset" ? "selected" : ""}>configs/experiments/bhsd_mt_nnunet_fold0.yaml (32×192×192, BS=4, λ=1.0)</option>
                <option value="run_9tvq8qld" ${state.activePreset === "run_9tvq8qld" ? "selected" : ""}>Run 9tvq8qld Config (32×192×192, 30k iters, rampup=8000)</option>
                <option value="base_vnet" ${state.activePreset === "base_vnet" ? "selected" : ""}>configs/base.yaml (VNet 5-stage, 32×160×160, 1 GPU)</option>
                <option value="custom" ${state.activePreset === "custom" ? "selected" : ""}>Custom Hyperparameters</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Live Iteration Scrubber & Instant Telemetry Bar -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-3 pt-3 items-center">
          <div class="lg:col-span-5 flex flex-col gap-1">
            <div class="flex items-center justify-between text-xs">
              <span class="font-semibold text-[var(--foreground)]">
                Training Iteration Scrubber: <code class="text-blue-400 font-bold">iter_num = ${state.iter_num.toLocaleString()}</code> / ${state.max_iteration.toLocaleString()}
              </span>
              <span class="text-[var(--muted-foreground)]">
                Epoch: <strong>${stepMetrics.epoch.toFixed(1)}</strong> (1 ep = ${hierarchy.epochLen} steps)
              </span>
            </div>
            <input
              id="global-iter-slider"
              type="range"
              min="0"
              max="${state.max_iteration}"
              step="50"
              value="${state.iter_num}"
              class="w-full accent-blue-500 cursor-pointer"
            />
          </div>

          <div class="lg:col-span-7 grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg px-2.5 py-1.5">
              <div class="text-xs text-[var(--muted-foreground)]">Per-Rank Batch (${state.world_size} GPU)</div>
              <div class="text-xs font-bold text-[var(--foreground)] mt-0.5">
                ${hierarchy.layout.local_batch_size} = <span class="text-emerald-400">${hierarchy.layout.local_labeled_bs}L</span> + <span class="text-amber-400">${hierarchy.layout.local_unlabeled_bs}U</span>
                <span class="text-[var(--muted-foreground)] font-normal">(${state.patch_z}×${state.patch_y}×${state.patch_x})</span>
              </div>
            </div>

            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg px-2.5 py-1.5">
              <div class="text-xs text-[var(--muted-foreground)]">Learning Rate <code>lr(t)</code></div>
              <div class="text-xs font-bold text-blue-400 mt-0.5">
                ${stepMetrics.lr.toFixed(6)} <span class="text-[var(--muted-foreground)] font-normal">(${state.lr_scheduler}^${state.poly_power})</span>
              </div>
            </div>

            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg px-2.5 py-1.5">
              <div class="text-xs text-[var(--muted-foreground)]">Consistency <code>w_cons(t)</code></div>
              <div class="text-xs font-bold text-amber-400 mt-0.5">
                ${stepMetrics.consW.toFixed(5)} <span class="text-[var(--muted-foreground)] font-normal">(${stepMetrics.inRampdown ? "Ramp-down" : "Ramp-up/Peak"})</span>
              </div>
            </div>

            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg px-2.5 py-1.5">
              <div class="text-xs text-[var(--muted-foreground)]">EMA <code>α_t</code> & Memory</div>
              <div class="text-xs font-bold text-purple-400 mt-0.5">
                ${stepMetrics.actualAlpha.toFixed(5)} <span class="text-[var(--muted-foreground)] font-normal">(~${Math.round(stepMetrics.effectiveMemorySteps)} steps)</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-[var(--border)]">
          ${tabs
            .map((t) => {
              const active = state.activeView === t.id;
              return `
                <button
                  data-view-tab="${t.id}"
                  class="px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                    active
                      ? "bg-blue-600 text-white shadow-sm"
                      : "bg-[var(--background)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] border border-[var(--border)]"
                  }"
                >
                  <span>${escapeHtml(t.label)}</span>
                  <span class="px-1.5 py-0.2 rounded text-xs ${
                    active ? "bg-blue-800 text-blue-100" : "bg-[var(--card)] text-[var(--muted-foreground)]"
                  }">${escapeHtml(t.badge)}</span>
                </button>
              `;
            })
            .join("")}
        </div>
      </header>
    `;
  }

  // ============================================================================
  // VIEW 1: INTERACTIVE PIPELINE & STEP-BY-STEP EXECUTION CANVAS
  // ============================================================================
  function renderPipelineView(hierarchy, stepMetrics, methodSpec) {
    const steps = methodSpec.execution_steps || [];
    const curStepIdx = Math.max(0, Math.min(steps.length - 1, state.currentStep - 1));
    const curStepObj = steps[curStepIdx] || steps[0];

    const selectedNode =
      methodSpec.nodes.find((n) => n.id === state.selectedNodeId) || methodSpec.nodes[0];
    const selectedCode = DATA.snippets[selectedNode.code_ref] || DATA.snippets["mt_train_step"];
    const secondaryCode = selectedNode.secondary_code_ref ? DATA.snippets[selectedNode.secondary_code_ref] : null;
    const nodeTensors = hierarchy.nodeShapes[selectedNode.shape_key] || [];

    // Build SVG Edges + Animated Particles
    const nodesById = {};
    methodSpec.nodes.forEach((n) => {
      nodesById[n.id] = n;
    });

    const edgeColors = {
      data: { stroke: "#38bdf8", label: "Data Tensor Flow" },
      forward_grad: { stroke: "#3b82f6", label: "Forward (requires_grad=True)" },
      forward_nograd: { stroke: "#a855f7", label: "Forward (torch.no_grad / detached)" },
      label: { stroke: "#10b981", label: "Ground Truth / Target Labels" },
      loss_grad: { stroke: "#f59e0b", label: "Scalar Loss Contribution" },
      backward_grad: { stroke: "#ef4444", label: "Backward Gradient ∇_θ L" },
      ema_transfer: { stroke: "#c084fc", label: "EMA Weight / Feedback Update" },
      validation: { stroke: "#14b8a6", label: "Periodic Sliding-Window Eval" },
    };

    const svgEdges = methodSpec.edges
      .map((edge) => {
        const src = nodesById[edge.from];
        const dst = nodesById[edge.to];
        if (!src || !dst) return "";

        const x1 = src.x + src.w / 2;
        const y1 = src.y + src.h / 2;
        const x2 = dst.x + dst.w / 2;
        const y2 = dst.y + dst.h / 2;

        // Compute smooth cubic bezier curve
        const dx = x2 - x1;
        const dy = y2 - y1;
        const cx1 = x1 + dx * 0.45;
        const cy1 = y1 + (Math.abs(dx) < 60 ? dy * 0.3 : 0);
        const cx2 = x2 - dx * 0.45;
        const cy2 = y2 - (Math.abs(dx) < 60 ? dy * 0.3 : 0);
        const pathD = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;

        const isActive = curStepObj.active_edges.includes(edge.id);
        const colorSpec = edgeColors[edge.kind] || edgeColors.data;
        const isDashed = edge.kind === "forward_nograd" || edge.kind === "ema_transfer" || edge.kind === "validation" || edge.kind === "backward_grad";
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2 - 8;

        return `
          <g class="transition-opacity duration-300" opacity="${isActive ? "1.0" : "0.42"}">
            <path
              d="${pathD}"
              fill="none"
              stroke="${colorSpec.stroke}"
              stroke-width="${isActive ? "3.2" : "1.8"}"
              ${isDashed ? 'stroke-dasharray="6,4"' : ""}
            />
            ${
              isActive
                ? `<circle r="5" fill="${colorSpec.stroke}">
                     <animateMotion dur="1.4s" repeatCount="indefinite" path="${pathD}" />
                   </circle>`
                : ""
            }
            <rect x="${midX - 74}" y="${midY - 11}" width="148" height="19" rx="4" fill="#0f172a" fill-opacity="0.88" stroke="${colorSpec.stroke}" stroke-opacity="0.45" stroke-width="0.8" />
            <text x="${midX}" y="${midY + 3}" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="600">${escapeHtml(edge.label)}</text>
          </g>
        `;
      })
      .join("");

    const categoryBadges = {
      data: { border: "border-sky-500/60", bg: "bg-sky-500/10", tag: "DATA STREAM", tagColor: "text-sky-400" },
      transform: { border: "border-cyan-500/60", bg: "bg-cyan-500/10", tag: "PERTURB / AUG", tagColor: "text-cyan-400" },
      model_trainable: { border: "border-blue-500/80", bg: "bg-blue-500/15", tag: "STUDENT (GRAD ON)", tagColor: "text-blue-400" },
      model_ema: { border: "border-purple-500/80", bg: "bg-purple-500/15", tag: "EMA TEACHER (DETACHED)", tagColor: "text-purple-400" },
      loss_sup: { border: "border-emerald-500/70", bg: "bg-emerald-500/10", tag: "SUPERVISED LOSS", tagColor: "text-emerald-400" },
      loss_cons: { border: "border-amber-500/70", bg: "bg-amber-500/10", tag: "CONSISTENCY LOSS", tagColor: "text-amber-400" },
      optimizer: { border: "border-rose-500/70", bg: "bg-rose-500/10", tag: "AMP & OPTIMIZER", tagColor: "text-rose-400" },
      validation: { border: "border-teal-500/70", bg: "bg-teal-500/10", tag: "3D SLIDING VAL", tagColor: "text-teal-400" },
    };

    const htmlNodes = methodSpec.nodes
      .map((node) => {
        const isStepActive = curStepObj.active_nodes.includes(node.id);
        const isSelected = state.selectedNodeId === node.id;
        const cat = categoryBadges[node.category] || categoryBadges.data;
        const shapes = hierarchy.nodeShapes[node.shape_key] || [];
        const primaryShape = shapes[0] ? formatShape(shapes[0].shape) : "";

        return `
          <div
            data-node-id="${node.id}"
            style="left: ${node.x}px; top: ${node.y}px; width: ${node.w}px; min-height: ${node.h}px;"
            class="absolute rounded-xl p-2.5 cursor-pointer transition-all duration-200 select-none bg-[var(--card)] border-2 ${
              isSelected
                ? "border-blue-400 ring-2 ring-blue-500/40 shadow-lg z-20"
                : isStepActive
                ? `${cat.border} ring-2 ring-amber-400/30 shadow-md z-10`
                : "border-[var(--border)] opacity-80 hover:opacity-100"
            }"
          >
            <div class="flex items-center justify-between gap-1 mb-1">
              <span class="text-xs font-bold tracking-tight ${cat.tagColor}">${cat.tag}</span>
              <span class="px-1.5 py-0.2 rounded text-xs font-mono ${
                node.grad_state === "requires_grad"
                  ? "bg-blue-500/20 text-blue-300"
                  : node.grad_state === "detached"
                  ? "bg-purple-500/20 text-purple-300"
                  : node.grad_state === "backward_sink"
                  ? "bg-rose-500/20 text-rose-300"
                  : "bg-slate-500/20 text-[var(--muted-foreground)]"
              }">
                ${node.grad_state === "requires_grad" ? "grad=True" : node.grad_state === "detached" ? "no_grad/detach" : node.grad_state === "backward_sink" ? "backward()" : "no_grad"}
              </span>
            </div>
            <div class="text-xs font-bold text-[var(--foreground)] leading-snug">${escapeHtml(node.title)}</div>
            <div class="text-xs text-[var(--muted-foreground)] leading-tight mt-0.5">${escapeHtml(node.subtitle)}</div>
            ${
              primaryShape
                ? `<div class="mt-1.5 pt-1 border-t border-[var(--border)] flex items-center justify-between text-xs font-mono">
                     <span class="text-[var(--muted-foreground)]">Tensor:</span>
                     <span class="text-emerald-400 font-semibold">${escapeHtml(primaryShape)}</span>
                   </div>`
                : ""
            }
          </div>
        `;
      })
      .join("");

    return `
      <div class="grid grid-cols-1 xl:grid-cols-12 gap-4">
        <!-- Left 8 Cols: Step-by-Step Controller + Interactive Pipeline Canvas -->
        <div class="xl:col-span-8 flex flex-col gap-3">
          <!-- Step-by-Step Execution Bar -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5 shadow-sm">
            <div class="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-[var(--border)]">
              <div class="flex items-center gap-2">
                <button id="btn-step-prev" class="px-2.5 py-1 rounded-lg bg-[var(--background)] border border-[var(--border)] text-xs font-semibold hover:border-blue-400">
                  ◀ Prev Step
                </button>
                <button id="btn-step-play" class="px-3 py-1 rounded-lg ${
                  state.isPlaying ? "bg-amber-600 text-white" : "bg-blue-600 text-white"
                } text-xs font-semibold shadow-sm">
                  ${state.isPlaying ? "⏸ Pause Animation" : "▶ Auto-Step Lifecycle"}
                </button>
                <button id="btn-step-next" class="px-2.5 py-1 rounded-lg bg-[var(--background)] border border-[var(--border)] text-xs font-semibold hover:border-blue-400">
                  Next Step ▶
                </button>
                <span class="text-xs font-bold text-[var(--foreground)] ml-1">
                  Step ${curStepObj.step} / ${steps.length}: <span class="text-blue-400">${escapeHtml(curStepObj.title)}</span>
                </span>
              </div>
              <span class="text-xs font-mono px-2 py-0.5 rounded bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)]">
                Source: ${escapeHtml(curStepObj.code_line)}
              </span>
            </div>

            <!-- Step Pills -->
            <div class="flex flex-wrap gap-1.5 my-2.5">
              ${steps
                .map(
                  (s) => `
                  <button
                    data-step-num="${s.step}"
                    class="px-2.5 py-1 rounded-md text-xs font-semibold transition ${
                      s.step === curStepObj.step
                        ? "bg-blue-600 text-white"
                        : "bg-[var(--background)] text-[var(--muted-foreground)] border border-[var(--border)] hover:text-[var(--foreground)]"
                    }"
                  >
                    S${s.step}: ${escapeHtml(s.title.split(" ")[0])}
                  </button>
                `
                )
                .join("")}
            </div>

            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-3 text-xs leading-relaxed">
              <div class="text-[var(--foreground)] font-medium">${escapeHtml(curStepObj.what_happens)}</div>
              <div class="mt-2 pt-2 border-t border-[var(--border)] flex flex-wrap items-center justify-between gap-2 font-mono text-amber-300">
                <span>Mathematical Operation: <code>${escapeHtml(curStepObj.key_formula)}</code></span>
              </div>
            </div>
          </div>

          <!-- Interactive Graph Canvas -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3 shadow-sm overflow-x-auto">
            <div class="flex items-center justify-between mb-2 text-xs text-[var(--muted-foreground)]">
              <span>💡 <strong>Click vào bất kỳ khối thành phần (Node) nào</strong> trên sơ đồ để xem Tensor Shapes, Công thức & Source Code thực tế ở bảng bên phải.</span>
              <div class="flex items-center gap-3">
                <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block"></span> Forward (Grad)</span>
                <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-purple-500 inline-block"></span> No-Grad / EMA</span>
                <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block"></span> Backward ∇_θ</span>
              </div>
            </div>

            <div class="relative bg-[var(--background)] border border-[var(--border)] rounded-xl overflow-hidden" style="width: 1170px; height: 465px;">
              <svg class="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1170 465">
                ${svgEdges}
              </svg>
              ${htmlNodes}
            </div>
          </div>
        </div>

        <!-- Right 4 Cols: Selected Component Deep Inspector & Live Source Code -->
        <div class="xl:col-span-4 flex flex-col gap-3">
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm flex flex-col gap-3">
            <div class="border-b border-[var(--border)] pb-2.5">
              <div class="text-xs font-bold uppercase tracking-wider text-blue-400">Selected Component Inspector</div>
              <h3 class="text-base font-bold text-[var(--foreground)] mt-0.5">${escapeHtml(selectedNode.title)}</h3>
              <p class="text-xs text-[var(--muted-foreground)] mt-1 leading-relaxed">${escapeHtml(selectedNode.summary)}</p>
            </div>

            <!-- Live Tensor Shapes at this Node -->
            <div>
              <div class="text-xs font-bold text-[var(--foreground)] mb-1.5 flex items-center justify-between">
                <span>Live Tensor Shapes (Symbolic Propagator)</span>
                <span class="text-xs font-mono text-[var(--muted-foreground)]">world_size=${state.world_size}</span>
              </div>
              <div class="space-y-1.5">
                ${nodeTensors
                  .map(
                    (t) => `
                    <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2 text-xs">
                      <div class="flex items-center justify-between font-mono">
                        <span class="font-semibold text-[var(--foreground)]">${escapeHtml(t.name)}</span>
                        <span class="text-emerald-400 font-bold">${escapeHtml(formatShape(t.shape))}</span>
                      </div>
                      <div class="flex items-center justify-between text-xs text-[var(--muted-foreground)] mt-0.5">
                        <span>dtype: <code>${escapeHtml(t.dtype)}</code></span>
                        <span>${escapeHtml(t.note)}</span>
                      </div>
                    </div>
                  `
                  )
                  .join("")}
              </div>
            </div>

            <!-- Key Implementation Details -->
            <div>
              <div class="text-xs font-bold text-[var(--foreground)] mb-1.5">Implementation Mechanisms & Nuances</div>
              <ul class="space-y-1.5 text-xs text-[var(--muted-foreground)] list-disc pl-4 leading-relaxed">
                ${(selectedNode.details || []).map((d) => `<li>${escapeHtml(d)}</li>`).join("")}
              </ul>
            </div>

            <!-- Verified Repository Source Code -->
            <div>
              <div class="flex items-center justify-between text-xs font-bold text-[var(--foreground)] mb-1">
                <span>Verified Source Code</span>
                <code class="text-xs text-blue-400">${escapeHtml(selectedCode.title)}</code>
              </div>
              <pre class="bg-slate-950 text-slate-200 border border-[var(--border)] rounded-lg p-2.5 text-xs font-mono overflow-x-auto max-h-64 leading-relaxed">${escapeHtml(
                selectedCode.code
              )}</pre>
            </div>

            ${
              secondaryCode
                ? `<div>
                     <div class="flex items-center justify-between text-xs font-bold text-[var(--foreground)] mb-1">
                       <span>Related Module</span>
                       <code class="text-xs text-purple-400">${escapeHtml(secondaryCode.title)}</code>
                     </div>
                     <pre class="bg-slate-950 text-slate-200 border border-[var(--border)] rounded-lg p-2.5 text-xs font-mono overflow-x-auto max-h-48 leading-relaxed">${escapeHtml(
                       secondaryCode.code
                     )}</pre>
                   </div>`
                : ""
            }
          </div>
        </div>
      </div>
    `;
  }

  // ============================================================================
  // VIEW 2: 3D VOLUMETRIC TENSOR & ANISOTROPIC ARCHITECTURE INSPECTOR
  // ============================================================================
  function renderTensorInspectorView(hierarchy) {
    const enc = hierarchy.encoderStages;
    const dec = hierarchy.decoderStages;
    const layout = hierarchy.layout;

    return `
      <div class="space-y-4">
        <!-- Interactive Architecture & Tensor Controls -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
          <div class="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-[var(--border)]">
            <div>
              <h2 class="text-base font-bold text-[var(--foreground)]">
                Symbolic 3D Volumetric Tensor & Anisotropic Architecture Explorer
              </h2>
              <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
                Thay đổi cấu hình Backbone, Patch Size 3D (Z, Y, X), hoặc Batch Size để quan sát cách kích thước tensor và độ phân giải biến đổi qua từng tầng Encoder-Decoder.
              </p>
            </div>

            <div class="flex flex-wrap items-center gap-3 text-xs">
              <div>
                <label class="block text-[var(--muted-foreground)] mb-0.5">Backbone Model</label>
                <select id="ctrl-backbone" class="bg-[var(--background)] border border-[var(--border)] rounded px-2 py-1 font-semibold">
                  <option value="nnunet3d" ${state.backbone === "nnunet3d" ? "selected" : ""}>nnUNet3D (InstanceNorm + LeakyReLU)</option>
                  <option value="vnet" ${state.backbone === "vnet" ? "selected" : ""}>VNet (BatchNorm + Residual)</option>
                  <option value="unet3d" ${state.backbone === "unet3d" ? "selected" : ""}>UNet3D (Standard Isotropic)</option>
                </select>
              </div>

              <div>
                <label class="block text-[var(--muted-foreground)] mb-0.5">Stages (nnUNet3D)</label>
                <select id="ctrl-stages" class="bg-[var(--background)] border border-[var(--border)] rounded px-2 py-1 font-semibold" ${state.backbone !== "nnunet3d" ? "disabled" : ""}>
                  <option value="5" ${Number(state.num_stages) === 5 ? "selected" : ""}>5 Stages</option>
                  <option value="6" ${Number(state.num_stages) === 6 ? "selected" : ""}>6 Stages (models/nnunet3d.yaml)</option>
                  <option value="7" ${Number(state.num_stages) === 7 ? "selected" : ""}>7 Stages (mean_teacher.yaml)</option>
                </select>
              </div>

              <div>
                <label class="block text-[var(--muted-foreground)] mb-0.5">Anisotropic Kernels (1,3,3)</label>
                <select id="ctrl-aniso" class="bg-[var(--background)] border border-[var(--border)] rounded px-2 py-1 font-semibold" ${state.backbone !== "nnunet3d" ? "disabled" : ""}>
                  <option value="true" ${state.anisotropic_kernels ? "selected" : ""}>True (Z downsample ${hierarchy.zDownsampleTotal}×, XY ${hierarchy.xyDownsampleTotal}×)</option>
                  <option value="false" ${!state.anisotropic_kernels ? "selected" : ""}>False (Isotropic 3×3×3)</option>
                </select>
              </div>

              <div>
                <label class="block text-[var(--muted-foreground)] mb-0.5">3D Patch Size [Z, Y, X]</label>
                <select id="ctrl-patch" class="bg-[var(--background)] border border-[var(--border)] rounded px-2 py-1 font-semibold">
                  <option value="28,256,256" ${state.patch_z === 28 && state.patch_y === 256 ? "selected" : ""}>[28, 256, 256] (nnUNet Option 1 CTClip)</option>
                  <option value="32,192,192" ${state.patch_z === 32 && state.patch_y === 192 ? "selected" : ""}>[32, 192, 192] (bhsd_mt_nnunet_fold0.yaml)</option>
                  <option value="32,160,160" ${state.patch_z === 32 && state.patch_y === 160 ? "selected" : ""}>[32, 160, 160] (configs/base.yaml)</option>
                </select>
              </div>

              <div>
                <label class="block text-[var(--muted-foreground)] mb-0.5">Global Batch / GPUs</label>
                <select id="ctrl-batch" class="bg-[var(--background)] border border-[var(--border)] rounded px-2 py-1 font-semibold">
                  <option value="8,4,2" ${state.batch_size === 8 && state.world_size === 2 ? "selected" : ""}>Global 8 (4L+4U) on 2 GPUs -> 2L+2U/GPU</option>
                  <option value="4,2,2" ${state.batch_size === 4 && state.world_size === 2 ? "selected" : ""}>Global 4 (2L+2U) on 2 GPUs -> 1L+1U/GPU</option>
                  <option value="4,2,1" ${state.batch_size === 4 && state.world_size === 1 ? "selected" : ""}>Global 4 (2L+2U) on 1 GPU -> 2L+2U/GPU</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Visual Encoder-Bottleneck-Decoder 3D Blocks -->
          <div class="mt-4">
            <div class="flex items-center justify-between text-xs mb-2">
              <span class="font-bold text-[var(--foreground)]">
                Anisotropic Spatial & Channel Hierarchy: Total Z Downsample = <code class="text-emerald-400 font-bold">${hierarchy.zDownsampleTotal}×</code> (${state.patch_z} → ${enc[enc.length - 1].spatial[0]}), Total XY Downsample = <code class="text-blue-400 font-bold">${hierarchy.xyDownsampleTotal}×</code> (${state.patch_y} → ${enc[enc.length - 1].spatial[1]})
              </span>
              <span class="text-[var(--muted-foreground)]">${escapeHtml(hierarchy.backboneSpec.norm)} • ${escapeHtml(hierarchy.backboneSpec.activation)}</span>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5 mt-2">
              ${enc
                .map((st, idx) => {
                  const boxH = Math.max(26, Math.round(78 * Math.pow(st.spatial[1] / state.patch_y, 0.35)));
                  const boxW = Math.max(26, Math.round(78 * Math.pow(st.spatial[2] / state.patch_x, 0.35)));
                  const depthZ = Math.max(10, Math.round(28 * (st.spatial[0] / state.patch_z)));
                  return `
                    <div class="bg-[var(--background)] border ${
                      idx === enc.length - 1 ? "border-amber-500/70" : st.is_z_preserved ? "border-emerald-500/50" : "border-[var(--border)]"
                    } rounded-xl p-2.5 flex flex-col justify-between">
                      <div>
                        <div class="flex items-center justify-between text-xs">
                          <span class="font-bold text-[var(--foreground)]">${escapeHtml(st.name)}</span>
                          <span class="px-1.5 py-0.2 rounded text-xs font-mono ${
                            st.is_anisotropic_kernel ? "bg-emerald-500/20 text-emerald-300" : "bg-blue-500/20 text-blue-300"
                          }">k=${st.kernel.join("×")}</span>
                        </div>
                        <div class="text-xs text-[var(--muted-foreground)] mt-0.5">${escapeHtml(st.pool_name)}</div>
                      </div>

                      <!-- Isometric 3D Tensor Cube Illustration -->
                      <div class="my-3 flex items-center justify-center" style="height: 92px;">
                        <div class="relative flex items-center justify-center" style="width: ${boxW}px; height: ${boxH}px; background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(147,51,234,0.35)); border: 1.5px solid #60a5fa; border-radius: 4px; box-shadow: ${depthZ / 2}px -${depthZ / 2}px 0px rgba(30,64,175,0.45);">
                          <span class="text-xs font-mono font-bold text-white drop-shadow">${st.out_channels}c</span>
                        </div>
                      </div>

                      <div class="border-t border-[var(--border)] pt-1.5 text-xs font-mono">
                        <div class="flex justify-between">
                          <span class="text-[var(--muted-foreground)]">(Z,Y,X):</span>
                          <span class="text-emerald-400 font-bold">${st.spatial.join("×")}</span>
                        </div>
                        <div class="flex justify-between mt-0.5">
                          <span class="text-[var(--muted-foreground)]">Tensor:</span>
                          <span class="text-[var(--foreground)]">[${st.tensor_shape.join(",")}]</span>
                        </div>
                        ${
                          st.is_z_preserved
                            ? `<div class="mt-1 text-xs text-emerald-400 font-sans font-medium">✓ Z preserved (stride Z=1)</div>`
                            : idx > 0
                            ? `<div class="mt-1 text-xs text-amber-400 font-sans font-medium">↓ Z halved (stride Z=2)</div>`
                            : `<div class="mt-1 text-xs text-sky-400 font-sans font-medium">Full input resolution</div>`
                        }
                      </div>
                    </div>
                  `;
                })
                .join("")}
            </div>
          </div>
        </div>

        <!-- Bottom Grid: Two-Stream Batch Slicing + Deep Supervision Multi-Scale Tree -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <!-- Card 1: Two-Stream Batch & Anti-Leakage Crop -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
            <h3 class="text-sm font-bold text-[var(--foreground)]">
              Two-Stream Batch Partitioning & Anti-Leakage Crop (<code>data/dataset_bhsd.py:218</code>)
            </h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-1 leading-relaxed">
              Mỗi batch cục bộ trên 1 GPU có kích thước <code>B_local = ${layout.local_batch_size}</code>. Hãy quan sát sự khác biệt giữa nửa đầu <code>[:labeled_bs]</code> và nửa sau <code>[labeled_bs:]</code>:
            </p>

            <div class="mt-3 space-y-2">
              ${Array.from({ length: layout.local_batch_size })
                .map((_, idx) => {
                  const isLab = idx < layout.local_labeled_bs;
                  return `
                    <div class="p-2.5 rounded-lg border ${
                      isLab ? "border-emerald-500/50 bg-emerald-500/10" : "border-amber-500/50 bg-amber-500/10"
                    } flex flex-wrap items-center justify-between gap-2 text-xs">
                      <div class="flex items-center gap-2">
                        <span class="px-2 py-0.5 rounded font-mono font-bold ${
                          isLab ? "bg-emerald-600 text-white" : "bg-amber-600 text-white"
                        }">Sample [${idx}]</span>
                        <span class="font-semibold text-[var(--foreground)]">
                          ${isLab ? "Labeled Stream (primary_indices)" : "Unlabeled Stream (secondary_indices)"}
                        </span>
                      </div>
                      <div class="flex flex-wrap items-center gap-2 font-mono text-xs">
                        <span class="px-2 py-0.5 rounded bg-[var(--background)] border border-[var(--border)]">
                          ${isLab ? "Crop: 33% FG-Oversampled" : "label=0 -> 100% Uniform Crop"}
                        </span>
                        <span class="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300">
                          ${isLab ? "✓ Supervised + ✓ Consistency" : "✗ Supervised | ✓ Consistency"}
                        </span>
                      </div>
                    </div>
                  `;
                })
                .join("")}
            </div>
          </div>

          <!-- Card 2: Decoder Skip Concatenations & Deep Supervision Heads -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
            <h3 class="text-sm font-bold text-[var(--foreground)]">
              Decoder Skip Concatenation & Multi-Scale Deep Supervision (<code>losses/deep_supervision.py</code>)
            </h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-1 leading-relaxed">
              Các đầu ra phụ từ <code>d2..d${hierarchy.numStages - 1}</code> được phóng đại bằng <code>F.interpolate(..., mode='trilinear', align_corners=False)</code> về <code>(${state.patch_z}, ${state.patch_y}, ${state.patch_x})</code> trước khi tính Supervised Loss:
            </p>

            <div class="mt-3 overflow-x-auto">
              <table class="w-full text-left border-collapse text-xs">
                <thead>
                  <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                    <th class="py-1.5 pr-2">Decoder</th>
                    <th class="py-1.5 px-2">Skip Concat</th>
                    <th class="py-1.5 px-2">Head Logits Shape</th>
                    <th class="py-1.5 px-2">Upsampled Target</th>
                    <th class="py-1.5 pl-2 text-right">Normalized w̃_k</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-[var(--border)] font-mono">
                  ${dec
                    .slice()
                    .reverse()
                    .map((d, idx) => {
                      const w = hierarchy.dsWeights[idx] || 0;
                      return `
                        <tr>
                          <td class="py-1.5 pr-2 font-bold text-[var(--foreground)]">${escapeHtml(d.name)} (${escapeHtml(d.head_name || "none")})</td>
                          <td class="py-1.5 px-2 text-[var(--muted-foreground)]">${d.up_in_channels}→${d.skip_channels} + ${d.skip_channels} = <strong>${d.concat_channels}c</strong></td>
                          <td class="py-1.5 px-2 text-emerald-400">${d.head_shape ? `[${d.head_shape.join(",")}]` : "—"}</td>
                          <td class="py-1.5 px-2 text-blue-400">${
                            idx === 0
                              ? "Native (1×)"
                              : `Trilinear → [${layout.local_labeled_bs},${state.num_classes},${state.patch_z},${state.patch_y},${state.patch_x}]`
                          }</td>
                          <td class="py-1.5 pl-2 text-right font-bold text-amber-400">${w ? w.toFixed(4) : "—"}</td>
                        </tr>
                      `;
                    })
                    .join("")}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // ============================================================================
  // VIEW 3: MATHEMATICAL, LOSS & BHSD METRIC PARADOX SANDBOX
  // ============================================================================
  function renderMathSandboxView(hierarchy, stepMetrics) {
    const sim = Core.simulateConsistencyLoss(state, {
      pred_discrepancy: state.sim_pred_discrepancy,
      fg_overlap: state.sim_fg_overlap,
    });
    const paradox = Core.simulateMetricParadox({
      pos_dice_level: state.paradox_pos_dice_level,
      fp_case_rate: state.paradox_fp_case_rate,
    });

    const formulas = [
      {
        id: "hybrid_cons",
        name: "1. Hybrid Consistency Loss (losses/consistency.py:90-107)",
        equation: "L_cons = 0.5 · L_SoftDice(P_S[:, 1:], P_T[:, 1:]) + 0.5 · [ ∑_{c=0}^5 w_c · MSE(P_S[:, c], P_T[:, c]) / ∑_{c=0}^5 w_c ]",
        code_ref: "consistency_loss",
        terms: [
          { sym: "P_S[:, 1:], P_T[:, 1:]", desc: "Xác suất Softmax FP32 của 5 lớp xuất huyết (1..5), loại bỏ hoàn toàn kênh 0 (Background) trong nhánh Soft Dice để chống confirmation bias." },
          { sym: "w_c = [0.05, 1, 1, 1, 1, 1]", desc: "Trọng số cân bằng kênh MSE (bg_consistency_weight = 0.05): giảm 20 lần áp lực từ Background (vốn chiếm >97% thể tích hộp sọ)." },
          { sym: "2·∑(P_S·P_T) / ∑(P_S² + P_T²)", desc: "Continuous Probability-to-Probability Soft Dice với mẫu số bậc hai L2 (compute_soft_dice_consistency)." },
        ],
      },
      {
        id: "sup_multiscale",
        name: "2. Multi-Scale Supervised Loss (methods/mean_teacher.py:138-158)",
        equation: "L_sup = ∑_{k=0}^5 w̃_k · 0.5 · [ CrossEntropy(↑Z_S^(k)[:B_L], Y[:B_L]) + MaskDiceLoss_{c≥1}(↑Z_S^(k)[:B_L], Y[:B_L]) ]",
        code_ref: "mt_train_step",
        terms: [
          { sym: "[:B_L] (self.labeled_bs)", desc: "Chỉ lấy lát cắt mẫu có nhãn thật ở nửa đầu batch cục bộ." },
          { sym: "↑Z_S^(k)", desc: "Phép nội suy F.interpolate(aux_lab, size=target_shape, mode='trilinear', align_corners=False) đưa 5 đầu ra phụ về độ phân giải gốc." },
          { sym: "w̃_k = 2⁻ᵏ / ∑ 2⁻ʲ", desc: "Trọng số giảm theo hàm mũ chuẩn nnU-Net: [0.5079, 0.2540, 0.1270, 0.0635, 0.0317, 0.0159]." },
        ],
      },
      {
        id: "ema_cosine",
        name: "3. Cosine EMA Teacher Update (methods/ssl_utils.py:106-120)",
        equation: "θ'_t ← α_t · θ'_{t-1} + (1 - α_t) · θ_t,   where   α_t = 0.99 + (α_max - 0.99) · 0.5 · (1 - cos(π · t / T_max))",
        code_ref: "ema_update_fn",
        terms: [
          { sym: "α_min = 0.99", desc: "Ở đầu quá trình train (t=0), Teacher nhận 1% trọng số mới từ Student mỗi bước (cửa sổ nhớ ~100 bước) để thoát nhanh khỏi khởi tạo ngẫu nhiên." },
          { sym: "α_max = 0.999", desc: "Ở cuối quá trình train (t=T_max), Teacher nhận 0.1% trọng số từ Student (cửa sổ nhớ ~1000 bước), tạo nhãn giả cực kỳ mượt và ổn định." },
          { sym: "ema_buffer.copy_(buffer)", desc: "Các buffer (như BatchNorm running_mean/var) được copy trực tiếp từ Student vì Teacher luôn ở chế độ eval()." },
        ],
      },
    ];

    const activeForm = formulas.find((f) => f.id === state.activeFormulaId) || formulas[0];
    const formCode = DATA.snippets[activeForm.code_ref];

    return `
      <div class="space-y-4">
        <!-- Section 1: Interactive Formula-to-Code Mapper -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
          <div class="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[var(--border)]">
            <h2 class="text-base font-bold text-[var(--foreground)]">Interactive Formula-to-Code & Mechanism Mapper</h2>
            <div class="flex flex-wrap gap-1.5">
              ${formulas
                .map(
                  (f) => `
                  <button
                    data-formula-id="${f.id}"
                    class="px-3 py-1 rounded-lg text-xs font-semibold ${
                      f.id === activeForm.id
                        ? "bg-blue-600 text-white"
                        : "bg-[var(--background)] text-[var(--muted-foreground)] border border-[var(--border)]"
                    }"
                  >
                    ${escapeHtml(f.name.split(" (")[0])}
                  </button>
                `
                )
                .join("")}
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 mt-3">
            <div class="lg:col-span-7 flex flex-col justify-between gap-3">
              <div class="bg-[var(--background)] border border-blue-500/40 rounded-xl p-3.5 font-mono text-sm text-amber-300 leading-relaxed">
                ${escapeHtml(activeForm.equation)}
              </div>

              <div class="space-y-2">
                ${activeForm.terms
                  .map(
                    (t) => `
                    <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5 text-xs">
                      <span class="font-mono font-bold text-blue-400">${escapeHtml(t.sym)}</span>
                      <p class="text-[var(--muted-foreground)] mt-1 leading-relaxed">${escapeHtml(t.desc)}</p>
                    </div>
                  `
                  )
                  .join("")}
              </div>
            </div>

            <div class="lg:col-span-5">
              <div class="text-xs font-bold text-[var(--foreground)] mb-1 flex justify-between">
                <span>Exact PyTorch Implementation</span>
                <code class="text-blue-400">${escapeHtml(formCode.title)}</code>
              </div>
              <pre class="bg-slate-950 text-slate-200 border border-[var(--border)] rounded-lg p-2.5 text-xs font-mono overflow-x-auto max-h-64 leading-relaxed">${escapeHtml(
                formCode.code
              )}</pre>
            </div>
          </div>
        </div>

        <!-- Section 2: Voxel-Level Consistency & Class-Balance Simulator + BHSD Metric Paradox Lab -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <!-- Left: Consistency Loss & Class-Balance Simulator -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm flex flex-col gap-3">
            <div>
              <h3 class="text-sm font-bold text-[var(--foreground)]">
                Interactive Consistency Loss & Class-Imbalance Sandbox (<code>losses/consistency.py</code>)
              </h3>
              <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
                Thử thay đổi <code>consistency_type</code>, <code>bg_consistency_weight</code> và <code>confidence_thresh</code> để thấy tại sao MSE thông thường thất bại trên CT xuất huyết não.
              </p>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-3 gap-2.5 bg-[var(--background)] border border-[var(--border)] rounded-lg p-3 text-xs">
              <div>
                <label class="block text-[var(--muted-foreground)] mb-1">consistency_type</label>
                <select id="sim-cons-type" class="w-full bg-[var(--card)] border border-[var(--border)] rounded px-2 py-1 font-semibold">
                  <option value="hybrid" ${state.consistency_type === "hybrid" ? "selected" : ""}>hybrid (0.5 Dice + 0.5 MSE)</option>
                  <option value="dice" ${state.consistency_type === "dice" ? "selected" : ""}>dice (Foreground Soft Dice)</option>
                  <option value="weighted_mse" ${state.consistency_type === "weighted_mse" ? "selected" : ""}>weighted_mse (Class-Balanced)</option>
                  <option value="kl" ${state.consistency_type === "kl" ? "selected" : ""}>kl (KL Divergence)</option>
                  <option value="raw_mse" ${state.consistency_type === "raw_mse" ? "selected" : ""}>raw_mse (Unweighted)</option>
                </select>
              </div>

              <div>
                <label class="block text-[var(--muted-foreground)] mb-1">
                  bg_weight: <strong>${state.bg_consistency_weight}</strong>
                </label>
                <input id="sim-bg-weight" type="range" min="0.0" max="1.0" step="0.05" value="${state.bg_consistency_weight}" class="w-full accent-blue-500" />
              </div>

              <div>
                <label class="block text-[var(--muted-foreground)] mb-1">
                  confidence_thresh: <strong>${state.confidence_thresh}</strong>
                </label>
                <input id="sim-conf-thresh" type="range" min="0.0" max="0.95" step="0.05" value="${state.confidence_thresh}" class="w-full accent-purple-500" />
              </div>
            </div>

            <!-- Output Loss Breakdown -->
            <div class="grid grid-cols-3 gap-2 text-xs">
              <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5">
                <div class="text-[var(--muted-foreground)]">FG Soft Dice Loss</div>
                <div class="text-sm font-bold text-emerald-400 mt-0.5">${sim.softDiceLoss.toFixed(4)}</div>
                <div class="text-xs text-[var(--muted-foreground)]">Excludes Class 0</div>
              </div>
              <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5">
                <div class="text-[var(--muted-foreground)]">Class-Balanced MSE</div>
                <div class="text-sm font-bold text-blue-400 mt-0.5">${sim.weightedMse.toFixed(4)}</div>
                <div class="text-xs text-[var(--muted-foreground)]">BG w=${state.bg_consistency_weight}</div>
              </div>
              <div class="bg-[var(--background)] border border-amber-500/50 rounded-lg p-2.5">
                <div class="text-[var(--muted-foreground)]">Active L_consistency</div>
                <div class="text-sm font-bold text-amber-400 mt-0.5">${sim.activeConsLoss.toFixed(4)}</div>
                <div class="text-xs text-[var(--muted-foreground)]">Weighted: ${(sim.activeConsLoss * stepMetrics.consW).toFixed(5)}</div>
              </div>
            </div>

            <!-- Per-Class Gradient Share Bars -->
            <div class="space-y-1.5 text-xs">
              <div class="font-semibold text-[var(--foreground)]">Tỷ trọng đóng góp vào Class-Balanced MSE theo từng lớp:</div>
              ${sim.classes
                .map(
                  (c) => `
                  <div>
                    <div class="flex justify-between text-xs mb-0.5">
                      <span>${escapeHtml(c.name)} <span class="text-[var(--muted-foreground)]">(w=${c.weight})</span></span>
                      <span class="font-mono font-semibold">${c.weighted_mse_share_pct.toFixed(1)}% of MSE</span>
                    </div>
                    <div class="w-full h-2 bg-[var(--background)] rounded-full overflow-hidden">
                      <div class="h-full ${c.idx === 0 ? "bg-slate-400" : "bg-blue-500"}" style="width: ${Math.min(100, c.weighted_mse_share_pct)}%;"></div>
                    </div>
                  </div>
                `
                )
                .join("")}
            </div>

            <!-- Confidence Threshold Dual-Condition Box -->
            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5 text-xs">
              <div class="font-semibold text-[var(--foreground)]">
                CT Foreground-Preserving Confidence Filter (<code>tau = ${state.confidence_thresh}</code>):
              </div>
              <div class="text-[var(--muted-foreground)] mt-0.5">
                Điều kiện kép <code>(fg_max >= min(tau, 0.4) = ${sim.fgEffThresh.toFixed(2)}) | (bg >= ${state.confidence_thresh})</code> giữ lại <strong>${sim.fgRetainedPct}%</strong> voxel tổn thương (so với chỉ <strong>${sim.naiveFgRetainedPct}%</strong> nếu dùng ngưỡng đơn <code>max_prob >= tau</code>).
              </div>
            </div>
          </div>

          <!-- Right: BHSD 38-Case Fold 0 Metric Paradox Simulator -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm flex flex-col gap-3">
            <div>
              <h3 class="text-sm font-bold text-[var(--foreground)]">
                BHSD Validation Metric Paradox Lab (38 Cases Fold 0: <code>Legacy mDice</code> vs <code>Positive Macro Dice</code>)
              </h3>
              <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
                Giải thích tại sao dự đoán <strong>toàn bộ Background (Recall = 0%)</strong> lại đạt <code>Legacy mDice = 0.5579</code>, khiến checkpoint <code>best_model.pth</code> ở iter 1,000–2,750 bị lệch khỏi mô hình phân đoạn tốt nhất!
              </p>
            </div>

            <div class="grid grid-cols-2 gap-3 bg-[var(--background)] border border-[var(--border)] rounded-lg p-3 text-xs">
              <div>
                <label class="block text-[var(--muted-foreground)] mb-1">
                  True Lesion Segmentation Quality: <strong>${Math.round(state.paradox_pos_dice_level * 100)}%</strong> (0% = Empty Model)
                </label>
                <input id="sim-paradox-pos" type="range" min="0.0" max="1.4" step="0.05" value="${state.paradox_pos_dice_level}" class="w-full accent-emerald-500" />
              </div>
              <div>
                <label class="block text-[var(--muted-foreground)] mb-1">
                  False Positive Rate on Negative Cases: <strong>${Math.round(state.paradox_fp_case_rate * 100)}%</strong>
                </label>
                <input id="sim-paradox-fp" type="range" min="0.0" max="1.4" step="0.05" value="${state.paradox_fp_case_rate}" class="w-full accent-rose-500" />
              </div>
            </div>

            <div class="grid grid-cols-3 gap-2 text-xs">
              <div class="bg-[var(--background)] border border-amber-500/50 rounded-lg p-2.5">
                <div class="text-[var(--muted-foreground)]">Simulated Legacy mDice</div>
                <div class="text-base font-bold text-amber-400 mt-0.5">${paradox.legacyMdice.toFixed(4)}</div>
                <div class="text-xs text-[var(--muted-foreground)]">Rewards empty neg cases!</div>
              </div>
              <div class="bg-[var(--background)] border border-emerald-500/50 rounded-lg p-2.5">
                <div class="text-[var(--muted-foreground)]">Positive Macro Dice</div>
                <div class="text-base font-bold text-emerald-400 mt-0.5">${paradox.positiveMacro.toFixed(4)}</div>
                <div class="text-xs text-[var(--muted-foreground)]">True lesion segmentation</div>
              </div>
              <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5">
                <div class="text-[var(--muted-foreground)]">All-BG Analytic Baseline</div>
                <div class="text-base font-bold text-rose-400 mt-0.5">${paradox.emptyBaselineMdice.toFixed(4)}</div>
                <div class="text-xs text-[var(--muted-foreground)]">(32+15+16+18+25)/190</div>
              </div>
            </div>

            <div class="overflow-x-auto">
              <table class="w-full text-left border-collapse text-xs">
                <thead>
                  <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                    <th class="py-1.5">Class</th>
                    <th class="py-1.5">GT Pos/Neg</th>
                    <th class="py-1.5">TN Neg Cases (Dice=1)</th>
                    <th class="py-1.5">Pos Case Dice</th>
                    <th class="py-1.5 text-right">Legacy Case-Mean</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-[var(--border)] font-mono">
                  ${paradox.results
                    .map(
                      (r) => `
                      <tr>
                        <td class="py-1.5 font-sans font-semibold text-[var(--foreground)]">${escapeHtml(r.name)}</td>
                        <td class="py-1.5 text-[var(--muted-foreground)]">${r.pos} pos / ${r.neg} neg</td>
                        <td class="py-1.5 text-amber-400">${r.tnNegCount}/${r.neg} (${r.fpNegCount} FP)</td>
                        <td class="py-1.5 text-emerald-400 font-bold">${r.simPosDice.toFixed(4)}</td>
                        <td class="py-1.5 text-right font-bold text-[var(--foreground)]">${r.legacyCaseMean.toFixed(4)}</td>
                      </tr>
                    `
                    )
                    .join("")}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // ============================================================================
  // VIEW 4: HYPERPARAMETER & SCHEDULE TIME-MACHINE
  // ============================================================================
  function renderScheduleTimeMachineView(hierarchy, stepMetrics) {
    const seriesData = Core.generateScheduleSeries(state, hierarchy.epochLen, 100);
    const pts = seriesData.points;
    const maxIter = Math.max(100, Number(state.max_iteration) || 12000);
    const cursorX = Math.max(0, Math.min(1, state.iter_num / maxIter)) * 680 + 45;

    function buildSvgPolyline(valFn, maxVal, minVal = 0) {
      const rng = Math.max(1e-9, maxVal - minVal);
      return pts
        .map((p, idx) => {
          const x = 45 + (idx / (pts.length - 1)) * 680;
          const norm = Math.max(0, Math.min(1, (valFn(p) - minVal) / rng));
          const y = 155 - norm * 130;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
    }

    const maxCons = Math.max(0.05, Number(state.consistency) || 0.05);
    const maxLr = Math.max(0.001, Number(state.base_lr) || 0.01);
    const lrPoints = buildSvgPolyline((p) => p.lr, maxLr, 0);
    const consPoints = buildSvgPolyline((p) => p.consW, maxCons, 0);
    const alphaPoints = buildSvgPolyline((p) => p.actualAlpha, 1.0, 0.985);

    return `
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <!-- Left 4 Cols: Interactive Hyperparameter Controls -->
        <div class="lg:col-span-4 bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm space-y-3">
          <h2 class="text-base font-bold text-[var(--foreground)] border-b border-[var(--border)] pb-2">
            Hyperparameter Explorer Controls
          </h2>

          <div class="space-y-2.5 text-xs">
            <div>
              <div class="flex justify-between mb-0.5">
                <span class="text-[var(--muted-foreground)]"><code>method.consistency</code> (λ_max)</span>
                <span class="font-mono font-bold text-amber-400">${state.consistency}</span>
              </div>
              <input id="hp-consistency" type="range" min="0.01" max="1.5" step="0.01" value="${state.consistency}" class="w-full accent-amber-500" />
            </div>

            <div>
              <div class="flex justify-between mb-0.5">
                <span class="text-[var(--muted-foreground)]"><code>method.consistency_rampup</code></span>
                <span class="font-mono font-bold text-amber-400">${state.consistency_rampup} (${state.consistency_rampup > 100 ? "iters" : "epochs!"})</span>
              </div>
              <input id="hp-rampup" type="range" min="0" max="10000" step="20" value="${state.consistency_rampup}" class="w-full accent-amber-500" />
            </div>

            <div>
              <div class="flex justify-between mb-0.5">
                <span class="text-[var(--muted-foreground)]"><code>consistency_rampdown_start</code></span>
                <span class="font-mono font-bold text-amber-400">${state.consistency_rampdown_start}</span>
              </div>
              <input id="hp-rampdown" type="range" min="0" max="${state.max_iteration}" step="200" value="${state.consistency_rampdown_start}" class="w-full accent-amber-500" />
            </div>

            <div>
              <div class="flex justify-between mb-0.5">
                <span class="text-[var(--muted-foreground)]"><code>consistency_min_factor</code></span>
                <span class="font-mono font-bold text-amber-400">${state.consistency_min_factor}</span>
              </div>
              <input id="hp-min-factor" type="range" min="0.0" max="1.0" step="0.05" value="${state.consistency_min_factor}" class="w-full accent-amber-500" />
            </div>

            <div class="pt-2 border-t border-[var(--border)]">
              <div class="flex justify-between mb-0.5">
                <span class="text-[var(--muted-foreground)]"><code>method.ema_decay</code> (α_target)</span>
                <span class="font-mono font-bold text-purple-400">${state.ema_decay}</span>
              </div>
              <input id="hp-ema-decay" type="range" min="0.985" max="0.9999" step="0.0005" value="${state.ema_decay}" class="w-full accent-purple-500" />
            </div>

            <div>
              <div class="flex justify-between mb-0.5">
                <span class="text-[var(--muted-foreground)]"><code>method.ema_schedule</code></span>
                <select id="hp-ema-sched" class="bg-[var(--background)] border border-[var(--border)] rounded px-2 py-0.5 font-semibold">
                  <option value="cosine" ${state.ema_schedule === "cosine" ? "selected" : ""}>cosine (0.99 → α_target)</option>
                  <option value="step" ${state.ema_schedule === "step" ? "selected" : ""}>step min(1 - 1/(t+1), α)</option>
                </select>
              </div>
            </div>

            <div class="pt-2 border-t border-[var(--border)]">
              <div class="flex justify-between mb-0.5">
                <span class="text-[var(--muted-foreground)]"><code>method.poly_power</code> (LR Decay)</span>
                <span class="font-mono font-bold text-blue-400">${state.poly_power}</span>
              </div>
              <input id="hp-poly-power" type="range" min="0.3" max="1.5" step="0.05" value="${state.poly_power}" class="w-full accent-blue-500" />
            </div>
          </div>

          <!-- Smart Implementation Alerts -->
          <div class="space-y-2 pt-2 border-t border-[var(--border)]">
            <div class="text-xs font-bold text-[var(--foreground)]">Smart Codebase Behavior Diagnostics</div>
            ${seriesData.alerts
              .map(
                (a) => `
                <div class="p-2.5 rounded-lg border text-xs leading-relaxed ${
                  a.level === "warning"
                    ? "border-amber-500/60 bg-amber-500/10 text-amber-200"
                    : a.level === "caution"
                    ? "border-rose-500/60 bg-rose-500/10 text-rose-200"
                    : "border-blue-500/50 bg-blue-500/10 text-blue-200"
                }">
                  <div class="font-bold flex justify-between">
                    <span>${escapeHtml(a.title)}</span>
                  </div>
                  <div class="text-xs opacity-90 mt-1">${escapeHtml(a.message)}</div>
                  <div class="text-xs font-mono opacity-75 mt-1">Ref: ${escapeHtml(a.code)}</div>
                </div>
              `
              )
              .join("")}
          </div>
        </div>

        <!-- Right 8 Cols: Synchronized Multi-Curve SVG Charts -->
        <div class="lg:col-span-8 space-y-3">
          <!-- Chart 1: Consistency Weight Schedule w_cons(t) -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
            <div class="flex flex-wrap items-center justify-between text-xs mb-2">
              <span class="font-bold text-[var(--foreground)]">
                1. Dynamic Consistency Weight <code>w_cons(t)</code> — Sigmoid Ramp-up (${escapeHtml(stepMetrics.rampupUnit)}) + Cosine Ramp-down
              </span>
              <span class="font-mono text-amber-400 font-bold">
                At iter ${state.iter_num}: w_cons = ${stepMetrics.consW.toFixed(5)} (rampup=${stepMetrics.rampupFactor.toFixed(3)}, decay=${stepMetrics.decayFactor.toFixed(3)})
              </span>
            </div>
            <svg viewBox="0 0 750 180" class="w-full h-40 bg-[var(--background)] border border-[var(--border)] rounded-lg">
              <line x1="45" y1="25" x2="725" y2="25" stroke="#334155" stroke-dasharray="3,3" />
              <line x1="45" y1="90" x2="725" y2="90" stroke="#334155" stroke-dasharray="3,3" />
              <line x1="45" y1="155" x2="725" y2="155" stroke="#475569" />
              <text x="40" y="29" text-anchor="end" fill="#94a3b8" font-size="12">${maxCons.toFixed(3)}</text>
              <text x="40" y="94" text-anchor="end" fill="#94a3b8" font-size="12">${(maxCons / 2).toFixed(3)}</text>
              <text x="40" y="158" text-anchor="end" fill="#94a3b8" font-size="12">0.000</text>
              <polyline fill="none" stroke="#f59e0b" stroke-width="2.8" points="${consPoints}" />
              <line x1="${cursorX}" y1="15" x2="${cursorX}" y2="160" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="4,3" />
              <text x="45" y="173" fill="#94a3b8" font-size="12">iter 0</text>
              <text x="385" y="173" text-anchor="middle" fill="#94a3b8" font-size="12">iter ${Math.round(maxIter / 2)}</text>
              <text x="725" y="173" text-anchor="end" fill="#94a3b8" font-size="12">iter ${maxIter}</text>
            </svg>
          </div>

          <!-- Chart 2: EMA Decay Schedule alpha_t & Teacher Memory Horizon -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
            <div class="flex flex-wrap items-center justify-between text-xs mb-2">
              <span class="font-bold text-[var(--foreground)]">
                2. EMA Decay <code>α_t</code> & Effective Teacher Memory Window <code>N_eff = 1 / (1 - α_t)</code>
              </span>
              <span class="font-mono text-purple-400 font-bold">
                At iter ${state.iter_num}: α_t = ${stepMetrics.actualAlpha.toFixed(5)} | Student Gain (1-α_t) = ${(stepMetrics.studentUpdateRate * 100).toFixed(3)}% | Memory ≈ ${Math.round(stepMetrics.effectiveMemorySteps)} steps
              </span>
            </div>
            <svg viewBox="0 0 750 180" class="w-full h-40 bg-[var(--background)] border border-[var(--border)] rounded-lg">
              <line x1="45" y1="25" x2="725" y2="25" stroke="#334155" stroke-dasharray="3,3" />
              <line x1="45" y1="90" x2="725" y2="90" stroke="#334155" stroke-dasharray="3,3" />
              <line x1="45" y1="155" x2="725" y2="155" stroke="#475569" />
              <text x="40" y="29" text-anchor="end" fill="#94a3b8" font-size="12">1.000</text>
              <text x="40" y="94" text-anchor="end" fill="#94a3b8" font-size="12">0.992</text>
              <text x="40" y="158" text-anchor="end" fill="#94a3b8" font-size="12">0.985</text>
              <polyline fill="none" stroke="#c084fc" stroke-width="2.8" points="${alphaPoints}" />
              <line x1="${cursorX}" y1="15" x2="${cursorX}" y2="160" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="4,3" />
            </svg>
          </div>

          <!-- Chart 3: Learning Rate Schedule lr(t) -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
            <div class="flex flex-wrap items-center justify-between text-xs mb-2">
              <span class="font-bold text-[var(--foreground)]">
                3. Learning Rate Schedule <code>lr(t) = base_lr * (1 - t/T_max)^${state.poly_power}</code>
              </span>
              <span class="font-mono text-blue-400 font-bold">
                At iter ${state.iter_num}: lr = ${stepMetrics.lr.toFixed(6)} (${Math.round((stepMetrics.lr / maxLr) * 100)}% of base_lr)
              </span>
            </div>
            <svg viewBox="0 0 750 180" class="w-full h-40 bg-[var(--background)] border border-[var(--border)] rounded-lg">
              <line x1="45" y1="25" x2="725" y2="25" stroke="#334155" stroke-dasharray="3,3" />
              <line x1="45" y1="90" x2="725" y2="90" stroke="#334155" stroke-dasharray="3,3" />
              <line x1="45" y1="155" x2="725" y2="155" stroke="#475569" />
              <text x="40" y="29" text-anchor="end" fill="#94a3b8" font-size="12">${maxLr.toFixed(3)}</text>
              <text x="40" y="94" text-anchor="end" fill="#94a3b8" font-size="12">${(maxLr / 2).toFixed(4)}</text>
              <text x="40" y="158" text-anchor="end" fill="#94a3b8" font-size="12">0.000</text>
              <polyline fill="none" stroke="#38bdf8" stroke-width="2.8" points="${lrPoints}" />
              <line x1="${cursorX}" y1="15" x2="${cursorX}" y2="160" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="4,3" />
            </svg>
          </div>
        </div>
      </div>
    `;
  }

  // ============================================================================
  // VIEW 5: GRADIENT FLOW, AMP OVERFLOW GUARD & 2D EMA TRAJECTORY LAB
  // ============================================================================
  function renderGradientAndEmaLabView(stepMetrics) {
    // Generate 2D parameter space trajectory (Student noisy SGD steps vs smooth EMA Teacher)
    const numSteps = 35;
    const studentPts = [];
    const teacherPts = [];
    let sx = 65,
      sy = 210;
    let tx = 65,
      ty = 210;
    const targetX = 520,
      targetY = 85;
    const noiseAmp = Number(state.traj_noise_scale) || 0.35;
    // Scale visualization alpha so user sees the lag vs smoothing clearly on 35 macro-steps
    const effVisAlpha = Math.min(0.92, Math.max(0.5, (stepMetrics.actualAlpha - 0.98) * 22));

    for (let i = 0; i <= numSteps; i++) {
      studentPts.push(`${sx.toFixed(1)},${sy.toFixed(1)}`);
      teacherPts.push(`${tx.toFixed(1)},${ty.toFixed(1)}`);
      const prog = i / numSteps;
      const oscX = Math.sin(i * 1.7) * 28 * noiseAmp * (1 - prog * 0.6);
      const oscY = Math.cos(i * 1.3) * 42 * noiseAmp * (1 - prog * 0.5);
      sx = sx + (targetX - sx) * 0.14 + oscX;
      sy = sy + (targetY - sy) * 0.14 + oscY;
      tx = effVisAlpha * tx + (1 - effVisAlpha) * sx;
      ty = effVisAlpha * ty + (1 - effVisAlpha) * sy;
    }

    const overflow = Boolean(state.amp_simulate_overflow);

    return `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <!-- Left: 2D Parameter Trajectory Playground (Student SGD vs Teacher EMA) -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm flex flex-col gap-3">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 class="text-sm font-bold text-[var(--foreground)]">
                2D Parameter Space Trajectory: Stochastic Student (θ) vs Weight-Averaged Teacher (θ')
              </h3>
              <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
                Minh họa cơ chế cốt lõi của Tarvainen & Valpola (2017): Tại sao trung bình trượt trọng số (EMA) tạo ra dự đoán ổn định hơn trung bình nhãn trực tiếp.
              </p>
            </div>
          </div>

          <div class="flex items-center justify-between bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5 text-xs">
            <span>Mini-batch & Perturbation Gradient Noise: <strong>${Math.round(noiseAmp * 100)}%</strong></span>
            <input id="ctrl-traj-noise" type="range" min="0.05" max="1.0" step="0.05" value="${noiseAmp}" class="w-44 accent-blue-500" />
          </div>

          <svg viewBox="0 0 600 260" class="w-full h-60 bg-[var(--background)] border border-[var(--border)] rounded-xl">
            <!-- Loss Contour Ellipses around Optimum (520, 85) -->
            <ellipse cx="520" cy="85" rx="220" ry="110" fill="none" stroke="#1e293b" stroke-width="1.5" />
            <ellipse cx="520" cy="85" rx="150" ry="75" fill="none" stroke="#334155" stroke-width="1.5" />
            <ellipse cx="520" cy="85" rx="85" ry="42" fill="none" stroke="#475569" stroke-width="1.5" />
            <ellipse cx="520" cy="85" rx="30" ry="16" fill="#10b981" fill-opacity="0.15" stroke="#10b981" stroke-width="1.5" />
            <circle cx="520" cy="85" r="4" fill="#10b981" />
            <text x="520" y="62" text-anchor="middle" fill="#10b981" font-size="12" font-weight="bold">Local Optimum θ*</text>

            <!-- Student Noisy Trajectory -->
            <polyline fill="none" stroke="#3b82f6" stroke-width="2" stroke-dasharray="4,3" points="${studentPts.join(" ")}" />
            <!-- Teacher Smooth EMA Trajectory -->
            <polyline fill="none" stroke="#c084fc" stroke-width="3.2" points="${teacherPts.join(" ")}" />

            <circle cx="65" cy="210" r="5" fill="#94a3b8" />
            <text x="65" y="230" text-anchor="middle" fill="#94a3b8" font-size="12">Init θ_0 = θ'_0</text>
          </svg>

          <div class="flex items-center justify-between text-xs text-[var(--muted-foreground)]">
            <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-blue-500 inline-block"></span> <strong>Student θ_t (SGD):</strong> Dao động theo từng mini-batch & nhiễu <code>student_noise_std</code></span>
            <span class="flex items-center gap-1.5"><span class="w-3 h-1 bg-purple-400 inline-block"></span> <strong>Teacher θ'_t (EMA):</strong> Triệt tiêu phương sai đạo hàm, hội tụ mượt vào tâm đáy phẳng</span>
          </div>
        </div>

        <!-- Right: Interactive AMP GradScaler & DDP Consensus State Machine -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm flex flex-col gap-3">
          <div class="flex items-center justify-between">
            <div>
              <h3 class="text-sm font-bold text-[var(--foreground)]">
                Coordinated AMP & DDP Gradient Handler State Machine (<code>training/gradient_handler.py</code>)
              </h3>
              <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
                Kiểm tra cách <code>GradientHandler.apply_gradients()</code> bảo vệ cả Student và EMA Teacher khi xảy ra tràn số FP16 trên bất kỳ GPU nào.
              </p>
            </div>
            <button
              id="btn-toggle-overflow"
              class="px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                overflow
                  ? "bg-rose-600 text-white shadow-sm"
                  : "bg-emerald-600 text-white shadow-sm"
              }"
            >
              ${overflow ? "⚠️ Simulating FP16 Inf/NaN on Rank 1" : "✓ Normal Finite Gradients (Click to Inject Inf)"}
            </button>
          </div>

          <div class="space-y-2 text-xs font-mono">
            <div class="p-2.5 rounded-lg border border-[var(--border)] bg-[var(--background)] flex justify-between items-center">
              <span>1. Check <code>torch.isfinite(loss)</code> + DDP <code>all_reduce(MIN)</code></span>
              <span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">PASS (loss_finite = 1)</span>
            </div>

            <div class="p-2.5 rounded-lg border border-[var(--border)] bg-[var(--background)] flex justify-between items-center">
              <span>2. <code>scaler.scale(loss).backward()</code> → <code>scaler.unscale_(optimizer)</code></span>
              <span class="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 font-bold">Student θ.grad populated | Teacher θ'.grad = None</span>
            </div>

            <div class="p-2.5 rounded-lg border ${
              overflow ? "border-rose-500 bg-rose-500/15" : "border-[var(--border)] bg-[var(--background)]"
            } flex justify-between items-center">
              <span>3. Inspect all <code>param.grad</code> + DDP <code>all_reduce(is_finite, MIN)</code></span>
              <span class="px-2 py-0.5 rounded font-bold ${
                overflow ? "bg-rose-600 text-white" : "bg-emerald-500/20 text-emerald-300"
              }">
                ${overflow ? "OVERFLOW DETECTED (is_finite == 0)" : "ALL RANKS FINITE (is_finite == 1)"}
              </span>
            </div>

            ${
              overflow
                ? `
                  <div class="p-3 rounded-lg border border-rose-500/70 bg-rose-500/10 text-rose-200 space-y-1.5 font-sans">
                    <div class="font-bold">🛑 Emergency Rollback Path Executed (lines 66–76):</div>
                    <ul class="list-disc pl-4 space-y-1">
                      <li><code>new_scale = scale_before * backoff_factor</code> (ví dụ <code>65536.0 → 32768.0</code>).</li>
                      <li><strong>BỎ QUA</strong> <code>clip_grad_norm_(12.0)</code> và <strong>BỎ QUA</strong> <code>scaler.step(optimizer)</code> trên MỌI GPU để giữ đồng bộ DDP.</li>
                      <li><strong>BỎ QUA</strong> <code>self.after_step_fn(iter_num)</code>: EMA Teacher không bị cập nhật khi Student không bước!</li>
                      <li><code>self.successful_updates</code> giữ nguyên, trả về <code>False</code>.</li>
                    </ul>
                  </div>
                `
                : `
                  <div class="p-3 rounded-lg border border-emerald-500/60 bg-emerald-500/10 text-emerald-200 space-y-1.5 font-sans">
                    <div class="font-bold">✅ Committed Update Path Executed (lines 78–88):</div>
                    <ul class="list-disc pl-4 space-y-1">
                      <li><code>torch.nn.utils.clip_grad_norm_(raw_m.parameters(), max_norm=12.0)</code> chống nổ đạo hàm 3D.</li>
                      <li><code>self.scaler.step(optimizer)</code> & <code>self.scaler.update()</code> cập nhật trọng số Student.</li>
                      <li>Gọi <code>self.after_step_fn(iter_num)</code> → kích hoạt <code>update_ema_variables(raw_net, raw_ema, α_t)</code>.</li>
                      <li>Tăng bộ đếm <code>self.successful_updates += 1</code>.</li>
                    </ul>
                  </div>
                `
            }
          </div>
        </div>
      </div>
    `;
  }

  // ============================================================================
  // VIEW 6: CONFIG DIFF, PAPER VS IMPLEMENTATION & EMPIRICAL RUN FORENSICS
  // ============================================================================
  function renderForensicsView(methodSpec) {
    const run = DATA.runs[state.selectedRunId] || DATA.runs["o6s4pex3"];
    const traj = run.validation_trajectory || [];
    const breakdown = run.class_breakdown_final || { classes: [], loss_windows: [] };

    // Build SVG validation trajectory curves for Legacy mDice vs Positive Macro Dice
    const maxIt = traj.length > 0 ? traj[traj.length - 1].iteration : 10000;
    function mapTrajLine(key) {
      return traj
        .map((r) => {
          const x = 45 + (r.iteration / maxIt) * 660;
          const y = 165 - Math.min(0.65, Math.max(0, r[key])) / 0.65 * 140;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
    }

    const legacyLine = mapTrajLine("legacy_mdice");
    const posMacroLine = mapTrajLine("positive_macro");
    const iphLine = mapTrajLine("Intraparenchymal");
    const ivhLine = mapTrajLine("Intraventricular");

    return `
      <div class="space-y-4">
        <!-- Top: Empirical Run Telemetry Explorer (o6s4pex3 & 9tvq8qld) -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
          <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[var(--border)]">
            <div>
              <h2 class="text-base font-bold text-[var(--foreground)]">
                Empirical Run Telemetry & Per-Class Forensics (Verified from <code>docs/analysis/</code>)
              </h2>
              <p class="text-xs text-[var(--muted-foreground)] mt-0.5">${escapeHtml(run.description)}</p>
            </div>

            <div class="flex gap-2">
              ${Object.values(DATA.runs)
                .map(
                  (r) => `
                  <button
                    data-run-id="${r.id}"
                    class="px-3 py-1.5 rounded-lg text-xs font-semibold ${
                      r.id === state.selectedRunId
                        ? "bg-blue-600 text-white"
                        : "bg-[var(--background)] text-[var(--muted-foreground)] border border-[var(--border)]"
                    }"
                  >
                    Run ${escapeHtml(r.id)}
                  </button>
                `
                )
                .join("")}
            </div>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 mt-3">
            <!-- Trajectory Chart -->
            <div class="lg:col-span-7">
              <div class="flex flex-wrap items-center justify-between text-xs mb-1.5">
                <span class="font-bold text-[var(--foreground)]">Validation Trajectory: Legacy mDice vs Positive Macro Dice</span>
                <div class="flex gap-3">
                  <span class="text-amber-400 font-semibold">— Legacy mDice (Best @ ${run.summary.best_legacy.iteration})</span>
                  <span class="text-emerald-400 font-semibold">— Positive Macro (Best @ ${run.summary.best_positive_macro.iteration})</span>
                  <span class="text-sky-400">— IPH</span>
                  <span class="text-purple-400">— IVH</span>
                </div>
              </div>
              <svg viewBox="0 0 730 185" class="w-full h-48 bg-[var(--background)] border border-[var(--border)] rounded-lg">
                <line x1="45" y1="25" x2="705" y2="25" stroke="#334155" stroke-dasharray="3,3" />
                <line x1="45" y1="95" x2="705" y2="95" stroke="#334155" stroke-dasharray="3,3" />
                <line x1="45" y1="165" x2="705" y2="165" stroke="#475569" />
                <text x="40" y="29" text-anchor="end" fill="#94a3b8" font-size="12">0.65</text>
                <text x="40" y="98" text-anchor="end" fill="#94a3b8" font-size="12">0.32</text>
                <text x="40" y="168" text-anchor="end" fill="#94a3b8" font-size="12">0.00</text>
                <!-- All-background analytic baseline 0.5579 -->
                <line x1="45" y1="44.8" x2="705" y2="44.8" stroke="#f43f5e" stroke-width="1.2" stroke-dasharray="5,4" />
                <text x="700" y="41" text-anchor="end" fill="#f43f5e" font-size="12">All-BG Legacy mDice = 0.5579</text>

                <polyline fill="none" stroke="#f59e0b" stroke-width="2.4" points="${legacyLine}" />
                <polyline fill="none" stroke="#10b981" stroke-width="2.8" points="${posMacroLine}" />
                <polyline fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="3,2" points="${iphLine}" />
                <polyline fill="none" stroke="#c084fc" stroke-width="1.5" stroke-dasharray="3,2" points="${ivhLine}" />
                <text x="45" y="179" fill="#94a3b8" font-size="12">iter 200</text>
                <text x="705" y="179" text-anchor="end" fill="#94a3b8" font-size="12">iter ${maxIt}</text>
              </svg>
            </div>

            <!-- Loss Ratio Table -->
            <div class="lg:col-span-5">
              <div class="text-xs font-bold text-[var(--foreground)] mb-1.5">
                Training Loss Windows (Why Weighted Consistency ≈ 0.5% of Scalar Loss)
              </div>
              <table class="w-full text-left border-collapse text-xs">
                <thead>
                  <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                    <th class="py-1.5">Iter Window</th>
                    <th class="py-1.5">Mean L_sup</th>
                    <th class="py-1.5">Raw L_cons</th>
                    <th class="py-1.5 text-right">(w·L_cons)/L_sup</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-[var(--border)] font-mono">
                  ${breakdown.loss_windows
                    .map(
                      (w) => `
                      <tr>
                        <td class="py-1.5 text-[var(--foreground)]">${escapeHtml(w.window)}</td>
                        <td class="py-1.5 text-blue-400">${w.sup_mean.toFixed(4)}</td>
                        <td class="py-1.5 text-amber-400">${w.cons_raw_mean.toFixed(4)}</td>
                        <td class="py-1.5 text-right font-bold text-purple-400">${w.weighted_ratio_pct.toFixed(4)}%</td>
                      </tr>
                    `
                    )
                    .join("")}
                </tbody>
              </table>
            </div>
          </div>

          <!-- Per-Class Validation Table -->
          <div class="mt-4 pt-3 border-t border-[var(--border)] overflow-x-auto">
            <div class="text-xs font-bold text-[var(--foreground)] mb-1.5">
              Per-Class Segmentation Breakdown at Iteration ${breakdown.iteration} (38 Validation Cases, Fold 0)
            </div>
            <table class="w-full text-left border-collapse text-xs">
              <thead>
                <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                  <th class="py-1.5">Hemorrhage Class</th>
                  <th class="py-1.5">Positive Cases / 38</th>
                  <th class="py-1.5">Positive Case Dice</th>
                  <th class="py-1.5">Legacy Case-Mean Dice</th>
                  <th class="py-1.5">Voxel Precision</th>
                  <th class="py-1.5">Voxel Recall</th>
                  <th class="py-1.5">Neg Cases with FP</th>
                  <th class="py-1.5 text-right">Mean FP Volume (mL)</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[var(--border)] font-mono">
                ${breakdown.classes
                  .map(
                    (c) => `
                    <tr>
                      <td class="py-1.5 font-sans font-bold text-[var(--foreground)]">${escapeHtml(c.name)}</td>
                      <td class="py-1.5">${c.pos_cases} / 38</td>
                      <td class="py-1.5 font-bold ${c.pos_dice === 0 ? "text-rose-400" : "text-emerald-400"}">${c.pos_dice.toFixed(4)}</td>
                      <td class="py-1.5 text-amber-400">${c.case_mean_dice.toFixed(4)}</td>
                      <td class="py-1.5">${c.pos_dice === 0 ? "Undefined (TP=FP=0)" : c.precision.toFixed(4)}</td>
                      <td class="py-1.5">${c.recall.toFixed(4)}</td>
                      <td class="py-1.5">${c.fp_neg_cases} / ${c.neg_cases}</td>
                      <td class="py-1.5 text-right">${c.fp_ml_mean.toFixed(3)} mL</td>
                    </tr>
                  `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Bottom: Multi-Tier Config Provenance & Paper vs Implementation Table -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <!-- Config Diff Table -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
            <h3 class="text-sm font-bold text-[var(--foreground)]">
              Multi-Tier Configuration Comparison (<code>core/config_parser.py:load_config</code>)
            </h3>
            <p class="text-xs text-[var(--muted-foreground)] mt-0.5">
              Đối chiếu trực tiếp sự khác biệt giữa <code>configs/base.yaml</code>, <code>configs/methods/mean_teacher.yaml</code> và <code>configs/experiments/bhsd_mt_nnunet_fold0.yaml</code>:
            </p>
            <div class="mt-3 overflow-x-auto">
              <table class="w-full text-left border-collapse text-xs">
                <thead>
                  <tr class="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                    <th class="py-1.5">Parameter</th>
                    <th class="py-1.5">base.yaml</th>
                    <th class="py-1.5 text-emerald-400">mean_teacher.yaml</th>
                    <th class="py-1.5 text-amber-400">bhsd_mt_nnunet_fold0.yaml</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-[var(--border)] font-mono">
                  <tr><td class="py-1.5">data.patch_size</td><td>[32, 160, 160]</td><td class="text-emerald-400 font-bold">[28, 256, 256]</td><td class="text-amber-400 font-bold">[32, 192, 192]</td></tr>
                  <tr><td class="py-1.5">data.batch_size / labeled_bs</td><td>4 / 2</td><td class="text-emerald-400 font-bold">8 / 4</td><td class="text-amber-400 font-bold">4 / 2</td></tr>
                  <tr><td class="py-1.5">data.labelnum</td><td>15</td><td class="text-emerald-400 font-bold">154</td><td class="text-amber-400 font-bold">154</td></tr>
                  <tr><td class="py-1.5">model.name (stages)</td><td>vnet (5)</td><td class="text-emerald-400 font-bold">nnunet3d (7)</td><td class="text-amber-400 font-bold">nnunet3d (7)</td></tr>
                  <tr><td class="py-1.5">method.consistency (λ)</td><td>—</td><td class="text-emerald-400 font-bold">0.05</td><td class="text-rose-400 font-bold">1.0 (20× higher!)</td></tr>
                  <tr><td class="py-1.5">method.consistency_rampup</td><td>—</td><td class="text-emerald-400 font-bold">4000.0 (iters)</td><td class="text-rose-400 font-bold">40.0 (epochs!)</td></tr>
                  <tr><td class="py-1.5">method.ema_decay (α)</td><td>—</td><td class="text-emerald-400 font-bold">0.999</td><td class="text-rose-400 font-bold">0.99 (flat cosine!)</td></tr>
                  <tr><td class="py-1.5">method.ema_schedule</td><td>—</td><td class="text-emerald-400">cosine</td><td>cosine (inherited)</td></tr>
                  <tr><td class="py-1.5">consistency_rampdown_start</td><td>—</td><td class="text-emerald-400">7800</td><td>7800 (inherited)</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Paper vs Code Implementation Table -->
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
            <h3 class="text-sm font-bold text-[var(--foreground)]">
              Original Paper (NeurIPS 2017) vs <code>BHSD_SSL</code> Implementation
            </h3>
            <div class="mt-2 space-y-2 text-xs">
              ${(DATA.methods.mean_teacher.paper_vs_impl || [])
                .map(
                  (item) => `
                  <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5">
                    <div class="font-bold text-blue-400">${escapeHtml(item.dimension)}</div>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-1">
                      <div><span class="text-[var(--muted-foreground)]">Paper:</span> ${escapeHtml(item.paper)}</div>
                      <div><span class="text-emerald-400 font-semibold">Repo Code:</span> ${escapeHtml(item.impl)}</div>
                    </div>
                  </div>
                `
                )
                .join("")}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // ============================================================================
  // MAIN RENDER & EVENT BINDING
  // ============================================================================
  function renderApp() {
    const root = document.getElementById("medssl-app-root");
    if (!root) return;

    const methodSpec = DATA.methods[state.method] || DATA.methods["mean_teacher"];
    const hierarchy = Core.computeBackboneHierarchy(state, DATA.backbones);
    const stepMetrics = Core.evaluateSchedulesAtStep(state.iter_num, state, hierarchy.epochLen);

    let viewHtml = "";
    if (state.activeView === "pipeline") {
      viewHtml = renderPipelineView(hierarchy, stepMetrics, methodSpec);
    } else if (state.activeView === "tensors") {
      viewHtml = renderTensorInspectorView(hierarchy);
    } else if (state.activeView === "math") {
      viewHtml = renderMathSandboxView(hierarchy, stepMetrics);
    } else if (state.activeView === "schedules") {
      viewHtml = renderScheduleTimeMachineView(hierarchy, stepMetrics);
    } else if (state.activeView === "grad_ema") {
      viewHtml = renderGradientAndEmaLabView(stepMetrics);
    } else if (state.activeView === "forensics") {
      viewHtml = renderForensicsView(methodSpec);
    }

    root.innerHTML = renderGlobalHeader(hierarchy, stepMetrics, methodSpec) + `<main>${viewHtml}</main>`;
    bindEvents(methodSpec);
  }

  function bindEvents(methodSpec) {
    const selMethod = document.getElementById("select-method");
    if (selMethod) {
      selMethod.onchange = (e) => {
        state.method = e.target.value;
        const m = DATA.methods[state.method];
        state.selectedNodeId = m.nodes[0].id;
        state.currentStep = 1;
        renderApp();
      };
    }

    const selPreset = document.getElementById("select-preset");
    if (selPreset) {
      selPreset.onchange = (e) => {
        if (e.target.value !== "custom") {
          applyPreset(e.target.value);
        }
      };
    }

    const iterSlider = document.getElementById("global-iter-slider");
    if (iterSlider) {
      iterSlider.oninput = (e) => {
        state.iter_num = parseInt(e.target.value, 10) || 0;
        renderApp();
      };
    }

    document.querySelectorAll("[data-view-tab]").forEach((btn) => {
      btn.onclick = () => {
        state.activeView = btn.getAttribute("data-view-tab");
        renderApp();
      };
    });

    // Pipeline View bindings
    document.querySelectorAll("[data-node-id]").forEach((el) => {
      el.onclick = () => {
        state.selectedNodeId = el.getAttribute("data-node-id");
        renderApp();
      };
    });

    document.querySelectorAll("[data-step-num]").forEach((btn) => {
      btn.onclick = () => {
        state.currentStep = parseInt(btn.getAttribute("data-step-num"), 10) || 1;
        renderApp();
      };
    });

    const btnPrev = document.getElementById("btn-step-prev");
    if (btnPrev) {
      btnPrev.onclick = () => {
        const total = methodSpec.execution_steps.length;
        state.currentStep = state.currentStep <= 1 ? total : state.currentStep - 1;
        const activeNodes = methodSpec.execution_steps[state.currentStep - 1].active_nodes;
        if (activeNodes && activeNodes.length > 0) state.selectedNodeId = activeNodes[activeNodes.length - 1];
        renderApp();
      };
    }

    const btnNext = document.getElementById("btn-step-next");
    if (btnNext) {
      btnNext.onclick = () => {
        const total = methodSpec.execution_steps.length;
        state.currentStep = state.currentStep >= total ? 1 : state.currentStep + 1;
        const activeNodes = methodSpec.execution_steps[state.currentStep - 1].active_nodes;
        if (activeNodes && activeNodes.length > 0) state.selectedNodeId = activeNodes[activeNodes.length - 1];
        renderApp();
      };
    }

    const btnPlay = document.getElementById("btn-step-play");
    if (btnPlay) {
      btnPlay.onclick = () => {
        state.isPlaying = !state.isPlaying;
        if (state.playTimer) {
          clearInterval(state.playTimer);
          state.playTimer = null;
        }
        if (state.isPlaying) {
          state.playTimer = setInterval(() => {
            const curMethod = DATA.methods[state.method];
            const total = curMethod.execution_steps.length;
            state.currentStep = state.currentStep >= total ? 1 : state.currentStep + 1;
            const activeNodes = curMethod.execution_steps[state.currentStep - 1].active_nodes;
            if (activeNodes && activeNodes.length > 0) state.selectedNodeId = activeNodes[activeNodes.length - 1];
            renderApp();
          }, 2400);
        }
        renderApp();
      };
    }

    // View 2 Tensor controls
    const ctrlBb = document.getElementById("ctrl-backbone");
    if (ctrlBb) {
      ctrlBb.onchange = (e) => {
        state.backbone = e.target.value;
        state.activePreset = "custom";
        renderApp();
      };
    }
    const ctrlStages = document.getElementById("ctrl-stages");
    if (ctrlStages) {
      ctrlStages.onchange = (e) => {
        state.num_stages = parseInt(e.target.value, 10);
        state.activePreset = "custom";
        renderApp();
      };
    }
    const ctrlAniso = document.getElementById("ctrl-aniso");
    if (ctrlAniso) {
      ctrlAniso.onchange = (e) => {
        state.anisotropic_kernels = e.target.value === "true";
        state.activePreset = "custom";
        renderApp();
      };
    }
    const ctrlPatch = document.getElementById("ctrl-patch");
    if (ctrlPatch) {
      ctrlPatch.onchange = (e) => {
        const parts = e.target.value.split(",").map(Number);
        state.patch_z = parts[0];
        state.patch_y = parts[1];
        state.patch_x = parts[2];
        state.activePreset = "custom";
        renderApp();
      };
    }
    const ctrlBatch = document.getElementById("ctrl-batch");
    if (ctrlBatch) {
      ctrlBatch.onchange = (e) => {
        const parts = e.target.value.split(",").map(Number);
        state.batch_size = parts[0];
        state.labeled_bs = parts[1];
        state.world_size = parts[2];
        state.activePreset = "custom";
        renderApp();
      };
    }

    // View 3 Math & Paradox controls
    document.querySelectorAll("[data-formula-id]").forEach((btn) => {
      btn.onclick = () => {
        state.activeFormulaId = btn.getAttribute("data-formula-id");
        renderApp();
      };
    });
    const simConsType = document.getElementById("sim-cons-type");
    if (simConsType) {
      simConsType.onchange = (e) => {
        state.consistency_type = e.target.value;
        state.activePreset = "custom";
        renderApp();
      };
    }
    const simBgW = document.getElementById("sim-bg-weight");
    if (simBgW) {
      simBgW.oninput = (e) => {
        state.bg_consistency_weight = parseFloat(e.target.value);
        state.activePreset = "custom";
        renderApp();
      };
    }
    const simConf = document.getElementById("sim-conf-thresh");
    if (simConf) {
      simConf.oninput = (e) => {
        state.confidence_thresh = parseFloat(e.target.value);
        state.activePreset = "custom";
        renderApp();
      };
    }
    const simParPos = document.getElementById("sim-paradox-pos");
    if (simParPos) {
      simParPos.oninput = (e) => {
        state.paradox_pos_dice_level = parseFloat(e.target.value);
        renderApp();
      };
    }
    const simParFp = document.getElementById("sim-paradox-fp");
    if (simParFp) {
      simParFp.oninput = (e) => {
        state.paradox_fp_case_rate = parseFloat(e.target.value);
        renderApp();
      };
    }

    // View 4 Schedule controls
    const bindSlider = (id, key) => {
      const el = document.getElementById(id);
      if (el) {
        el.oninput = (e) => {
          state[key] = parseFloat(e.target.value);
          state.activePreset = "custom";
          renderApp();
        };
      }
    };
    bindSlider("hp-consistency", "consistency");
    bindSlider("hp-rampup", "consistency_rampup");
    bindSlider("hp-rampdown", "consistency_rampdown_start");
    bindSlider("hp-min-factor", "consistency_min_factor");
    bindSlider("hp-ema-decay", "ema_decay");
    bindSlider("hp-poly-power", "poly_power");

    const hpEmaSched = document.getElementById("hp-ema-sched");
    if (hpEmaSched) {
      hpEmaSched.onchange = (e) => {
        state.ema_schedule = e.target.value;
        state.activePreset = "custom";
        renderApp();
      };
    }

    // View 5 Gradient & EMA Lab controls
    const ctrlTrajNoise = document.getElementById("ctrl-traj-noise");
    if (ctrlTrajNoise) {
      ctrlTrajNoise.oninput = (e) => {
        state.traj_noise_scale = parseFloat(e.target.value);
        renderApp();
      };
    }
    const btnOverflow = document.getElementById("btn-toggle-overflow");
    if (btnOverflow) {
      btnOverflow.onclick = () => {
        state.amp_simulate_overflow = !state.amp_simulate_overflow;
        renderApp();
      };
    }

    // View 6 Run Forensics selector
    document.querySelectorAll("[data-run-id]").forEach((btn) => {
      btn.onclick = () => {
        state.selectedRunId = btn.getAttribute("data-run-id");
        renderApp();
      };
    });
  }

  return {
    init: renderApp,
    state,
  };
})();

document.addEventListener("DOMContentLoaded", function () {
  if (window.MedSSLViews) {
    window.MedSSLViews.init();
  }
});
