/**
 * Core Simulation, Symbolic Tensor Shape Propagator & Mathematical Engine
 * for the MedSSL-BHSD Research Method Visualization System.
 * 100% Method-Agnostic & Verified against BHSD_SSL Python Source Code.
 */

window.MedSSLCore = (function () {
  "use strict";

  // 1. Exact port of core/batching.py: resolve_batch_layout
  function resolveBatchLayout(batchSize, labeledBs, worldSize, method) {
    const ws = Math.max(1, parseInt(worldSize, 10) || 1);
    const bs = Math.max(1, parseInt(batchSize, 10) || 4);
    const lbsInput = Math.max(1, Math.min(bs, parseInt(labeledBs, 10) || 2));
    const labeled = method === "supervised" ? bs : lbsInput;
    const unlabeled = bs - labeled;

    const warnings = [];
    if (labeled % ws !== 0 || unlabeled % ws !== 0) {
      warnings.push(
        `[BatchLayout Error] Global batch (${labeled}L + ${unlabeled}U) không chia hết cho world_size=${ws} (core/batching.py:27).`
      );
    }
    const localL = Math.max(1, Math.floor(labeled / ws));
    const localU = method === "supervised" ? 0 : Math.max(1, Math.floor(unlabeled / ws));
    const localBs = localL + localU;

    if ((method === "pick" || method === "bcp") && (localL < 2 || localL % 2 !== 0 || localL !== localU)) {
      warnings.push(
        `[CutMix Pairing Warning] Phương pháp '${method}' yêu cầu local_labeled_bs == local_unlabeled_bs là số chẵn >= 2 trên mỗi GPU (hiện tại: ${localL}L + ${localU}U).`
      );
    }

    return {
      global_batch_size: bs,
      global_labeled_bs: labeled,
      global_unlabeled_bs: unlabeled,
      world_size: ws,
      local_batch_size: localBs,
      local_labeled_bs: localL,
      local_unlabeled_bs: localU,
      sub_bs: Math.max(1, Math.floor(localL / 2)),
      warnings: warnings,
    };
  }

  // 2. Exact port of losses/deep_supervision.py: compute_ds_weights & resolve_ds_weights
  function computeDsWeights(numScales, customWeights) {
    if (numScales <= 1) return [1.0];
    let raw = [];
    if (Array.isArray(customWeights) && customWeights.length >= numScales) {
      raw = customWeights.slice(0, numScales).map((x) => Math.max(0, Number(x) || 0));
    } else {
      for (let i = 0; i < numScales; i++) {
        raw.push(1.0 / Math.pow(2, i));
      }
    }
    const sum = raw.reduce((a, b) => a + b, 0) || 1.0;
    return raw.map((w) => w / sum);
  }

  // 3. Symbolic 3D Backbone Hierarchy & Tensor Shape Propagator
  function computeBackboneHierarchy(state, backboneSpecs) {
    const bbId = state.backbone || "nnunet3d";
    const spec = backboneSpecs[bbId] || backboneSpecs["nnunet3d"];
    const numStages = spec.supports_num_stages.includes(Number(state.num_stages))
      ? Number(state.num_stages)
      : spec.supports_num_stages[spec.supports_num_stages.length - 1];
    const aniso = spec.supports_anisotropic ? Boolean(state.anisotropic_kernels) : false;
    const dsEnabled = spec.supports_deep_supervision ? Boolean(state.deep_supervision) : false;

    const layout = resolveBatchLayout(state.batch_size, state.labeled_bs, state.world_size, state.method);
    const B = layout.local_batch_size;
    const B_L = layout.local_labeled_bs;
    const B_U = layout.local_unlabeled_bs;
    const C_in = 1;
    const numClasses = Number(state.num_classes) || 6;
    const Z0 = Number(state.patch_z) || 28;
    const Y0 = Number(state.patch_y) || 256;
    const X0 = Number(state.patch_x) || 256;

    const channels = spec.stage_rules.channels_by_stages[String(numStages)] || [32, 64, 128, 256, 320];
    const kernels = aniso ? spec.stage_rules.kernels_anisotropic : spec.stage_rules.kernels_isotropic;
    const strides = aniso ? spec.stage_rules.pool_strides_anisotropic : spec.stage_rules.pool_strides_isotropic;

    const encoderStages = [];
    let curZ = Z0,
      curY = Y0,
      curX = X0;
    const spatialWarnings = [];

    for (let s = 0; s < numStages; s++) {
      const isBottleneck = s === numStages - 1;
      const inCh = s === 0 ? C_in : channels[s - 1];
      const outCh = channels[s];
      const k = kernels[s] || [3, 3, 3];
      let poolStride = [1, 1, 1];
      if (s > 0) {
        poolStride = strides[s - 1] || [2, 2, 2];
        if (curZ % poolStride[0] !== 0 || curY % poolStride[1] !== 0 || curX % poolStride[2] !== 0) {
          spatialWarnings.push(
            `Stage ${s + 1} (${curZ}×${curY}×${curX}) không chia hết cho stride (${poolStride.join(",")}).`
          );
        }
        curZ = Math.max(1, Math.floor(curZ / poolStride[0]));
        curY = Math.max(1, Math.floor(curY / poolStride[1]));
        curX = Math.max(1, Math.floor(curX / poolStride[2]));
      }
      encoderStages.push({
        stage: s + 1,
        name: isBottleneck ? `bottleneck (Stage ${s + 1})` : `enc${s + 1}`,
        pool_name: s === 0 ? "Input Volume" : `pool${s} (stride ${poolStride.join("×")})`,
        stride_from_prev: poolStride,
        kernel: k,
        in_channels: inCh,
        out_channels: outCh,
        spatial: [curZ, curY, curX],
        tensor_shape: [B, outCh, curZ, curY, curX],
        is_anisotropic_kernel: k[0] === 1,
        is_z_preserved: s > 0 && poolStride[0] === 1,
        scale_factor_z: Math.round(Z0 / curZ),
        scale_factor_xy: Math.round(Y0 / curY),
      });
    }

    // Decoder stages (from numStages - 1 down to 1)
    const decoderStages = [];
    const auxOutputs = [];
    for (let s = numStages - 2; s >= 0; s--) {
      const encTarget = encoderStages[s];
      const prevDecOrBtn = s === numStages - 2 ? encoderStages[numStages - 1] : decoderStages[decoderStages.length - 1];
      const upStride = strides[s] || [2, 2, 2];
      const outCh = encTarget.out_channels;
      const concatCh = outCh * 2; // cat([up, skip])
      const z = encTarget.spatial[0];
      const y = encTarget.spatial[1];
      const x = encTarget.spatial[2];
      const hasAuxHead = dsEnabled && s >= 1; // d2..d6 have ds_conv2..ds_conv6; d1 has final_conv
      const headName = s === 0 ? "final_conv" : hasAuxHead ? `ds_conv${s + 1}` : null;

      const decObj = {
        stage: s + 1,
        name: `dec${s + 1}`,
        up_name: `up${s + 1} (stride ${upStride.join("×")})`,
        skip_from: `enc${s + 1}`,
        kernel: kernels[s] || [3, 3, 3],
        up_in_channels: prevDecOrBtn.out_channels,
        skip_channels: outCh,
        concat_channels: concatCh,
        out_channels: outCh,
        spatial: [z, y, x],
        tensor_shape: [B, outCh, z, y, x],
        head_name: headName,
        head_shape: headName ? [B, numClasses, z, y, x] : null,
        scale_label: `1/${Math.round(Y0 / y)} XY, 1/${Math.round(Z0 / z)} Z`,
      };
      decoderStages.push(decObj);
    }

    // Order aux outputs to match models/backbones/nnunet3d.py lines 258-266:
    // aux = [ds_conv2(d2), ds_conv3(d3), ds_conv4(d4), ds_conv5(d5), ds_conv6(d6)]
    if (dsEnabled) {
      for (let s = 1; s < numStages - 1; s++) {
        const decMatch = decoderStages.find((d) => d.stage === s + 1);
        if (decMatch && decMatch.head_shape) {
          auxOutputs.push({
            index: s - 1,
            name: decMatch.head_name,
            from_decoder: decMatch.name,
            raw_shape: decMatch.head_shape,
            labeled_slice_shape: [B_L, numClasses, decMatch.spatial[0], decMatch.spatial[1], decMatch.spatial[2]],
            upsampled_labeled_shape: [B_L, numClasses, Z0, Y0, X0],
            scale_xy: 1.0 / Math.round(Y0 / decMatch.spatial[1]),
          });
        }
      }
    }

    const totalScales = 1 + auxOutputs.length;
    const dsWeights = computeDsWeights(totalScales);

    // Map node shape keys for instant inspection
    const nodeShapes = {
      batch_input: [
        { name: "volume_batch (Local GPU)", shape: [B, 1, Z0, Y0, X0], dtype: "float32", note: `${B_L} Labeled + ${B_U} Unlabeled` },
        { name: "label_batch (Local GPU)", shape: [B, Z0, Y0, X0], dtype: "int64", note: `[0..${B_L - 1}] GT ∈ {0..${numClasses - 1}}, [${B_L}..${B - 1}] ≡ 0` },
      ],
      student_input: [
        { name: "scale (Intensity)", shape: [B, 1, 1, 1, 1], dtype: "float32", note: `U(${(1 - state.student_intensity_scale).toFixed(2)}, ${(1 + state.student_intensity_scale).toFixed(2)})` },
        { name: "noise (Gaussian)", shape: [B, 1, Z0, Y0, X0], dtype: "float32", note: `N(0, ${state.student_noise_std}²)` },
        { name: "student_input", shape: [B, 1, Z0, Y0, X0], dtype: "float32", note: state.perturb_student ? "Perturbed Volume" : "Unperturbed (perturb_student=False)" },
      ],
      student_outputs: [
        { name: "student_logits (main 'seg')", shape: [B, numClasses, Z0, Y0, X0], dtype: "float16 (AMP)", note: `weight w̃_0 = ${dsWeights[0].toFixed(4)}` },
        ...auxOutputs.map((a, idx) => ({
          name: `student_aux[${idx}] (${a.name})`,
          shape: a.raw_shape,
          dtype: "float16 (AMP)",
          note: `weight w̃_${idx + 1} = ${dsWeights[idx + 1].toFixed(4)}`,
        })),
      ],
      teacher_outputs: [
        { name: "teacher_logits (no_grad)", shape: [B, numClasses, Z0, Y0, X0], dtype: "float16 (AMP)", note: "eval() mode on clean volume_batch" },
        {
          name: "teacher_aux",
          shape: dsEnabled && state.consistency_multiscale ? `${auxOutputs.length} aux tensors` : "[] (Empty List)",
          dtype: "no_grad",
          note: state.consistency_multiscale ? "return_all_scales=True" : "Skipped (consistency_multiscale=False)",
        },
      ],
      sup_loss_tensors: [
        { name: "lab_logits", shape: [B_L, numClasses, Z0, Y0, X0], dtype: "float16", note: `Sliced [:labeled_bs=${B_L}]` },
        { name: "lab_targets", shape: [B_L, Z0, Y0, X0], dtype: "int64", note: `Ground truth labels` },
        ...(dsEnabled
          ? [
              {
                name: "aux_lab (Trilinear Upsampled)",
                shape: [B_L, numClasses, Z0, Y0, X0],
                dtype: "float16",
                note: `F.interpolate(${auxOutputs.length} scales -> ${Z0}×${Y0}×${X0})`,
              },
            ]
          : []),
      ],
      cons_loss_tensors: [
        { name: "student_prob", shape: [B, numClasses, Z0, Y0, X0], dtype: "float32", note: "F.softmax(student_logits.float(), dim=1)" },
        { name: "teacher_prob", shape: [B, numClasses, Z0, Y0, X0], dtype: "float32", note: "F.softmax(teacher_logits.float(), dim=1)" },
        { name: "fg_s & fg_t (Soft Dice)", shape: [B, Math.max(1, numClasses - 1), Z0, Y0, X0], dtype: "float32", note: "Foreground channels [:, 1:] (Excludes BG)" },
        { name: "mse_per_class", shape: [numClasses], dtype: "float32", note: `Weighted by [${state.bg_consistency_weight}, 1, 1, 1, 1, 1]` },
      ],
      optimizer_state: [
        { name: "total_loss", shape: [1], dtype: "float32 (scalar)", note: "L_sup + w_cons(t) * L_cons" },
        { name: "Student param.grad", shape: "All θ_student", dtype: "float32 (unscaled)", note: "clip_grad_norm_(max_norm=12.0)" },
      ],
      validation_state: [
        { name: "Gaussian 3D Weight Map", shape: [Z0, Y0, X0], dtype: "float32", note: "sigma_scale = 1/8, center = 1.0" },
        { name: "Sliding Window Score Map", shape: [numClasses, "W_Z", "W_Y", "W_X"], dtype: "float32", note: "Accumulated over overlapping patches (stride Z=8, XY=32)" },
      ],
      pseudo_labels: [
        { name: "pl1, pl2 (Argmax Targets)", shape: [B, Z0, Y0, X0], dtype: "int64", note: "Discrete class indices {0..5} (stop-gradient)" },
      ],
    };

    const labelnum = Math.max(1, parseInt(state.labelnum, 10) || 154);
    const epochLen = Math.max(1, Math.floor(Math.floor(labelnum / layout.world_size) / layout.local_labeled_bs));

    return {
      backboneSpec: spec,
      numStages,
      aniso,
      dsEnabled,
      layout,
      epochLen,
      encoderStages,
      decoderStages,
      auxOutputs,
      dsWeights,
      nodeShapes,
      spatialWarnings,
      zDownsampleTotal: Math.round(Z0 / encoderStages[encoderStages.length - 1].spatial[0]),
      xyDownsampleTotal: Math.round(Y0 / encoderStages[encoderStages.length - 1].spatial[1]),
    };
  }

  // 4. Exact port of Schedule Evaluators (LR, Rampup, Rampdown, EMA Alpha)
  function sigmoidRampup(current, rampupLength) {
    if (rampupLength <= 0) return 1.0;
    const c = Math.max(0.0, Math.min(current, rampupLength));
    const phase = 1.0 - c / rampupLength;
    return Math.exp(-5.0 * phase * phase);
  }

  function evaluateSchedulesAtStep(iterNum, state, epochLen) {
    const maxIter = Math.max(1, Number(state.max_iteration) || 12000);
    const t = Math.max(0, Math.min(iterNum, maxIter));
    const curEpoch = t / Math.max(1, epochLen);

    // 4.1 Learning Rate (core/optimizers.py:64-96)
    const baseLr = Number(state.base_lr) || 0.01;
    const polyPower = Number(state.poly_power) || 0.7;
    const warmupIters = Math.max(0, parseInt(state.warmup_iters, 10) || 0);
    const schedType = (state.lr_scheduler || "poly").toLowerCase();
    const minLr = 1e-6;
    let lr = baseLr;
    if (warmupIters > 0 && t < warmupIters) {
      lr = Math.max(baseLr * (t / Math.max(1, warmupIters)), minLr);
    } else {
      const effIter = t - warmupIters;
      const effMax = Math.max(1, maxIter - warmupIters);
      if (schedType === "cosine" || schedType === "cosine_annealing") {
        const prog = Math.min(Math.max(effIter / effMax, 0.0), 1.0);
        lr = Math.max(baseLr * 0.5 * (1.0 + Math.cos(Math.PI * prog)), minLr);
      } else if (schedType === "constant") {
        lr = baseLr;
      } else {
        const decay = Math.pow(Math.max(1.0 - effIter / effMax, 0.0), polyPower);
        lr = Math.max(baseLr * decay, minLr);
      }
    }

    // 4.2 Consistency Weight (methods/mean_teacher.py:94-110)
    const consMax = Number(state.consistency) ?? 0.05;
    const rampup = Number(state.consistency_rampup) ?? 4000.0;
    const rampdownStart = Number(state.consistency_rampdown_start) ?? 7800;
    const minFactor = Number(state.consistency_min_factor) ?? 0.25;

    let baseW = consMax;
    let rampupUnit = "disabled";
    let rampupFactor = 1.0;
    if (rampup <= 0) {
      baseW = consMax;
      rampupFactor = 1.0;
    } else if (rampup > 100) {
      rampupUnit = "iterations (rampup > 100)";
      rampupFactor = sigmoidRampup(t, rampup);
      baseW = consMax * rampupFactor;
    } else {
      rampupUnit = `epochs (rampup <= 100, 1 ep = ${epochLen} iters)`;
      rampupFactor = sigmoidRampup(curEpoch, rampup);
      baseW = consMax * rampupFactor;
    }

    let decayFactor = 1.0;
    let consW = baseW;
    const inRampdown = rampdownStart > 0 && t > rampdownStart;
    if (inRampdown) {
      const progress = Math.min(Math.max((t - rampdownStart) / Math.max(maxIter - rampdownStart, 1), 0.0), 1.0);
      decayFactor = minFactor + (1.0 - minFactor) * 0.5 * (1.0 + Math.cos(Math.PI * progress));
      consW = baseW * decayFactor;
    }

    // 4.3 EMA Alpha (methods/ssl_utils.py:93-114)
    const targetAlpha = Number(state.ema_decay) || 0.999;
    const emaSched = (state.ema_schedule || "cosine").toLowerCase();
    let actualAlpha = targetAlpha;
    if (emaSched === "cosine" && maxIter > 0) {
      const minAlpha = 0.99;
      const progress = Math.min(Math.max(t / maxIter, 0.0), 1.0);
      actualAlpha = minAlpha + (targetAlpha - minAlpha) * 0.5 * (1.0 - Math.cos(Math.PI * progress));
    } else {
      actualAlpha = Math.min(1.0 - 1.0 / (t + 1), targetAlpha);
    }
    const studentUpdateRate = 1.0 - actualAlpha;
    const effectiveMemorySteps = 1.0 / Math.max(1e-6, 1.0 - actualAlpha);

    return {
      iter: t,
      epoch: curEpoch,
      lr,
      consW,
      rampupFactor,
      decayFactor,
      rampupUnit,
      inRampdown,
      actualAlpha,
      studentUpdateRate,
      effectiveMemorySteps,
    };
  }

  function generateScheduleSeries(state, epochLen, numPoints = 120) {
    const maxIter = Math.max(100, Number(state.max_iteration) || 12000);
    const points = [];
    for (let i = 0; i <= numPoints; i++) {
      const iter = Math.round((i / numPoints) * maxIter);
      points.push(evaluateSchedulesAtStep(iter, state, epochLen));
    }

    // Detect smart implementation diagnostics
    const alerts = [];
    const rampup = Number(state.consistency_rampup) ?? 4000;
    if (rampup > 0 && rampup <= 100) {
      alerts.push({
        level: "warning",
        code: "methods/mean_teacher.py:98-101",
        title: "Chế độ Ramp-up theo Epoch đang kích hoạt (consistency_rampup <= 100)",
        message: `Vì consistency_rampup = ${rampup} <= 100, hàm get_current_consistency_weight() truyền cur_epoch thay vì iter_num vào sigmoid_rampup(). Với epoch_len = ${epochLen} steps/epoch, quá trình ramp-up hoàn tất ở bước ${Math.round(rampup * epochLen)}!`,
      });
    } else if (rampup > 100) {
      alerts.push({
        level: "info",
        code: "methods/mean_teacher.py:98-99",
        title: "Chế độ Ramp-up theo Iteration đang kích hoạt (consistency_rampup > 100)",
        message: `Vì consistency_rampup = ${rampup} > 100, hàm truyền trực tiếp iter_num vào sigmoid_rampup(iter_num, ${rampup}). Đạt đỉnh w_cons = ${state.consistency} tại đúng iteration ${rampup}.`,
      });
    }

    if (state.ema_schedule === "cosine" && Math.abs(Number(state.ema_decay) - 0.99) < 1e-6) {
      alerts.push({
        level: "warning",
        code: "methods/ssl_utils.py:107-109",
        title: "Đường cong Cosine EMA bị suy biến thành hằng số α_t ≡ 0.99",
        message:
          "Trong update_ema_variables(), min_alpha được gán cứng bằng 0.99. Khi ema_decay = 0.99 (như trong preset bhsd_mt_nnunet_fold0.yaml), biểu thức (alpha - min_alpha) = 0 khiến α_t giữ nguyên ở 0.99 suốt quá trình train (cửa sổ nhớ cố định 100 bước).",
      });
    }

    if (Number(state.consistency) >= 0.5) {
      alerts.push({
        level: "caution",
        code: "methods/mean_teacher.py:177",
        title: "Trọng số Consistency cao (consistency >= 0.5)",
        message: `Với consistency = ${state.consistency}, thành phần unsupervised có thể lấn át supervised gradient ở các lớp hiếm (EDH, SAH) nếu Teacher dự đoán xác suất thấp, dễ dẫn đến confirmation bias.`,
      });
    }

    if (Number(state.confidence_thresh) > 0.0) {
      alerts.push({
        level: "info",
        code: "losses/consistency.py:72-76",
        title: "Chế độ CT Foreground-Preserving Confidence Mask đang bật",
        message: `Khi confidence_thresh = ${state.confidence_thresh} > 0, code áp dụng điều kiện kép: giữ lại voxel nếu (max_fg_prob >= min(${state.confidence_thresh}, 0.4)) HOẶC (bg_prob >= ${state.confidence_thresh}) để bảo vệ các tổn thương có độ tự tin trung bình.`,
      });
    }

    return { points, alerts };
  }

  // 5. Interactive Voxel-Level Consistency Loss & BHSD Metric Paradox Simulator
  function simulateConsistencyLoss(state, simParams) {
    // Simulated 6-class voxel distributions across a 3D CT patch (BG=97.5% voxels, 5 hemorrhage classes=2.5%)
    const bgWeight = Number(state.bg_consistency_weight) ?? 0.05;
    const confThresh = Number(state.confidence_thresh) ?? 0.0;
    const consType = (state.consistency_type || "hybrid").toLowerCase();

    // Class names and simulated per-class Student vs Teacher probability discrepancy & Dice overlap
    const gapScale = Number(simParams.pred_discrepancy) || 0.18;
    const fgOverlap = Number(simParams.fg_overlap) || 0.72;

    const classes = [
      { idx: 0, name: "Background (Class 0)", voxel_share: 0.975, weight: bgWeight, raw_mse: 0.042 * gapScale * 5, soft_dice: 0.985 },
      { idx: 1, name: "Epidural - EDH (Class 1)", voxel_share: 0.001, weight: 1.0, raw_mse: 0.065 * gapScale * 5, soft_dice: fgOverlap * 0.45 },
      { idx: 2, name: "Intraparenchymal - IPH (Class 2)", voxel_share: 0.009, weight: 1.0, raw_mse: 0.038 * gapScale * 5, soft_dice: Math.min(0.95, fgOverlap * 1.15) },
      { idx: 3, name: "Intraventricular - IVH (Class 3)", voxel_share: 0.005, weight: 1.0, raw_mse: 0.041 * gapScale * 5, soft_dice: Math.min(0.95, fgOverlap * 1.05) },
      { idx: 4, name: "Subarachnoid - SAH (Class 4)", voxel_share: 0.004, weight: 1.0, raw_mse: 0.058 * gapScale * 5, soft_dice: fgOverlap * 0.75 },
      { idx: 5, name: "Subdural - SDH (Class 5)", voxel_share: 0.006, weight: 1.0, raw_mse: 0.052 * gapScale * 5, soft_dice: fgOverlap * 0.85 },
    ];

    // Confidence mask retention calculation (losses/consistency.py:72-80)
    const fgEffThresh = confThresh > 0 ? Math.min(confThresh, 0.4) : 0.0;
    const bgRetainedPct = confThresh > 0 ? Math.max(15, Math.round((1.0 - Math.pow(confThresh, 2.2) * 0.45) * 100)) : 100;
    const fgRetainedPct = confThresh > 0 ? Math.max(10, Math.round((1.0 - fgEffThresh * 0.65) * 100)) : 100;
    const naiveFgRetainedPct = confThresh > 0 ? Math.max(2, Math.round((1.0 - Math.pow(confThresh, 1.1) * 0.92) * 100)) : 100;

    const totalWeight = classes.reduce((s, c) => s + c.weight, 0);
    const weightedMse = classes.reduce((s, c) => s + c.weight * c.raw_mse, 0) / (totalWeight + 1e-6);
    const unweightedMse = classes.reduce((s, c) => s + c.raw_mse, 0) / classes.length;

    const fgClasses = classes.slice(1);
    const meanFgDiceScore = fgClasses.reduce((s, c) => s + c.soft_dice, 0) / fgClasses.length;
    const softDiceLoss = 1.0 - meanFgDiceScore;
    const klLoss = weightedMse * 2.35;

    let activeConsLoss = 0.5 * weightedMse + 0.5 * softDiceLoss;
    if (consType === "dice") activeConsLoss = softDiceLoss;
    else if (consType === "weighted_mse" || consType === "mse") activeConsLoss = weightedMse;
    else if (consType === "kl") activeConsLoss = klLoss;
    else if (consType === "raw_mse") activeConsLoss = unweightedMse;

    // Gradient share per class in MSE term
    classes.forEach((c) => {
      c.weighted_mse_share_pct = ((c.weight * c.raw_mse) / ( weightedMse * totalWeight + 1e-9)) * 100;
    });

    return {
      classes,
      weightedMse,
      unweightedMse,
      meanFgDiceScore,
      softDiceLoss,
      klLoss,
      activeConsLoss,
      fgEffThresh,
      bgRetainedPct,
      fgRetainedPct,
      naiveFgRetainedPct,
    };
  }

  // 6. BHSD 38-Case Fold 0 Validation Metric Paradox Simulator
  function simulateMetricParadox(paradoxParams) {
    // Ground-truth Fold 0 Validation Cohort (38 cases total)
    const cohort = [
      { id: "EDH", name: "Epidural (EDH)", pos: 6, neg: 32 },
      { id: "IPH", name: "Intraparenchymal (IPH)", pos: 23, neg: 15 },
      { id: "IVH", name: "Intraventricular (IVH)", pos: 22, neg: 16 },
      { id: "SAH", name: "Subarachnoid (SAH)", pos: 20, neg: 18 },
      { id: "SDH", name: "Subdural (SDH)", pos: 13, neg: 25 },
    ];
    const totalCases = 38;
    const posDiceMultiplier = Number(paradoxParams.pos_dice_level) ?? 1.0; // 0.0 (empty model) to 1.5
    const fpRateScale = Number(paradoxParams.fp_case_rate) ?? 1.0; // 0.0 (no FP on neg cases) to 1.5

    // Reference values at iter 8,400 of run o6s4pex3
    const refPosDice = { EDH: 0.0, IPH: 0.5395, IVH: 0.4872, SAH: 0.2559, SDH: 0.3453 };
    const refFpNegCases = { EDH: 0, IPH: 13, IVH: 14, SAH: 18, SDH: 25 };

    const results = cohort.map((c) => {
      const basePosD = refPosDice[c.id];
      const simPosDice = Math.min(0.95, Math.max(0.0, basePosD * posDiceMultiplier));
      const fpNegCount = Math.min(c.neg, Math.max(0, Math.round(refFpNegCases[c.id] * fpRateScale)));
      const tnNegCount = c.neg - fpNegCount; // True negative cases get Dice = 1.0 in legacy metric!
      // Legacy case-mean Dice = (sum of pos case Dice + 1.0 * tnNegCount + 0.0 * fpNegCount) / 38
      const legacyCaseMean = (c.pos * simPosDice + tnNegCount * 1.0) / totalCases;
      const emptyModelLegacy = c.neg / totalCases;
      return {
        ...c,
        simPosDice,
        fpNegCount,
        tnNegCount,
        legacyCaseMean,
        emptyModelLegacy,
      };
    });

    const legacyMdice = results.reduce((s, r) => s + r.legacyCaseMean, 0) / results.length;
    const positiveMacro = results.reduce((s, r) => s + r.simPosDice, 0) / results.length;
    const emptyBaselineMdice = results.reduce((s, r) => s + r.emptyModelLegacy, 0) / results.length; // 0.55789

    return {
      results,
      legacyMdice,
      positiveMacro,
      emptyBaselineMdice,
    };
  }

  return {
    resolveBatchLayout,
    computeDsWeights,
    computeBackboneHierarchy,
    sigmoidRampup,
    evaluateSchedulesAtStep,
    generateScheduleSeries,
    simulateConsistencyLoss,
    simulateMetricParadox,
  };
})();

