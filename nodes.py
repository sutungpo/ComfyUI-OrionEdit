# ==========================================
# Custom Node 1: OrionEdit ApplyOrionEditAttention
# ==========================================
import os
import logging
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths
from safetensors.torch import load_file

logger = logging.getLogger("OrionEdit")

A_END, B_END, N_BLOCKS = 20, 40, 60
_REQUIRED_STAGE_C = {
    "proj.A1", "proj.A2", "proj.A3",
    "proj.B1", "proj.B2", "proj.B3",
    "synthesis_fusion_logits",
}


def _stage(index):
    if index < A_END:
        return "A"
    if index < B_END:
        return "B"
    return "C"


def _rope(q, k, pe):
    if pe is not None:
        from comfy.ldm.flux.math import apply_rope1
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    return q, k


def _sdpa(q, k, v, mask=None):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )


def _segments(opts, n_img, mode):
    lengths = opts.get("reference_image_num_tokens")
    if lengths is None or len(lengths) == 0:
        raise RuntimeError(
            "[OrionEdit] Qwen reference_image_num_tokens is missing. "
            "Check that the Qwen Image Edit conditioning path supplies ref_latents."
        )

    lengths = [int(x) for x in lengths]
    if any(x <= 0 for x in lengths):
        raise RuntimeError(f"[OrionEdit] Invalid reference token lengths: {lengths}")

    if mode == "edit":
        if len(lengths) != 2:
            raise RuntimeError(
                "[OrionEdit] Edit mode expects exactly [reference_tokens, source_tokens]; "
                f"received {lengths}. For multiple references, combine them into one "
                "reference canvas before Qwen encoding."
            )
        nr, ns = lengths
    elif mode == "fusion":
        nr, ns = sum(lengths), 0
    else:
        raise RuntimeError(f"[OrionEdit] Unsupported mode: {mode}")

    nz = n_img - nr - ns
    if nz <= 0:
        raise RuntimeError(
            f"[OrionEdit] Invalid image stream sizes: image_tokens={n_img}, "
            f"reference/source token lengths={lengths}."
        )

    return (
        slice(0, nz),
        slice(nz, nz + nr),
        slice(nz + nr, nz + nr + ns),
    )


def _text_mask(x, text_mask, nt, total):
    if text_mask is None:
        return None
    m = torch.zeros(
        (x.shape[0], 1, 1, total), device=x.device, dtype=x.dtype
    )
    m[..., :nt] = text_mask.reshape(x.shape[0], 1, 1, nt).to(
        device=x.device, dtype=x.dtype
    )
    return m


def _flow_mask(base, batch, total, nt, syn, ref, src,
               device, dtype, hard, beta):
    if base is None:
        m = torch.zeros((batch, 1, total, total), device=device, dtype=dtype)
    else:
        m = base.expand(batch, 1, total, total).clone()

    syn, ref, src = (
        slice(nt + s.start, nt + s.stop) for s in (syn, ref, src)
    )
    txt = slice(0, nt)
    blocked = torch.finfo(dtype).min

    def block(q, k):
        if q.start == q.stop or k.start == k.stop:
            return
        if hard:
            m[:, :, q, k] = blocked
        else:
            m[:, :, q, k] -= beta

    # OrionEdit reverse-causal image-stream restrictions.
    block(ref, txt)
    block(ref, syn)
    if src.stop > src.start:
        block(ref, src)
    block(src, txt)
    block(src, syn)
    return m


def _debug_tensor(label, tensor):
    if not logger.isEnabledFor(logging.DEBUG) or tensor is None:
        return
    finite = bool(torch.isfinite(tensor).all().item())
    logger.debug(
        "[OrionEdit] %s: shape=%s dtype=%s device=%s finite=%s",
        label, tuple(tensor.shape), tensor.dtype, tensor.device, finite
    )
    if not finite:
        raise RuntimeError(f"[OrionEdit] Non-finite values detected in {label}.")


class _Projector(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A1 = nn.Parameter(torch.empty(d, r), requires_grad=False)
        self.A2 = nn.Parameter(torch.empty(d, r), requires_grad=False)
        self.A3 = nn.Parameter(torch.empty(d, r), requires_grad=False)
        self.B1 = nn.Parameter(torch.zeros(r, r))
        self.B2 = nn.Parameter(torch.zeros(r, r))
        self.B3 = nn.Parameter(torch.zeros(r, r))
        self.alpha = 1.0

    @staticmethod
    def apply(x, a, b, alpha):
        a = a.to(device=x.device, dtype=x.dtype)
        b = b.to(device=x.device, dtype=x.dtype)
        return x + alpha * ((x @ a) @ b @ a.T)


class _StageC(nn.Module):
    def __init__(self, d, rank):
        super().__init__()
        self.proj = _Projector(d, rank)
        self.synthesis_fusion_logits = nn.Parameter(torch.zeros(3))

    def forward(self, attn, x, text, text_mask, pe, opts, mode,
                q, k, v, tq, tk, tv):
        first_param = next(self.parameters())
        if first_param.device != x.device or first_param.dtype != x.dtype:
            self.to(device=x.device, dtype=x.dtype)

        b, ni, _ = x.shape
        nt, h = text.shape[1], attn.heads
        total = nt + ni
        base_mask = _text_mask(x, text_mask, nt, total)
        base = _sdpa(q, k, v, base_mask).transpose(1, 2).reshape(
            b, total, -1
        ).to(q.dtype)

        txt_out, img_out = base[:, :nt], base[:, nt:].clone()
        syn, ref, src = _segments(opts, ni, mode)

        qm = q[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        km = k[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        vm = v[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        tkm = k[:, :, :nt].transpose(1, 2).reshape(b, nt, -1)
        tvm = tv.transpose(1, 2).reshape(b, nt, -1)

        p = self.proj
        qz, kz, vz = qm[:, syn], km[:, syn], vm[:, syn]
        kz1 = p.apply(kz, p.A1, p.B1, p.alpha)
        vz1 = p.apply(vz, p.A1, p.B1, p.alpha)
        kr = p.apply(km[:, ref], p.A2, p.B2, p.alpha)
        vr = p.apply(vm[:, ref], p.A2, p.B2, p.alpha)
        qr = p.apply(qz, p.A2, p.B2, p.alpha)

        if src.stop > src.start:
            ks = p.apply(km[:, src], p.A3, p.B3, p.alpha)
            vs = p.apply(vm[:, src], p.A3, p.B3, p.alpha)
            qs = p.apply(qz, p.A3, p.B3, p.alpha)
            bk = torch.cat((tkm, kz1, kr, ks), 1)
            bv = torch.cat((tvm, vz1, vr, vs), 1)
            bq = torch.cat((qr, qs), 1)
        else:
            qa = p.apply(qz, p.A1, p.B1, p.alpha)
            bk = torch.cat((tkm, kz1, kr), 1)
            bv = torch.cat((tvm, vz1, vr), 1)
            bq = torch.cat((qr, qa), 1)

        bq = bq.view(b, bq.shape[1], h, -1).transpose(1, 2).contiguous()
        bk = bk.view(b, bk.shape[1], h, -1).transpose(1, 2).contiguous()
        bv = bv.view(b, bv.shape[1], h, -1).transpose(1, 2).contiguous()

        _debug_tensor("Stage C projected queries", bq)
        _debug_tensor("Stage C projected keys", bk)
        _debug_tensor("Stage C projected values", bv)

        branch = _sdpa(bq, bk, bv).transpose(1, 2).reshape(
            b, bq.shape[2], -1
        ).to(bq.dtype)

        nz = syn.stop - syn.start
        weights = torch.softmax(self.synthesis_fusion_logits, dim=0)
        img_out[:, syn] = (
            weights[0] * img_out[:, syn]
            + weights[1] * branch[:, :nz]
            + weights[2] * branch[:, nz:2 * nz]
        )

        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)

        _debug_tensor("Stage C image output", img_out)
        return img_out, attn.to_add_out(txt_out)


def _attention(processor, attn, x, text, text_mask, pe, opts, mode, index):
    if text is None:
        raise RuntimeError("[OrionEdit] Qwen text stream is missing.")

    b, ni, _ = x.shape
    nt, h = text.shape[1], attn.heads

    iq = attn.to_q(x).view(b, ni, h, -1).transpose(1, 2).contiguous()
    ik = attn.to_k(x).view(b, ni, h, -1).transpose(1, 2).contiguous()
    iv = attn.to_v(x).view(b, ni, h, -1).transpose(1, 2)
    tq = attn.add_q_proj(text).view(b, nt, h, -1).transpose(1, 2).contiguous()
    tk = attn.add_k_proj(text).view(b, nt, h, -1).transpose(1, 2).contiguous()
    tv = attn.add_v_proj(text).view(b, nt, h, -1).transpose(1, 2)

    iq, ik = attn.norm_q(iq), attn.norm_k(ik)
    tq, tk = attn.norm_added_q(tq), attn.norm_added_k(tk)

    q = torch.cat((tq, iq), 2)
    k = torch.cat((tk, ik), 2)
    v = torch.cat((tv, iv), 2)

    if pe is not None:
        from comfy.ldm.flux.math import apply_rope1
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)

    if logger.isEnabledFor(logging.DEBUG):
        _debug_tensor(f"Block {index} Q", q)
        _debug_tensor(f"Block {index} K", k)
        _debug_tensor(f"Block {index} V", v)

    if index >= B_END:
        return processor(
            attn, x, text, text_mask, pe, opts, mode,
            q, k, v, tq, tk, tv
        )

    syn, ref, src = _segments(opts, ni, mode)
    base = _text_mask(x, text_mask, nt, nt + ni)

    if index < A_END:
        mask = _flow_mask(
            base, b, nt + ni, nt, syn, ref, src,
            x.device, x.dtype, True, 0.0
        )
    else:
        beta = 8.0 - 6.0 * (
            (index - A_END) / (B_END - A_END - 1)
        )
        mask = _flow_mask(
            base, b, nt + ni, nt, syn, ref, src,
            x.device, x.dtype, False, beta
        )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[OrionEdit] Block %d mask: shape=%s dtype=%s device=%s",
            index, tuple(mask.shape), mask.dtype, mask.device
        )
        _debug_tensor(f"Block {index} attention mask", mask)

    out = _sdpa(q, k, v, mask).transpose(1, 2).reshape(
        b, nt + ni, -1
    ).to(q.dtype)

    txt_out, img_out = out[:, :nt], out[:, nt:]
    img_out = attn.to_out[0](img_out)
    if len(attn.to_out) > 1:
        img_out = attn.to_out[1](img_out)

    _debug_tensor(f"Block {index} image output", img_out)
    return img_out, attn.to_add_out(txt_out)


class _BlockPatch:
    def __init__(self, block, processor, mode, index, previous=None):
        self.block = block
        self.processor = processor
        self.mode = mode
        self.index = index
        self.previous = previous
        self._logged_runtime = False
        self._log_lock = threading.Lock()

    def _log_first_execution(self, hidden_states, encoder_hidden_states, opts):
        with self._log_lock:
            if self._logged_runtime:
                return
            self._logged_runtime = True

        ni = hidden_states.shape[1]
        nt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        lengths = opts.get("reference_image_num_tokens")
        syn, ref, src = _segments(opts, ni, self.mode)

        if self.index < A_END:
            detail = "hard reverse-causal mask"
        elif self.index < B_END:
            beta = 8.0 - 6.0 * (
                (self.index - A_END) / (B_END - A_END - 1)
            )
            detail = f"soft reverse-causal mask; beta={beta:.4f}"
        else:
            detail = "decoupled attention with Stage C projector"

        logger.info(
            "[OrionEdit] Runtime Stage %s block %d: EXECUTED (%s)",
            _stage(self.index), self.index, detail
        )
        logger.info(
            "[OrionEdit] Block %d streams: mode=%s, "
            "reference_image_num_tokens=%s, text_tokens=%d, image_tokens=%d, "
            "syn=[%d:%d] (%d), ref=[%d:%d] (%d), src=[%d:%d] (%d)",
            self.index, self.mode, lengths, nt, ni,
            syn.start, syn.stop, syn.stop - syn.start,
            ref.start, ref.stop, ref.stop - ref.start,
            src.start, src.stop, src.stop - src.start
        )
        logger.info(
            "[OrionEdit] Block %d hidden states: shape=%s dtype=%s device=%s",
            self.index, tuple(hidden_states.shape),
            hidden_states.dtype, hidden_states.device
        )

    def __call__(self, args, extra):
        attn = self.block.attn
        original = attn.forward

        def patched_forward(
            hidden_states,
            encoder_hidden_states=None,
            encoder_hidden_states_mask=None,
            attention_mask=None,
            image_rotary_emb=None,
            transformer_options=None,
            **kwargs
        ):
            opts = transformer_options or {}
            self._log_first_execution(hidden_states, encoder_hidden_states, opts)
            return _attention(
                self.processor, attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, image_rotary_emb, opts,
                self.mode, self.index
            )

        attn.forward = patched_forward
        try:
            return (
                self.previous(args, extra)
                if self.previous is not None
                else extra["original_block"](args)
            )
        finally:
            attn.forward = original

    def to(self, device_or_dtype):
        if isinstance(self.processor, nn.Module):
            self.processor.to(device_or_dtype)
        if self.previous is not None and hasattr(self.previous, "to"):
            self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        if self.previous is not None and hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        return (
            self.previous.models()
            if self.previous is not None and hasattr(self.previous, "models")
            else []
        )


class ApplyOrionEditAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["edit", "fusion"], {"default": "edit"}),
                "processor_weights": (
                    "STRING",
                    {"default": "orionedit/pytorch_processor_weights.safetensors"}
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "OrionEdit"

    def apply(self, model, mode, processor_weights):
        if not hasattr(model, "clone") or not hasattr(model, "set_model_patch_replace"):
            raise TypeError("[OrionEdit] Expected a ComfyUI MODEL patcher input.")

        path = processor_weights.strip()
        if not os.path.isabs(path):
            path = os.path.join(folder_paths.models_dir, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[OrionEdit] Processor weights not found: {path}"
            )

        logger.info("[OrionEdit] Loading processor weights: %s", path)
        state = load_file(path, device="cpu")

        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "transformer_blocks", None)
        block_count = len(blocks) if blocks is not None else 0
        if block_count != N_BLOCKS:
            raise RuntimeError(
                f"[OrionEdit] Expected {N_BLOCKS} Qwen transformer blocks; "
                f"found {block_count}."
            )

        replacements = (
            patched.model_options.get("transformer_options", {})
            .get("patches_replace", {}).get("dit", {})
        )

        stage_c_count = 0
        for i, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if (
                attn is None
                or not hasattr(attn, "to_q")
                or not hasattr(attn, "add_q_proj")
            ):
                raise RuntimeError(
                    f"[OrionEdit] Block {i} is not the expected Qwen "
                    "double-stream attention block."
                )

            if i == 0:
                logger.info(
                    "[OrionEdit] Model=%s; blocks=%d; mode=%s",
                    type(diffusion_model).__name__, block_count, mode
                )
                logger.info(
                    "[OrionEdit] INT8 projection modules: "
                    "to_q=%s, to_k=%s, to_v=%s, to_out=%s",
                    type(attn.to_q).__name__, type(attn.to_k).__name__,
                    type(attn.to_v).__name__, type(attn.to_out[0]).__name__
                )

            processor = None
            if i >= B_END:
                prefix = f"transformer.transformer_blocks.{i}.attn.processor."
                sd = {
                    k[len(prefix):]: v
                    for k, v in state.items()
                    if k.startswith(prefix)
                }
                missing = _REQUIRED_STAGE_C - set(sd)
                if missing:
                    raise RuntimeError(
                        f"[OrionEdit] Missing Stage C weights for block {i}: "
                        f"{sorted(missing)}"
                    )

                rank = int(sd["proj.B1"].shape[0])
                d = int(sd["proj.A1"].shape[0])
                model_d = int(
                    getattr(attn, "inner_dim", attn.heads * attn.dim_head)
                )
                if d != model_d:
                    raise RuntimeError(
                        f"[OrionEdit] Stage C dimension mismatch at block {i}: "
                        f"checkpoint={d}, model={model_d}."
                    )

                for key in _REQUIRED_STAGE_C:
                    if not torch.isfinite(sd[key]).all():
                        raise RuntimeError(
                            f"[OrionEdit] Non-finite processor weight "
                            f"{key} in block {i}."
                        )

                processor = _StageC(d, rank)
                processor.load_state_dict(sd, strict=True)
                stage_c_count += 1

            key = ("double_block", i)
            previous = replacements.get(key)
            if isinstance(previous, _BlockPatch):
                raise RuntimeError(
                    "[OrionEdit] OrionEdit attention is already applied "
                    "to this model."
                )

            patch = _BlockPatch(block, processor, mode, i, previous)
            patched.set_model_patch_replace(
                patch, "dit", "double_block", i
            )

        if stage_c_count != 20:
            raise RuntimeError(
                f"[OrionEdit] Expected 20 Stage C processors; loaded {stage_c_count}."
            )

        logger.info(
            "[OrionEdit] Patch installation complete: "
            "Stage A=20 blocks (0-19), Stage B=20 blocks (20-39), "
            "Stage C=20 blocks (40-59); Stage C processors loaded=%d.",
            stage_c_count
        )
        logger.info(
            "[OrionEdit] Runtime logs are emitted once per block. "
            "Set logging level DEBUG for tensor/mask diagnostics and "
            "NaN/Inf checks."
        )
        return (patched,)

# ==========================================
# Custom Node 2: OrionEdit helper
# ==========================================
import inspect
import torch
import torch.nn.functional as F

class QwenAttentionSpyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "spy_attention"
    CATEGORY = "OrionEdit"

    def spy_attention(self, model):
        print("\n" + "=" * 80)
        print("[AttentionSpy] STATIC ATTENTION INSPECTION")
        print("=" * 80)

        # 1. Safely unwrap ComfyUI ModelPatcher / ModelPatcherDynamic
        print(f"[*] Outer Model Wrapper: {model.__class__.__name__}")
        
        # Traverse down through .model and .diffusion_model
        target = model
        if hasattr(target, "model"):
            target = target.model
            print(f"[*] Unwrapped inner model: {target.__class__.__name__}")
            
        if hasattr(target, "diffusion_model"):
            diff_model = target.diffusion_model
            print(f"[*] Found diffusion_model: {diff_model.__class__.__module__}.{diff_model.__class__.__name__}")
        else:
            diff_model = target
            print(f"[*] Using root model: {diff_model.__class__.__module__}.{diff_model.__class__.__name__}")

        print(f"[*] Defined in File: {inspect.getfile(diff_model.__class__)}")

        # 2. Locate Transformer Blocks
        blocks = None
        blocks_attr_name = None
        # Check both diff_model and outer target in case blocks are top-level
        for candidate in [diff_model, getattr(model, "model", None), model]:
            if candidate is None:
                continue
            for attr in ["transformer_blocks", "double_blocks", "blocks", "layers"]:
                if hasattr(candidate, attr):
                    blocks = getattr(candidate, attr)
                    blocks_attr_name = attr
                    diff_model = candidate
                    break
            if blocks is not None:
                break

        if blocks is None:
            print("[X] Could not locate transformer blocks. Available attributes on model:")
            print(dir(diff_model))
            return (model,)

        print(f"[*] Found {len(blocks)} transformer blocks under attribute: '{blocks_attr_name}'")
        sample_block = blocks[0]
        print(f"[*] Block 0 Class: {sample_block.__class__.__name__}")

        # 3. Locate Attention Submodule
        attn_module = None
        attn_attr = None
        for attr in ["attn", "attn1", "self_attn", "joint_attn"]:
            if hasattr(sample_block, attr):
                attn_module = getattr(sample_block, attr)
                attn_attr = attr
                break

        if attn_module is None:
            print("[X] Could not locate attention submodule in block. Available block children:")
            for n, _ in sample_block.named_children():
                print(f"    - {n}")
            return (model,)

        # 4. Log Module Identity & Source File
        attn_cls = attn_module.__class__
        print(f"[*] Attention Attribute: block.{attn_attr}")
        print(f"[*] Attention Class:     {attn_cls.__module__}.{attn_cls.__name__}")
        print(f"[*] Defined in File:     {inspect.getfile(attn_cls)}")

        # 5. Check for Diffusers-style Processor vs Native Forward
        if hasattr(attn_module, "processor"):
            proc = attn_module.processor
            print(f"[*] Execution Delegate:  Uses .processor -> {proc.__class__.__module__}.{proc.__class__.__name__}")
            print(f"[*] Processor File:      {inspect.getfile(proc.__class__)}")
            exec_target = proc.forward if hasattr(proc, "forward") else proc.__call__
        else:
            print("[*] Execution Delegate:  Native forward method (No .processor attribute)")
            exec_target = attn_module.forward

        # 6. Print Source Code of the Attention Implementation
        print("\n" + "-" * 80)
        print("[AttentionSpy] SOURCE CODE OF THE EXECUTING ATTENTION FUNCTION:")
        print("-" * 80)
        try:
            source_lines, start_line = inspect.getsourcelines(exec_target)
            print(f"// File: {inspect.getfile(exec_target)} (Lines {start_line}-{start_line + len(source_lines)})")
            for line in source_lines[:50]:  # print first 50 lines
                print(line, end="")
            if len(source_lines) > 50:
                print("\n... [truncated for readability] ...")
        except Exception as e:
            print(f"Could not retrieve source code: {e}")
        print("-" * 80 + "\n")

        # 7. Runtime Trap: Intercept the exact SDPA computation
        active_block_idx = {"current": None}
        original_sdpa = F.scaled_dot_product_attention

        def sdpa_interceptor(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            if active_block_idx["current"] is not None:
                b_idx = active_block_idx["current"]
                active_block_idx["current"] = None  # log only once per step

                # Caller frame analysis
                stack = inspect.stack()
                caller_frame = stack[1]
                
                print("=" * 80)
                print(f"[AttentionSpy] RUNTIME EXECUTION DISPATCH (Block #{b_idx})")
                print("=" * 80)
                print(f"[*] Core Math Function:   torch.nn.functional.scaled_dot_product_attention")
                print(f"[*] Called from Function: {caller_frame.function}")
                print(f"[*] Caller File:          {caller_frame.filename} (Line {caller_frame.lineno})")
                print(f"[*] Query Shape:          {list(query.shape)} (dtype={query.dtype}, device={query.device})")
                print(f"[*] Key Shape:            {list(key.shape)}")
                print(f"[*] Value Shape:          {list(value.shape)}")
                print(f"[*] Attention Mask:       {'Present ' + str(list(attn_mask.shape)) if attn_mask is not None else 'None (Fully Bidirectional)'}")
                print(f"[*] Dropout / Causal:     dropout={dropout_p}, is_causal={is_causal}")

                if torch.cuda.is_available():
                    backends = {
                        "FlashAttention": torch.backends.cuda.flash_sdp_enabled(),
                        "MemEfficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                        "Math": torch.backends.cuda.math_sdp_enabled(),
                    }
                    print(f"[*] Enabled SDPA Backends: {backends}")
                print("=" * 80 + "\n")

            if scale is not None:
                return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        F.scaled_dot_product_attention = sdpa_interceptor

        # Attach pre-hook
        def pre_hook(idx):
            def _fn(module, args, kwargs):
                active_block_idx["current"] = idx
            return _fn

        for target_idx in [0, 40]:
            if target_idx < len(blocks):
                getattr(blocks[target_idx], attn_attr).register_forward_pre_hook(pre_hook(target_idx), with_kwargs=True)

        return (model,)

# ==========================================
# Node Registration Mapping
# ==========================================

NODE_CLASS_MAPPINGS = {
    "ApplyOrionEditAttention": ApplyOrionEditAttention,
    "QwenAttentionSpyNode": QwenAttentionSpyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyOrionEditAttention": "Apply OrionEdit Attention (Qwen INT8)",
    "QwenAttentionSpyNode": "Debug Qwen Attention Spy",
}