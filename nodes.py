# ==========================================
# Custom Node 1: OrionEdit ApplyOrionEditAttention
# ==========================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths
from safetensors.torch import load_file

A_END, B_END, N_BLOCKS = 20, 40, 60


def _rope(q, k, pe):
    if pe is not None:
        from comfy.ldm.flux.math import apply_rope1
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    return q, k


def _sdpa(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)


def _segments(opts, n_img, mode):
    lengths = opts.get("reference_image_num_tokens")
    if not lengths:
        raise RuntimeError("OrionEdit requires Qwen ref_latents; reference_image_num_tokens is missing.")
    lengths = [int(x) for x in lengths]
    if any(x <= 0 for x in lengths):
        raise RuntimeError(f"Invalid reference token lengths: {lengths}")
    if mode == "edit":
        if len(lengths) != 2:
            raise RuntimeError(f"Edit mode requires [reference, source] token lengths; got {lengths}.")
        nr, ns = lengths
    else:
        nr, ns = sum(lengths), 0
    nz = n_img - nr - ns
    if nz <= 0:
        raise RuntimeError(f"Invalid stream sizes: image tokens={n_img}, reference/source={lengths}.")
    return slice(0, nz), slice(nz, nz + nr), slice(nz + nr, nz + nr + ns)


def _text_mask(x, text_mask, nt, total):
    if text_mask is None:
        return None
    m = torch.zeros((x.shape[0], 1, 1, total), device=x.device, dtype=x.dtype)
    m[..., :nt] = text_mask.reshape(x.shape[0], 1, 1, nt).to(device=x.device, dtype=x.dtype)
    return m


def _flow_mask(base, batch, total, nt, syn, ref, src, device, dtype, hard, beta):
    m = torch.zeros((batch, 1, total, total), device=device, dtype=dtype) if base is None else base.expand(batch, 1, total, total).clone()
    syn, ref, src = (slice(nt + s.start, nt + s.stop) for s in (syn, ref, src))
    txt = slice(0, nt)
    blocked = torch.finfo(dtype).min
    def block(q, k):
        if q.start == q.stop or k.start == k.stop:
            return
        if hard:
            m[:, :, q, k] = blocked
        else:
            m[:, :, q, k] -= beta
    block(ref, txt); block(ref, syn)
    if src.stop > src.start:
        block(ref, src)
    block(src, txt); block(src, syn)
    return m


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

    def forward(self, attn, x, text, text_mask, pe, opts, mode, q, k, v, tq, tk, tv):
        if next(self.parameters()).device != x.device or next(self.parameters()).dtype != x.dtype:
            self.to(device=x.device, dtype=x.dtype)
        b, ni, _ = x.shape
        nt, h = text.shape[1], attn.heads
        total = nt + ni
        base_mask = _text_mask(x, text_mask, nt, total)
        base = _sdpa(q, k, v, base_mask).transpose(1, 2).reshape(b, total, -1).to(q.dtype)
        txt_out, img_out = base[:, :nt], base[:, nt:].clone()
        syn, ref, src = _segments(opts, ni, mode)
        qm = q[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        km = k[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        vm = v[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        tkm = k[:, :, :nt].transpose(1, 2).reshape(b, nt, -1)
        tvm = tv.transpose(1, 2).reshape(b, nt, -1)
        p = self.proj
        qz, kz, vz = qm[:, syn], km[:, syn], vm[:, syn]
        kz1 = p.apply(kz, p.A1, p.B1, p.alpha); vz1 = p.apply(vz, p.A1, p.B1, p.alpha)
        kr = p.apply(km[:, ref], p.A2, p.B2, p.alpha)
        vr = p.apply(vm[:, ref], p.A2, p.B2, p.alpha)
        qr = p.apply(qz, p.A2, p.B2, p.alpha)
        if src.stop > src.start:
            ks = p.apply(km[:, src], p.A3, p.B3, p.alpha)
            vs = p.apply(vm[:, src], p.A3, p.B3, p.alpha)
            qs = p.apply(qz, p.A3, p.B3, p.alpha)
            bk, bv, bq = torch.cat((tkm, kz1, kr, ks), 1), torch.cat((tvm, vz1, vr, vs), 1), torch.cat((qr, qs), 1)
        else:
            qa = p.apply(qz, p.A1, p.B1, p.alpha)
            bk, bv, bq = torch.cat((tkm, kz1, kr), 1), torch.cat((tvm, vz1, vr), 1), torch.cat((qr, qa), 1)
        bq = bq.view(b, bq.shape[1], h, -1).transpose(1, 2).contiguous()
        bk = bk.view(b, bk.shape[1], h, -1).transpose(1, 2).contiguous()
        bv = bv.view(b, bv.shape[1], h, -1).transpose(1, 2).contiguous()
        branch = _sdpa(bq, bk, bv).transpose(1, 2).reshape(b, bq.shape[2], -1).to(bq.dtype)
        nz = syn.stop - syn.start
        w = torch.softmax(self.synthesis_fusion_logits, dim=0)
        img_out[:, syn] = w[0] * img_out[:, syn] + w[1] * branch[:, :nz] + w[2] * branch[:, nz:2*nz]
        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        return img_out, attn.to_add_out(txt_out)


def _attention(processor, attn, x, text, text_mask, pe, opts, mode, index):
    if text is None:
        raise RuntimeError("OrionEdit requires the Qwen text stream.")
    b, ni, _ = x.shape; nt = text.shape[1]; h = attn.heads
    iq = attn.to_q(x).view(b, ni, h, -1).transpose(1, 2).contiguous()
    ik = attn.to_k(x).view(b, ni, h, -1).transpose(1, 2).contiguous()
    iv = attn.to_v(x).view(b, ni, h, -1).transpose(1, 2)
    tq = attn.add_q_proj(text).view(b, nt, h, -1).transpose(1, 2).contiguous()
    tk = attn.add_k_proj(text).view(b, nt, h, -1).transpose(1, 2).contiguous()
    tv = attn.add_v_proj(text).view(b, nt, h, -1).transpose(1, 2)
    iq, ik, tq, tk = attn.norm_q(iq), attn.norm_k(ik), attn.norm_added_q(tq), attn.norm_added_k(tk)
    q, k, v = torch.cat((tq, iq), 2), torch.cat((tk, ik), 2), torch.cat((tv, iv), 2)
    from comfy.ldm.flux.math import apply_rope1
    if pe is not None:
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    if index >= B_END:
        return processor(attn, x, text, text_mask, pe, opts, mode, q, k, v, tq, tk, tv)
    syn, ref, src = _segments(opts, ni, mode)
    base = _text_mask(x, text_mask, nt, nt + ni)
    if index < A_END:
        mask = _flow_mask(base, b, nt + ni, nt, syn, ref, src, x.device, x.dtype, True, 0.0)
    else:
        beta = 8.0 - 6.0 * ((index - A_END) / (B_END - A_END - 1))
        mask = _flow_mask(base, b, nt + ni, nt, syn, ref, src, x.device, x.dtype, False, beta)
    out = _sdpa(q, k, v, mask).transpose(1, 2).reshape(b, nt + ni, -1).to(q.dtype)
    txt_out, img_out = out[:, :nt], out[:, nt:]
    img_out = attn.to_out[0](img_out)
    if len(attn.to_out) > 1:
        img_out = attn.to_out[1](img_out)
    return img_out, attn.to_add_out(txt_out)


class _BlockPatch:
    def __init__(self, block, processor, mode, index, previous=None):
        self.block, self.processor, self.mode, self.index, self.previous = block, processor, mode, index, previous

    def __call__(self, args, extra):
        attn = self.block.attn
        original = attn.forward
        def patched_forward(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                            attention_mask=None, image_rotary_emb=None, transformer_options=None, **kwargs):
            return _attention(self.processor, attn, hidden_states, encoder_hidden_states,
                              encoder_hidden_states_mask, image_rotary_emb, transformer_options or {},
                              self.mode, self.index)
        attn.forward = patched_forward
        try:
            return self.previous(args, extra) if self.previous is not None else extra["original_block"](args)
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
        return self.previous.models() if self.previous is not None and hasattr(self.previous, "models") else []


class ApplyOrionEditAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["edit", "fusion"], {"default": "edit"}),
            "processor_weights": ("STRING", {"default": "orionedit/pytorch_processor_weights.safetensors"}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "OrionEdit"

    def apply(self, model, mode, processor_weights):
        if not hasattr(model, "clone") or not hasattr(model, "set_model_patch_replace"):
            raise TypeError("Expected a ComfyUI MODEL patcher input.")
        path = processor_weights.strip()
        if not os.path.isabs(path):
            path = os.path.join(folder_paths.models_dir, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"OrionEdit processor weights not found: {path}")
        state = load_file(path, device="cpu")
        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None or len(blocks) != N_BLOCKS:
            raise RuntimeError(f"Expected the 60-block Qwen Image Edit transformer; found {len(blocks) if blocks is not None else 0} blocks.")
        replacements = patched.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for i, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None or not hasattr(attn, "to_q") or not hasattr(attn, "add_q_proj"):
                raise RuntimeError(f"Block {i} is not the expected Qwen double-stream attention block.")
            processor = None
            if i >= B_END:
                prefix = f"transformer.transformer_blocks.{i}.attn.processor."
                sd = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                required = {"proj.A1", "proj.A2", "proj.A3", "proj.B1", "proj.B2", "proj.B3", "synthesis_fusion_logits"}
                missing = required - set(sd)
                if missing:
                    raise RuntimeError(f"Missing Stage C weights for block {i}: {sorted(missing)}")
                rank, d = int(sd["proj.B1"].shape[0]), int(sd["proj.A1"].shape[0])
                model_d = int(getattr(attn, "inner_dim", attn.heads * attn.dim_head))
                if d != model_d:
                    raise RuntimeError(f"Stage C dimension mismatch at block {i}: checkpoint {d}, model {model_d}.")
                processor = _StageC(d, rank)
                processor.load_state_dict(sd, strict=True)
            key = ("double_block", i)
            previous = replacements.get(key)
            if isinstance(previous, _BlockPatch):
                raise RuntimeError("OrionEdit attention is already applied to this model.")
            patched.set_model_patch_replace(_BlockPatch(block, processor, mode, i, previous), "dit", "double_block", i)
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