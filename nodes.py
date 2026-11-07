# ==========================================
# Custom Node 1: OrionEdit ApplyOrionEditAttention
# ==========================================
"""OrionEdit attention adapter for ComfyUI Qwen Image Edit 2511.

This node patches only the attention execution of the existing Qwen double-stream
blocks. The original projection modules (including INT8/ConvRot implementations)
remain owned and executed by ComfyUI.

Install this folder under ComfyUI/custom_nodes/ComfyUI_OrionEdit/.
"""

from __future__ import annotations

import os
import threading
import types
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention_masked
    from comfy.ldm.flux.math import apply_rope1
except Exception as exc:  # pragma: no cover - import error is reported by ComfyUI
    raise ImportError(
        "ComfyUI_OrionEdit must be installed inside ComfyUI and requires safetensors."
    ) from exc


_STAGE_A_END = 20
_STAGE_B_END = 40
_TOTAL_BLOCKS = 60
_PATCH_LOCK = threading.RLock()


def _to_additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Comfy's Qwen path normally converts integer masks before attention."""
    if mask.dtype == torch.bool or not torch.is_floating_point(mask):
        return (mask.to(dtype) - 1.0) * torch.finfo(dtype).max
    return mask.to(dtype)


def _get_stream_slices(
    transformer_options: Dict[str, Any],
    image_length: int,
    mode: str,
) -> Dict[str, slice]:
    """Build image-token slices from Comfy Qwen's reference_image_num_tokens."""
    counts = transformer_options.get("reference_image_num_tokens")
    if not counts:
        raise RuntimeError(
            "OrionEdit needs Qwen ref_latents. No 'reference_image_num_tokens' "
            "was found in transformer_options. Connect reference/source latent inputs."
        )
    counts = [int(x) for x in counts]
    if any(x < 0 for x in counts):
        raise ValueError(f"Invalid reference token lengths: {counts}")

    ref_total = sum(counts)
    syn_len = image_length - ref_total
    if syn_len <= 0:
        raise RuntimeError(
            f"Invalid Qwen stream lengths: image tokens={image_length}, reference tokens={counts}."
        )

    if mode == "edit":
        if len(counts) != 2:
            raise RuntimeError(
                "Edit mode expects exactly two ref_latents in this order: "
                "[composited reference, source image]. For multiple references, "
                "composite them into one reference latent first."
            )
        ref_len, src_len = counts
    elif mode == "fusion":
        if len(counts) < 2:
            raise RuntimeError("Fusion mode expects at least two reference latents.")
        ref_len, src_len = ref_total, 0
    else:
        raise ValueError(f"Unknown OrionEdit mode: {mode}")

    if syn_len + ref_len + src_len != image_length:
        raise RuntimeError(
            "Orion stream lengths do not match Qwen's image-token sequence: "
            f"syn={syn_len}, ref={ref_len}, src={src_len}, total={image_length}."
        )

    return {
        "syn": slice(0, syn_len),
        "ref": slice(syn_len, syn_len + ref_len),
        "src": slice(syn_len + ref_len, image_length),
    }


def _joint_image_slices(seq_txt: int, seg: Dict[str, slice]) -> Tuple[slice, slice, slice]:
    return tuple(
        slice(seq_txt + seg[name].start, seq_txt + seg[name].stop)
        for name in ("syn", "ref", "src")
    )  # type: ignore[return-value]


def _make_joint_mask(
    hidden_states: torch.Tensor,
    encoder_hidden_states_mask: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    seq_txt: int,
    seq_img: int,
) -> Optional[torch.Tensor]:
    """Create a broadcastable additive mask, including text padding if present."""
    batch, device, dtype = hidden_states.shape[0], hidden_states.device, hidden_states.dtype
    total = seq_txt + seq_img
    mask = None

    if encoder_hidden_states_mask is not None:
        txt_mask = _to_additive_mask(encoder_hidden_states_mask, dtype)
        if txt_mask.ndim == 2:
            txt_mask = txt_mask[:, None, None, :]
        elif txt_mask.ndim == 3:
            txt_mask = txt_mask[:, :, None, :]
        key_mask = torch.zeros((batch, 1, 1, total), device=device, dtype=dtype)
        key_mask[..., :seq_txt] = txt_mask.reshape(batch, 1, 1, seq_txt)
        mask = key_mask

    if attention_mask is not None:
        extra = _to_additive_mask(attention_mask, dtype).to(device)
        if extra.ndim == 2:
            extra = extra[:, None, None, :]
        elif extra.ndim == 3:
            extra = extra[:, None, :, :]
        mask = extra if mask is None else mask + extra

    return mask


def _apply_information_flow_mask(
    mask: torch.Tensor,
    txt: slice,
    syn: slice,
    ref: slice,
    src: slice,
    *,
    hard: bool,
    soft_beta: float,
) -> None:
    """Orion reverse-causal masking on [text | syn | ref | src]."""
    blocked_value = torch.finfo(mask.dtype).min

    def block(q: slice, k: slice) -> None:
        if q.start == q.stop or k.start == k.stop:
            return
        if hard:
            mask[..., q, k] = blocked_value
        else:
            mask[..., q, k] -= soft_beta

    # Reference queries see reference keys only.
    block(ref, txt)
    block(ref, syn)
    if src.stop > src.start:
        block(ref, src)

    # Source queries see reference and source keys, not text or synthesis.
    block(src, txt)
    block(src, syn)
    # Text and synthesis queries are intentionally unrestricted.


def _run_attention(
    attn: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    transformer_options: Dict[str, Any],
) -> torch.Tensor:
    q_box = AttentionTensorContainer(q)
    k_box = AttentionTensorContainer(k)
    v_box = AttentionTensorContainer(v)
    out = optimized_attention_masked(
        q_box,
        k_box,
        v_box,
        attn.heads,
        mask,
        transformer_options=transformer_options,
        skip_reshape=True,
        preferred_attention=getattr(attn, "comfy_attention", None),
    )
    return out


class TriOrthogonalLowRankProjector(nn.Module):
    """Checkpoint-compatible Orion tri-subspace low-rank projector."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        # Values are overwritten by the official processor checkpoint.
        self.A1 = nn.Parameter(torch.empty(dim, rank), requires_grad=False)
        self.A2 = nn.Parameter(torch.empty(dim, rank), requires_grad=False)
        self.A3 = nn.Parameter(torch.empty(dim, rank), requires_grad=False)
        self.B1 = nn.Parameter(torch.zeros(rank, rank))
        self.B2 = nn.Parameter(torch.zeros(rank, rank))
        self.B3 = nn.Parameter(torch.zeros(rank, rank))
        self.alpha = 1.0

    @staticmethod
    def apply(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, alpha: float) -> torch.Tensor:
        # ComfyUI may move patches by device without changing their dtype. Match
        # projector tensors to the activation dtype/device before matmul.
        A = A.to(device=x.device, dtype=x.dtype)
        B = B.to(device=x.device, dtype=x.dtype)
        return x + alpha * ((x @ A) @ B @ A.transpose(0, 1))


class OrionAttentionProcessor(nn.Module):
    def __init__(self, stage: str, dim: int, rank: int = 256, soft_beta: float = 6.0):
        super().__init__()
        self.stage = stage
        self.soft_beta = float(soft_beta)
        self.proj = TriOrthogonalLowRankProjector(dim, rank) if stage == "C" else None
        if stage == "C":
            self.synthesis_fusion_logits = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _project(attn, hidden_states, encoder_hidden_states):
        b, li, _ = hidden_states.shape
        lt = encoder_hidden_states.shape[1]
        heads = attn.heads

        # Match ComfyUI's native Qwen layout: [batch, heads, sequence, head_dim].
        iq = attn.to_q(hidden_states).unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()
        ik = attn.to_k(hidden_states).unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()
        iv = attn.to_v(hidden_states).unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()
        tq = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()
        tk = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()
        tv = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (heads, -1)).transpose(1, 2).contiguous()

        if getattr(attn, "norm_q", None) is not None:
            iq = attn.norm_q(iq)
        if getattr(attn, "norm_k", None) is not None:
            ik = attn.norm_k(ik)
        if getattr(attn, "norm_added_q", None) is not None:
            tq = attn.norm_added_q(tq)
        if getattr(attn, "norm_added_k", None) is not None:
            tk = attn.norm_added_k(tk)

        # ComfyUI's native Qwen path applies RoPE to the concatenated [text|image] tensors.
        q = torch.cat([tq, iq], dim=2)
        k = torch.cat([tk, ik], dim=2)
        v = torch.cat([tv, iv], dim=2)
        return iq, ik, iv, tq, tk, tv, q, k, v, lt, li

    def forward(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[Dict[str, Any]] = None,
        mode: str = "edit",
        block_index: int = 0,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            raise RuntimeError("OrionEdit requires Qwen's text encoder_hidden_states stream.")
        options = transformer_options or {}
        iq, ik, iv, tq, tk, tv, q, k, v, seq_txt, seq_img = self._project(
            attn, hidden_states, encoder_hidden_states
        )

        if image_rotary_emb is not None:
            q = apply_rope1(q, image_rotary_emb)
            k = apply_rope1(k, image_rotary_emb)

        base_mask = _make_joint_mask(
            hidden_states, encoder_hidden_states_mask, attention_mask, seq_txt, seq_img
        )

        seg = _get_stream_slices(options, seq_img, mode)
        syn_j, ref_j, src_j = _joint_image_slices(seq_txt, seg)
        txt_j = slice(0, seq_txt)

        if self.stage in ("A", "B"):
            total_len = seq_txt + seq_img
            if base_mask is None:
                base_mask = torch.zeros(
                    (hidden_states.shape[0], 1, total_len, total_len),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            elif base_mask.shape[-2] == 1:
                # Expand the broadcastable text-padding mask before applying
                # query-specific Orion restrictions.
                base_mask = base_mask.expand(-1, -1, total_len, -1).clone()
            _apply_information_flow_mask(
                base_mask,
                txt_j,
                syn_j,
                ref_j,
                src_j,
                hard=(self.stage == "A"),
                soft_beta=self.soft_beta,
            )

        joint_out = _run_attention(attn, q, k, v, base_mask, options)
        joint_out = joint_out.to(q.dtype)
        txt_out = joint_out[:, :seq_txt, :]
        img_out = joint_out[:, seq_txt:, :]

        if self.stage == "C":
            assert self.proj is not None
            # Q/K/V in [B, sequence, hidden_dim], matching the official processor.
            # Use post-RoPE Q/K for the Stage-C branches, as the official
            # Diffusers processor does; V remains unrotated.
            img_q = q[:, :, seq_txt:, :].transpose(1, 2).reshape(hidden_states.shape[0], seq_img, -1)
            img_k = k[:, :, seq_txt:, :].transpose(1, 2).reshape(hidden_states.shape[0], seq_img, -1)
            img_v = iv.transpose(1, 2).reshape(hidden_states.shape[0], seq_img, -1)
            txt_k = k[:, :, :seq_txt, :].transpose(1, 2).reshape(hidden_states.shape[0], seq_txt, -1)
            txt_v = tv.transpose(1, 2).reshape(hidden_states.shape[0], seq_txt, -1)

            syn, ref, src = seg["syn"], seg["ref"], seg["src"]
            syn_len = syn.stop - syn.start
            base_syn = img_out[:, syn, :].clone()
            q_syn = img_q[:, syn, :]
            k_syn, v_syn = img_k[:, syn, :], img_v[:, syn, :]
            k_ref, v_ref = img_k[:, ref, :], img_v[:, ref, :]
            p = self.proj

            k_syn_p = p.apply(k_syn, p.A1, p.B1, p.alpha)
            v_syn_p = p.apply(v_syn, p.A1, p.B1, p.alpha)
            k_ref_p = p.apply(k_ref, p.A2, p.B2, p.alpha)
            v_ref_p = p.apply(v_ref, p.A2, p.B2, p.alpha)
            q_ref_branch = p.apply(q_syn, p.A2, p.B2, p.alpha)

            if src.stop > src.start:
                k_src = p.apply(img_k[:, src, :], p.A3, p.B3, p.alpha)
                v_src = p.apply(img_v[:, src, :], p.A3, p.B3, p.alpha)
                q_src_branch = p.apply(q_syn, p.A3, p.B3, p.alpha)
                branch_k = torch.cat([txt_k, k_syn_p, k_ref_p, k_src], dim=1)
                branch_v = torch.cat([txt_v, v_syn_p, v_ref_p, v_src], dim=1)
                branch_q = torch.cat([q_ref_branch, q_src_branch], dim=1)
            else:
                q_alt = p.apply(q_syn, p.A1, p.B1, p.alpha)
                branch_k = torch.cat([txt_k, k_syn_p, k_ref_p], dim=1)
                branch_v = torch.cat([txt_v, v_syn_p, v_ref_p], dim=1)
                branch_q = torch.cat([q_ref_branch, q_alt], dim=1)

            heads = attn.heads
            head_dim = branch_q.shape[-1] // heads
            bsz = branch_q.shape[0]
            q_branch = branch_q.view(bsz, -1, heads, head_dim).transpose(1, 2).contiguous()
            k_branch = branch_k.view(bsz, -1, heads, head_dim).transpose(1, 2).contiguous()
            v_branch = branch_v.view(bsz, -1, heads, head_dim).transpose(1, 2).contiguous()
            branch_out = _run_attention(attn, q_branch, k_branch, v_branch, None, options)
            branch_out = branch_out.to(q_branch.dtype)
            out_ref = branch_out[:, :syn_len, :]
            out_other = branch_out[:, syn_len : 2 * syn_len, :]
            weights = torch.softmax(self.synthesis_fusion_logits, dim=0)
            img_out[:, syn, :] = (
                weights[0] * base_syn + weights[1] * out_ref + weights[2] * out_other
            )

        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        txt_out = attn.to_add_out(txt_out)
        return img_out, txt_out


class OrionBlockReplacement:
    """ComfyUI patches_replace adapter; only swaps attention.forward during a block call."""

    def __init__(self, attn, processor, mode: str, block_index: int, previous=None):
        self.attn = attn
        self.processor = processor
        self.mode = mode
        self.block_index = int(block_index)
        self.previous = previous

    def __call__(self, args, extra_args):
        original_forward = self.attn.forward
        processor = self.processor
        mode = self.mode
        block_index = self.block_index

        def orion_forward(
            _attn,
            hidden_states,
            encoder_hidden_states=None,
            encoder_hidden_states_mask=None,
            attention_mask=None,
            image_rotary_emb=None,
            transformer_options=None,
            **kwargs,
        ):
            return processor(
                _attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                transformer_options=transformer_options,
                mode=mode,
                block_index=block_index,
                **kwargs,
            )

        with _PATCH_LOCK:
            self.attn.forward = types.MethodType(orion_forward, self.attn)
            try:
                if self.previous is not None:
                    return self.previous(args, extra_args)
                return extra_args["original_block"](args)
            finally:
                self.attn.forward = original_forward

    def to(self, device_or_dtype):
        self.processor.to(device_or_dtype)
        if self.previous is not None and hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def models(self):
        if self.previous is not None and hasattr(self.previous, "models"):
            return self.previous.models()
        return []

    def cleanup(self):
        if self.previous is not None and hasattr(self.previous, "cleanup"):
            self.previous.cleanup()


def _find_diffusion_model(model):
    if hasattr(model, "get_model_object"):
        for path in ("diffusion_model", "model.diffusion_model"):
            try:
                obj = model.get_model_object(path)
                if obj is not None:
                    return obj
            except Exception:
                pass
    base = getattr(model, "model", None)
    return getattr(base, "diffusion_model", None)


def _extract_layer_state(state: Dict[str, torch.Tensor], block_index: int) -> Dict[str, torch.Tensor]:
    marker = f"transformer_blocks.{block_index}.attn.processor."
    result = {}
    for key, value in state.items():
        pos = key.find(marker)
        if pos >= 0:
            name = key[pos + len(marker) :]
            # Keep only the decoupling processor parameters.
            if name in {
                "proj.A1", "proj.A2", "proj.A3",
                "proj.B1", "proj.B2", "proj.B3",
                "synthesis_fusion_logits",
            }:
                result[name] = value
    return result


class ApplyOrionEditAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "processor_weights": ("STRING", {"default": "", "multiline": False}),
                "mode": (["edit", "fusion"], {"default": "edit"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "OrionEdit"

    def apply(self, model, processor_weights: str, mode: str = "edit"):
        path = os.path.expanduser(os.path.expandvars(processor_weights.strip()))
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"OrionEdit processor checkpoint not found: {path!r}")

        diffusion_model = _find_diffusion_model(model)
        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError(
                "This node supports ComfyUI's native QwenImageTransformer2DModel "
                "with diffusion_model.transformer_blocks."
            )
        if len(blocks) != _TOTAL_BLOCKS:
            raise RuntimeError(f"Expected 60 Qwen transformer blocks, found {len(blocks)}.")

        state = load_file(path, device="cpu")
        patched = model.clone()
        existing = (
            patched.model_options.get("transformer_options", {})
            .get("patches_replace", {})
            .get("dit", {})
        )

        for index, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                raise RuntimeError(f"Qwen block {index} has no .attn module.")
            dim = int(getattr(attn, "inner_dim", attn.heads * attn.dim_head))

            if index < _STAGE_A_END:
                processor = OrionAttentionProcessor("A", dim)
            elif index < _STAGE_B_END:
                beta = 8.0 - 6.0 * ((index - _STAGE_A_END) / max(_STAGE_B_END - _STAGE_A_END - 1, 1))
                processor = OrionAttentionProcessor("B", dim, soft_beta=beta)
            else:
                layer_state = _extract_layer_state(state, index)
                required = {
                    "proj.A1", "proj.A2", "proj.A3",
                    "proj.B1", "proj.B2", "proj.B3",
                    "synthesis_fusion_logits",
                }
                missing = required.difference(layer_state)
                if missing:
                    raise RuntimeError(
                        f"Processor checkpoint is missing block {index} parameters: {sorted(missing)}"
                    )
                rank = int(layer_state["proj.A1"].shape[1])
                if int(layer_state["proj.A1"].shape[0]) != dim:
                    raise RuntimeError(
                        f"Block {index} projector dimension {layer_state['proj.A1'].shape[0]} "
                        f"does not match Qwen attention dimension {dim}."
                    )
                processor = OrionAttentionProcessor("C", dim, rank=rank)
                processor.load_state_dict(layer_state, strict=True)

            previous = existing.get(("double_block", index))
            patch = OrionBlockReplacement(attn, processor, mode, index, previous)
            patched.set_model_patch_replace(patch, "dit", "double_block", index)

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