import os
import math
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file as load_sft

import folder_paths
import comfy.utils
import comfy.sd

# Register custom model subfolder for processor weights
if "processors" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        "processors", os.path.join(folder_paths.models_dir, "processors")
    )

# Resilient import: works whether transformer_orion.py is in the root or a models/ subfolder
try:
    from .transformer_orion import set_orion_block_processors
except ImportError:
    from .models.transformer_orion import set_orion_block_processors


# ==========================================
# Mathematical Latent Packing Helpers
# ==========================================

def _pack_latents_2d(latents: torch.Tensor) -> torch.Tensor:
    """Pack [B, C, H, W] or [B, C, 1, H, W] -> [B, (H//2)*(W//2), C*4]."""
    # Handle 5D spatio-temporal latents from Qwen-Image 3D VAE
    if latents.ndim == 5:
        if latents.shape[2] == 1:
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        elif latents.shape[1] == 1:
            latents = latents.squeeze(1)  # [B, 1, C, H, W] -> [B, C, H, W]
        else:
            latents = latents[:, :, 0, :, :]

    B, C, H, W = latents.shape
    x = latents.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // 2) * (W // 2), C * 4)


def _unpack_latents_2d(packed: torch.Tensor, H: int, W: int, C: int = 16) -> torch.Tensor:
    """Unpack [B, (H//2)*(W//2), C*4] -> [B, C, H, W]."""
    B, L, D = packed.shape
    x = packed.view(B, H // 2, W // 2, C, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, H, W)


def calculate_dimensions(target_area: int, ratio: float):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _build_composited_reference_canvas(refs, source):
    if len(refs) == 1:
        return refs[0]
    strip_w = sum(r.width for r in refs)
    strip_h = max(r.height for r in refs)
    strip = Image.new("RGB", (strip_w, strip_h), (0, 0, 0))
    x = 0
    for ref in refs:
        strip.paste(ref, (x, 0))
        x += ref.width

    canvas_w, canvas_h = source.size if source is not None else refs[0].size
    scale = min(canvas_w / strip_w, canvas_h / strip_h)
    new_w = max(1, int(round(strip_w * scale)))
    new_h = max(1, int(round(strip_h * scale)))
    resized_strip = strip.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    canvas.paste(resized_strip, ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2))
    return canvas


def build_image_stream_slices(L_syn: int, L_ref: int, L_src: int, start_offset: int = 0):
    return {
        "syn": slice(start_offset, start_offset + L_syn),
        "ref": slice(start_offset + L_syn, start_offset + L_syn + L_ref),
        "src": slice(start_offset + L_syn + L_ref, start_offset + L_syn + L_ref + L_src),
    }


def build_dns_seg_slices(L_syn: int, packed_lengths, has_source: bool):
    lengths = list(packed_lengths)
    if not has_source:
        return build_image_stream_slices(L_syn, sum(lengths), 0)
    return build_image_stream_slices(L_syn, lengths[0], lengths[1])


# ==========================================
# Custom Node 1: Apply OrionEdit Patcher
# ==========================================

class ApplyOrionEditPatcher:
    @classmethod
    def INPUT_TYPES(s):
        proc_files = folder_paths.get_filename_list("processors")
        if not proc_files:
            # Fallback scan in loras directory if user placed it there
            proc_files = [f for f in folder_paths.get_filename_list("loras") if "proc" in f.lower()]
        proc_list = proc_files if proc_files else ["None"]

        return {
            "required": {
                "model": ("MODEL",),
                "processor_name": (proc_list,),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "soft_beta": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patcher"
    CATEGORY = "OrionEdit"

    def apply_patcher(self, model, processor_name, lora_name, soft_beta, lora_strength):
        print(f"\n[OrionEdit] Initializing ApplyOrionEditPatcher...")
        # 1. Strictly adhere to ComfyUI contract: clone ModelPatcher
        model = model.clone()

        # 2. Apply LoRA on quantized/int8 model in RAM via native ModelPatcher
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path and os.path.exists(lora_path):
                print(f"[OrionEdit] Loading LoRA: {lora_name} (strength: {lora_strength})")
                lora_weights = comfy.utils.load_torch_file(lora_path)
                model, _ = comfy.sd.load_lora_for_models(model, None, lora_weights, lora_strength, 0)
                print(f"[OrionEdit] LoRA successfully patched to model.")
            else:
                print(f"[OrionEdit Warning] LoRA file not found: {lora_name}")

        # 3. Inject Orion Attention Processors onto Diffusion Transformer
        target_model = getattr(model.model, "diffusion_model", model.model)
        print(f"[OrionEdit] Installing 3-stage processors across 60 transformer layers...")
        set_orion_block_processors(target_model, d=3072, rank=256, use_gate=False)

        # 4. Safely load Tri-Orthogonal processor weights directly into the processors
        if processor_name != "None":
            proc_path = folder_paths.get_full_path("processors", processor_name)
            if not proc_path:
                proc_path = folder_paths.get_full_path("loras", processor_name)

            if proc_path and os.path.exists(proc_path):
                print(f"[OrionEdit] Loading processor weights from: {proc_path}")
                state = load_sft(proc_path, device="cpu")
                
                # Group keys by block index: transformer_blocks.{i}.attn.processor...
                loaded_count = 0
                for bi, blk in enumerate(target_model.transformer_blocks):
                    proc = getattr(blk.attn, "processor", None)
                    if proc is not None and hasattr(proc, "state_dict"):
                        prefix_candidates = [
                            f"transformer.transformer_blocks.{bi}.attn.processor.",
                            f"transformer_blocks.{bi}.attn.processor.",
                            f"diffusion_model.transformer_blocks.{bi}.attn.processor.",
                        ]
                        proc_sd = {}
                        for k, v in state.items():
                            for pfx in prefix_candidates:
                                if k.startswith(pfx):
                                    sub_key = k[len(pfx):]
                                    proc_sd[sub_key] = v
                                    break
                        if proc_sd:
                            proc.load_state_dict(proc_sd, strict=False)
                            loaded_count += len(proc_sd)

                print(f"[OrionEdit] Successfully injected {loaded_count} processor tensors into attention blocks.")
            else:
                print(f"[OrionEdit Warning] Processor file not found: {processor_name}")

        # 5. Intercept KSampler execution via native function wrapper
        def orion_unet_wrapper(apply_model, args):
            latent_in = args["input"]
            c = args["c"]
            transformer_options = c.get("transformer_options", {})
            
            # Check multiple standard ComfyUI locations where cond metadata might live
            cond_orion = c.get("orion_cond", None)
            if cond_orion is None and "transformer_options" in c:
                cond_orion = c["transformer_options"].get("orion_cond", None)

            # If this is purely unconditional and Orion cond is absent, proceed normally
            if cond_orion is None:
                return apply_model(latent_in, args["timestep"], **c)

            ref_latents = cond_orion["ref_latents"]
            seg_slices_img = cond_orion["seg_slices_img"]
            img_shapes = cond_orion["img_shapes"]

            orig_ndim = latent_in.ndim
            H, W = latent_in.shape[-2], latent_in.shape[-1]
            C = latent_in.shape[1] if (orig_ndim == 4 or latent_in.shape[1] != 1) else latent_in.shape[2]

            syn_packed = _pack_latents_2d(latent_in)

            # Sequence: [syn | ref | src]
            packed_input = torch.cat(
                [syn_packed, ref_latents.to(syn_packed.device, dtype=syn_packed.dtype)], dim=1
            )

            # Pass Orion parameters into transformer_options
            attn_kwargs = transformer_options.get("attention_kwargs", {}).copy()
            attn_kwargs["seg_slices_img"] = seg_slices_img
            attn_kwargs["soft_beta"] = soft_beta
            transformer_options["attention_kwargs"] = attn_kwargs
            transformer_options["orion_cond"] = cond_orion
            c["transformer_options"] = transformer_options
            c["img_shapes"] = img_shapes

            out = apply_model(packed_input, args["timestep"], **c)

            syn_out = out[:, :syn_packed.shape[1], :]
            unpacked = _unpack_latents_2d(syn_out, H, W, C=C)

            if orig_ndim == 5:
                if latent_in.shape[1] == 1:
                    unpacked = unpacked.unsqueeze(1)
                else:
                    unpacked = unpacked.unsqueeze(2)

            return unpacked

        model.set_model_unet_function_wrapper(orion_unet_wrapper)
        print(f"[OrionEdit] Function wrapper attached to model. Ready for KSampler.\n")
        return (model,)


# ==========================================
# Custom Node 2: OrionEdit Prep Conditioning
# ==========================================

class OrionEditPrepConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_image_1": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
            },
            "optional": {
                "reference_image_2": ("IMAGE",),
                "source_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "prepare"
    CATEGORY = "OrionEdit"

    def prepare(self, conditioning, vae, reference_image_1, width, height,
                reference_image_2=None, source_image=None):

        print(f"\n[OrionEdit Prep] Preparing condition latents and image stream slices...")

        def tensor2pil(img_tensor):
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            np_img = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return Image.fromarray(np_img)

        def pil2tensor(pil_img):
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            return torch.from_numpy(np_img).unsqueeze(0)

        # 1. Classify task mode (editing vs. fusion)
        refs = [tensor2pil(reference_image_1)]
        if reference_image_2 is not None:
            refs.append(tensor2pil(reference_image_2))

        has_source = source_image is not None
        if has_source:
            source_pil = tensor2pil(source_image)
            composed_ref = _build_composited_reference_canvas(refs, source_pil)
            condition_pils = [composed_ref, source_pil]
            print(f"[OrionEdit Prep] Mode: EDITING (1 composited reference canvas + 1 source image)")
        else:
            condition_pils = refs
            print(f"[OrionEdit Prep] Mode: FUSION ({len(refs)} reference images)")

        # 2. Rescale condition images to target VAE dimensions and encode
        vae_image_sizes = []
        packed_cond_list = []
        packed_lengths = []
        detected_5d = False

        for idx, img in enumerate(condition_pils):
            vw, vh = calculate_dimensions(1024 * 1024, img.width / img.height)
            vae_image_sizes.append((vw, vh))
            img_resized = img.resize((vw, vh), Image.BICUBIC)

            # Native VAE encode
            lat = vae.encode(pil2tensor(img_resized)[:, :, :, :3])
            print(f"[OrionEdit Prep] Condition Image {idx+1} ({vw}x{vh}) encoded to VAE latent shape: {list(lat.shape)}")
            
            if lat.ndim == 5:
                detected_5d = True

            packed_lat = _pack_latents_2d(lat)
            packed_cond_list.append(packed_lat)
            packed_lengths.append(packed_lat.shape[1])

        packed_cond_latents = torch.cat(packed_cond_list, dim=1)
        print(f"[OrionEdit Prep] Total packed condition sequence length: {packed_cond_latents.shape[1]}")

        # 3. Compute slice indices and 3D positional grid shapes
        L_syn = (height // 16) * (width // 16)
        seg_slices_img = build_dns_seg_slices(L_syn, packed_lengths, has_source=has_source)
        print(f"[OrionEdit Prep] Stream Slices -> syn: {seg_slices_img['syn']}, ref: {seg_slices_img['ref']}, src: {seg_slices_img['src']}")

        img_shapes = [
            [
                (1, height // 16, width // 16),
                *[(1, vh // 16, vw // 16) for vw, vh in vae_image_sizes],
            ]
        ]

        # 4. Enrich conditioning dictionary for native KSampler
        new_conditioning = []
        for emb, meta in conditioning:
            new_meta = meta.copy()
            new_meta["orion_cond"] = {
                "ref_latents": packed_cond_latents,
                "seg_slices_img": seg_slices_img,
                "img_shapes": img_shapes,
            }
            new_conditioning.append([emb, new_meta])

        # 5. Output target noise latent placeholder matching VAE dimensionality (4D vs 5D)
        if detected_5d:
            sample_tensor = torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32)
        else:
            sample_tensor = torch.zeros(1, 16, height // 8, width // 8, dtype=torch.float32)

        print(f"[OrionEdit Prep] Target latent placeholder shape: {list(sample_tensor.shape)}\n")
        latent_target = {"samples": sample_tensor}

        return (new_conditioning, latent_target)

# ==========================================
# Custom Node 3: OrionEdit helper
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
    "ApplyOrionEditPatcher": ApplyOrionEditPatcher,
    "OrionEditPrepConditioning": OrionEditPrepConditioning,
    "QwenAttentionSpyNode": QwenAttentionSpyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyOrionEditPatcher": "Apply OrionEdit Patcher",
    "OrionEditPrepConditioning": "OrionEdit Prep Latents & Conditioning",
    "QwenAttentionSpyNode": "Debug Qwen Attention Spy",
}