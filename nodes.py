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

# Import Orion block processors from local models package
from .transformer_orion import set_orion_block_processors


# ==========================================
# Mathematical Latent Packing Helpers
# ==========================================

def _pack_latents_2d(latents: torch.Tensor) -> torch.Tensor:
    """Pack [B, C, H, W] -> [B, (H//2)*(W//2), C*4]."""
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
            # Fallback scan in loras directory
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
        # 1. Strictly adhere to ComfyUI contract: clone ModelPatcher
        model = model.clone()

        # 2. Apply LoRA on quantized/int8 model in RAM via native ModelPatcher
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path and os.path.exists(lora_path):
                lora_weights = comfy.utils.load_torch_file(lora_path)
                model, _ = comfy.sd.load_lora_for_models(model, None, lora_weights, lora_strength, 0)

        # 3. Inject Orion Attention Processors onto Diffusion Transformer
        target_model = getattr(model.model, "diffusion_model", model.model)
        set_orion_block_processors(target_model, d=3072, rank=256, use_gate=False)

        # 4. Load Tri-Orthogonal processor weights into Stage C layers (40..59)
        if processor_name != "None":
            proc_path = folder_paths.get_full_path("processors", processor_name)
            if not proc_path:
                proc_path = folder_paths.get_full_path("loras", processor_name)

            if proc_path and os.path.exists(proc_path):
                state = load_sft(proc_path, device="cpu")
                dec_sd = {}
                for key, val in state.items():
                    if "attn.processor" in key:
                        clean_key = key
                        for prefix in ["transformer.", "diffusion_model.", "model."]:
                            if clean_key.startswith(prefix):
                                clean_key = clean_key[len(prefix):]
                        dec_sd[clean_key] = val
                target_model.load_state_dict(dec_sd, strict=False)

        # 5. Intercept KSampler execution via native function wrapper
        def orion_unet_wrapper(apply_model, args):
            latent_in = args["input"]  # [B, 16, H//8, W//8]
            c = args["c"]
            transformer_options = c.get("transformer_options", {})
            cond_orion = c.get("orion_cond", None)

            # If conditioning doesn't come from OrionPrep, run standard pass
            if cond_orion is None:
                return apply_model(latent_in, args["timestep"], **c)

            ref_latents = cond_orion["ref_latents"]
            seg_slices_img = cond_orion["seg_slices_img"]
            img_shapes = cond_orion["img_shapes"]

            B, C, H, W = latent_in.shape
            syn_packed = _pack_latents_2d(latent_in)

            # Sequence: [syn | ref | src]
            packed_input = torch.cat(
                [syn_packed, ref_latents.to(syn_packed.device, dtype=syn_packed.dtype)], dim=1
            )

            # Inject Orion slice routing and stage parameters
            attn_kwargs = transformer_options.get("attention_kwargs", {}).copy()
            attn_kwargs["seg_slices_img"] = seg_slices_img
            attn_kwargs["soft_beta"] = soft_beta
            transformer_options["attention_kwargs"] = attn_kwargs
            c["transformer_options"] = transformer_options
            c["img_shapes"] = img_shapes

            # Forward pass through streamed INT8 blocks
            out = apply_model(packed_input, args["timestep"], **c)

            # Slice target portion and unpack back to standard latent format
            syn_out = out[:, :syn_packed.shape[1], :]
            return _unpack_latents_2d(syn_out, H, W, C=16)

        model.set_model_unet_function_wrapper(orion_unet_wrapper)
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
        else:
            condition_pils = refs

        # 2. Rescale condition images to target VAE dimensions
        vae_image_sizes = []
        packed_cond_list = []
        packed_lengths = []

        for img in condition_pils:
            vw, vh = calculate_dimensions(1024 * 1024, img.width / img.height)
            vae_image_sizes.append((vw, vh))
            img_resized = img.resize((vw, vh), Image.BICUBIC)

            # Native VAE encode -> [1, 16, vh//8, vw//8]
            lat = vae.encode(pil2tensor(img_resized)[:, :, :, :3])
            packed_lat = _pack_latents_2d(lat)
            packed_cond_list.append(packed_lat)
            packed_lengths.append(packed_lat.shape[1])

        packed_cond_latents = torch.cat(packed_cond_list, dim=1)

        # 3. Compute slice indices and 3D positional grid shapes
        L_syn = (height // 16) * (width // 16)
        seg_slices_img = build_dns_seg_slices(L_syn, packed_lengths, has_source=has_source)

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

        # 5. Output target noise latent placeholder for KSampler
        latent_target = {
            "samples": torch.zeros(1, 16, height // 8, width // 8, dtype=torch.float32)
        }

        return (new_conditioning, latent_target)


# ==========================================
# Node Registration Mapping
# ==========================================

NODE_CLASS_MAPPINGS = {
    "ApplyOrionEditPatcher": ApplyOrionEditPatcher,
    "OrionEditPrepConditioning": OrionEditPrepConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyOrionEditPatcher": "Apply OrionEdit Patcher",
    "OrionEditPrepConditioning": "OrionEdit Prep Latents & Conditioning",
}