import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_108 = unetloader.load_unet(
            unet_name="wan2.1_vace_1.3B_fp16.safetensors", weight_dtype="default"
        )

        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_110 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_109 = loraloader.load_lora(
            lora_name="Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
            strength_model=0.5,
            strength_clip=1,
            model=get_value_at_index(unetloader_108, 0),
            clip=get_value_at_index(cliploader_110, 0),
        )

        loraloader_159 = loraloader.load_lora(
            lora_name="stickman_epoch_100.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(loraloader_109, 0),
            clip=get_value_at_index(loraloader_109, 1),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="stickman dancing, 2D animation, cel-shaded, clean line art, crisp lines, sharp focus, high resolution, high quality, 4k, vector-like clarity, flat colors, solid fills, consistent line weight, clean shading, smooth gradients, frame-accurate, keyframe style, no blur artifacts, razor-sharp edges, simplified background, clear silhouettes, well-defined contours, precise inked lines, minimal noise, blank white background. AWEMELLO_STICKMAN_STYLE",
            clip=get_value_at_index(loraloader_159, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="blurry, out of focus, motion blur, camera shake, soft focus, double image, ghosting, streaks, low resolution, pixelated, compression artifacts, jpeg artifacts, noise, grain, smudged lines, watercolor bleed, painterly smudge, depth of field, shaky camera, overblurred, motion streaks, fuzzy edges, low detail, inconsistent line weight, low quality, hands, palms, fingers",
            clip=get_value_at_index(loraloader_159, 1),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_105 = vaeloader.load_vae(vae_name="wan_2.1_vae.safetensors")

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_134 = loadimage.load_image(image="stickman - Copy.jpg")

        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_154 = vhs_loadvideo.load_video(
            video="control_vid.mp4",
            force_rate=30,
            custom_width=0,
            custom_height=0,
            frame_load_cap=81,
            skip_first_frames=0,
            select_every_nth=1,
            format="AnimateDiff",
            unique_id=7580115987371150614,
        )

        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        resizeandpadimage = NODE_CLASS_MAPPINGS["ResizeAndPadImage"]()
        wanvacetovideo = NODE_CLASS_MAPPINGS["WanVaceToVideo"]()
        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
        trimvideolatent = NODE_CLASS_MAPPINGS["TrimVideoLatent"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            dwpreprocessor_152 = dwpreprocessor.estimate_pose(
                detect_hand="disable",
                detect_body="enable",
                detect_face="disable",
                resolution=256,
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                scale_stick_for_xinsr_cn="disable",
                image=get_value_at_index(vhs_loadvideo_154, 0),
            )

            resizeandpadimage_156 = resizeandpadimage.EXECUTE_NORMALIZED(
                target_width=480,
                target_height=832,
                padding_color="black",
                interpolation="area",
                image=get_value_at_index(dwpreprocessor_152, 0),
            )

            resizeandpadimage_155 = resizeandpadimage.EXECUTE_NORMALIZED(
                target_width=480,
                target_height=832,
                padding_color="white",
                interpolation="area",
                image=get_value_at_index(loadimage_134, 0),
            )

            wanvacetovideo_49 = wanvacetovideo.EXECUTE_NORMALIZED(
                width=480,
                height=832,
                length=get_value_at_index(vhs_loadvideo_154, 1),
                batch_size=1,
                strength=1,
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                vae=get_value_at_index(vaeloader_105, 0),
                control_video=get_value_at_index(resizeandpadimage_156, 0),
                reference_image=get_value_at_index(resizeandpadimage_155, 0),
            )

            modelsamplingsd3_48 = modelsamplingsd3.patch(
                shift=6, model=get_value_at_index(loraloader_159, 0)
            )

            ksampleradvanced_168 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=4,
                cfg=2,
                sampler_name="dpmpp_2m",
                scheduler="beta",
                start_at_step=0,
                end_at_step=2,
                return_with_leftover_noise="enable",
                model=get_value_at_index(modelsamplingsd3_48, 0),
                positive=get_value_at_index(wanvacetovideo_49, 0),
                negative=get_value_at_index(wanvacetovideo_49, 1),
                latent_image=get_value_at_index(wanvacetovideo_49, 2),
            )

            ksampleradvanced_169 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=4,
                cfg=1,
                sampler_name="dpmpp_2m",
                scheduler="beta",
                start_at_step=2,
                end_at_step=4,
                return_with_leftover_noise="disable",
                model=get_value_at_index(modelsamplingsd3_48, 0),
                positive=get_value_at_index(wanvacetovideo_49, 0),
                negative=get_value_at_index(wanvacetovideo_49, 1),
                latent_image=get_value_at_index(ksampleradvanced_168, 0),
            )

            trimvideolatent_58 = trimvideolatent.EXECUTE_NORMALIZED(
                trim_amount=get_value_at_index(wanvacetovideo_49, 3),
                samples=get_value_at_index(ksampleradvanced_169, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(trimvideolatent_58, 0),
                vae=get_value_at_index(vaeloader_105, 0),
            )

            vhs_videocombine_153 = vhs_videocombine.combine_video(
                frame_rate=30,
                loop_count=0,
                filename_prefix="pose/img",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(resizeandpadimage_156, 0),
                unique_id=8003006861796959269,
            )

            vhs_videocombine_167 = vhs_videocombine.combine_video(
                frame_rate=30,
                loop_count=0,
                filename_prefix="AnimateDiff",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(vaedecode_8, 0),
                unique_id=2169148089914376827,
            )


if __name__ == "__main__":
    main()
