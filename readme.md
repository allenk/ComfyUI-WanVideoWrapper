# ComfyUI wrapper nodes for [WanVideo](https://github.com/Wan-Video/Wan2.1) and related models.


## Memory use update (again)

I've made everythign less reliant on torch.compile for VRAM efficiency, so things should work better even without it. Also figured workaround for some issues when using compile that made first run use drastically more VRAM, issue I battled with myself a lot.


## Update notification that can affect memory use in old workflows

In a recent update I changed how unmerged LoRA weights are handled:

Previously mostly due to my laziness they were always loaded from RAM when used, this was of course inefficient and also made using torch.compile for LoRA applying difficult, thus forcing a graph break when using unmerged LoRAs.

Now the LoRA weights are assigned as buffers to the corresponding modules, so they are part of the blocks and obey the block swapping unifying the offloading and allowing LoRA weights to benefit from the prefetch feature for async offoading. Downside is that this means if you did not use block swap, you will see increased memory use as the LoRAs are part of the model and all on VRAM.

If you use block swap, the LoRAs are swapped along the rest of the block, but the block size is now larger, this means you may have to compensate with couple of more blocks swapped.

Example situation: you use 1GB LoRA unmerged and swap 20 blocks on 14B model, we can divide the LoRA size by block count, single block grows by 25MB, 20 blocks grow by 500MB, so your VRAM usage would be 500MB more than before, to compensate you swap 2 more blocks.

### Unrelated other VRAM issue with torch.compile

After any update that modifies the model code and when using torch.compile it's common to run into issues with VRAM, this can be caused by using older pytorch/triton version without latest compile fixes, and/or from old triton caches, mostly in Windows. This manifests in the issue that first run of new input size may have drastically increased memory use, which can clear from simply running it again, and once cached, not manifest again. Again I've only seen this happen in Windows.

To clear your Triton cache you can delete the contents of following (default) folders:

`C:\Users\<username>\.triton`
`C:\Users\<username>\AppData\Local\Temp\torchinductor_<username>`


## Note: Due to the stupid amount of bots or people thinking this is some of video generation service, I've blocked new accounts from posting issues for now.

# WORK IN PROGRESS (perpetually)

# InfiniteTalk / MultiTalk Stability Notes

The `MultiTalk + infinitetalk` path can be much more thermally aggressive than
a typical Wan / LTX style workload. The main failure mode may look like a
driver timeout or a random unstable run, but in practice the root cause can be
more physical than software-facing:

- short SM-heavy bursts can stay pinned unusually hard
- power can jump faster than the cooling response catches up
- local hotspot behavior may become worse than what common software telemetry suggests
- a system may hit a thermal wall or power wall before VRAM use looks dangerous

This means a system can appear "fine" in normal monitoring tools while still
behaving as if it is right at the edge of a crash.

## What was fixed here

This repo includes three practical mitigations for this path:

1. Safe default speaker-mask generation for 2-speaker MultiTalk when
   `ref_target_masks` are missing.
2. A defensive fallback when `x_ref_attn_map` is missing, preventing early
   `NoneType` crashes in the MultiTalk attention path.
3. Configurable step/window throttling for `infinitetalk` mode to create
   deliberate "breathing points" between GPU work segments.

Those fixes address both sides of the problem:

- correctness: make the workflow actually run instead of crashing early
- runtime stability: reduce the chance that the GPU is driven into a hostile
  thermal / boost regime

## Why this path is different

Observed pattern during validation:

```text
Normal Wan / LTX style run:
  power rises gradually
  heat rises gradually
  fan response has time to catch up

InfiniteTalk / MultiTalk:
  short windows can stay pinned on SM-heavy work
  local heat can spike unusually fast
  fan and power response can become much sharper
```

Very simplified sketch:

```text
Stress / heat density
^
|                        InfiniteTalk
|                    ___/\/\__/\/\___/\/\__
|                 __/
|   Wan / LTX   _/  \____    ____    ______
|                         \__/    \__/
+-------------------------------------------------> time
```

Another way to think about it:

```text
             easy to see in software?     easy to cool in time?
normal load            yes                       usually yes
InfiniTalk burst       not always                not always
```

## Why the throttle exists

This delay is not just cosmetic padding. The goal is to break up the most
problematic path into smaller chunks so the GPU gets natural recovery points:

- between denoise steps
- between audio windows

Important limitation: this cannot interrupt a single long CUDA kernel that is
already in flight. If one fused kernel is itself too aggressive, no external
sleep call can pause it halfway through. What the throttle *can* do is reduce
continuous back-to-back pressure across steps and windows, which was enough to
materially improve real-world stability in testing.

## Default throttle behavior

For `infinitetalk` mode the current defaults are:

- `COMFYUI_MULTITALK_STEP_IDLE_SECONDS = 1.0`
- `COMFYUI_MULTITALK_WINDOW_IDLE_SECONDS = 1.0`

You can override them with environment variables:

```powershell
$env:COMFYUI_MULTITALK_STEP_IDLE_SECONDS="0.5"
$env:COMFYUI_MULTITALK_WINDOW_IDLE_SECONDS="0.5"
```

Or disable them entirely:

```powershell
$env:COMFYUI_MULTITALK_STEP_IDLE_SECONDS="0"
$env:COMFYUI_MULTITALK_WINDOW_IDLE_SECONDS="0"
```

## Practical validation summary

One validated `MultiTalk + infinitetalk` run used:

- `347` frames
- `7` windows
- `6` denoise steps per window
- auto-generated 2-speaker masks
- explicit `1.0s` step throttle
- explicit `1.0s` window throttle

The measured behavior after patching was:

- the workflow completed successfully instead of failing early
- total runtime increased by about `1 minute`
- GPU behavior became noticeably calmer
- fan noise was less extreme
- thermal behavior looked more controlled
- the workload no longer felt like it was "one bad transient away" from a crash

A representative monitored range during the stabilized run looked roughly like:

- board power near the low `400W` range
- GPU temperature mostly around the upper `70C` to low `80C` range
- SM utilization still frequently near saturation

That last point matters: the throttle does **not** make the workload easy. It
simply makes it less hostile.

## Practical interpretation

The patched path should be viewed as:

- more reliable
- more repeatable
- less acoustically violent
- less likely to trip a sudden instability event

It should **not** be viewed as proof that all GPUs can safely run this model at
full unrestricted boost.

## Thermal wall / power wall / hotspot notes

For this workload, the limiting factor may be one of the following:

- `thermal wall`
  the card or silicon region is getting hot enough that boost behavior becomes
  unstable or defensive
- `power wall`
  transient power demand is too aggressive, especially when boost tries to
  chase short bursts
- `local hotspot wall`
  one region is much hotter than average, even if the reported global temp
  still looks survivable
- `long SM-saturating kernels`
  certain kernels keep compute resources pinned with very little natural rest

Simplified picture:

```text
                 +----------------------+
requested work ->|  fused / heavy path  |-> sharp transient stress
                 +----------------------+
                              |
                              v
                    +-------------------+
                    | boost / heat / VR |
                    +-------------------+
                              |
                 +------------+------------+
                 |                         |
                 v                         v
           survives cleanly         hits wall / TDR / hang
```

## If a system is still unstable

If throttling alone is not enough, try the following:

- reduce `frame_window_size`
- reduce resolution
- reduce denoise step count
- cap board power
- increase `COMFYUI_MULTITALK_STEP_IDLE_SECONDS`
- increase `COMFYUI_MULTITALK_WINDOW_IDLE_SECONDS`
- avoid assuming that "normal VRAM usage" means "safe thermal behavior"

In practice, reducing workload density and limiting boost behavior can matter
more than registry-based TDR tweaks.

## Bottom line

This patch is a stability aid, not a universal guarantee. It improves a real
edge case, fixes a real correctness bug, and creates safer default behavior for
an unusually aggressive path, but some systems may still need tighter power
limits or smaller workloads to stay fully stable.

# Why should I use custom nodes when WanVideo works natively?

Short answer: Unless it's a model/feature not available yet on native, you shouldn't.

Long answer: Due to the complexity of ComfyUI core code, and my lack of coding experience, in many cases it's far easier and faster to implement new models and features to a standalone wrapper, so this is a way to test things relatively quickly. I consider this my personal sandbox (which is obviously open for everyone) to play with without having to worry about compability issues etc, but as such this code is always work in progress and prone to have issues. Also not all new models end up being worth the trouble to implement in core Comfy, though I've also made some patcher nodes to allow using them in native workflows, such as the [ATI](https://huggingface.co/bytedance-research/ATI) node available in this wrapper. This is also the end goal, idea isn't to compete or even offer alternatives to everything available in native workflows. All that said (this is clearly not a sales pitch) I do appreciate everyone using these nodes to explore new releases and possibilities with WanVideo.

# Installation
1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

  `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt`

## Models

https://huggingface.co/Kijai/WanVideo_comfy/tree/main

fp8 scaled models (personal recommendation):

https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled

Text encoders to `ComfyUI/models/text_encoders`

Clip vision to `ComfyUI/models/clip_vision`

Transformer (main video model) to `ComfyUI/models/diffusion_models`

Vae to `ComfyUI/models/vae`

You can also use the native ComfyUI text encoding and clip vision loader with the wrapper instead of the original models:

![image](https://github.com/user-attachments/assets/6a2fd9a5-8163-4c93-b362-92ef34dbd3a4)

GGUF models can now be loaded in the main model loader as well.

---
Supported extra models:

SkyReels: https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9

WanVideoFun: https://huggingface.co/collections/alibaba-pai/wan21-fun-v11-680f514c89fe7b4df9d44f17

ReCamMaster: https://github.com/KwaiVGI/ReCamMaster

VACE: https://github.com/ali-vilab/VACE

Phantom: https://huggingface.co/bytedance-research/Phantom

ATI: https://huggingface.co/bytedance-research/ATI

Uni3C: https://github.com/alibaba-damo-academy/Uni3C

MiniMaxRemover: https://huggingface.co/zibojia/minimax-remover

MAGREF: https://huggingface.co/MAGREF-Video/MAGREF

FantasyTalking: https://github.com/Fantasy-AMAP/fantasy-talking

FantasyPortrait: https://github.com/Fantasy-AMAP/fantasy-portrait

MultiTalk: https://github.com/MeiGen-AI/MultiTalk

EchoShot: https://github.com/D2I-ai/EchoShot

Stand-In: https://github.com/WeChatCV/Stand-In

HuMo: https://github.com/Phantom-video/HuMo

WanAnimate: https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate

Lynx: https://github.com/bytedance/lynx

MoCha: https://github.com/Orange-3DV-Team/MoCha

UniLumos: https://github.com/alibaba-damo-academy/Lumos-Custom

Bindweave: https://github.com/bytedance/BindWeave

Training free techniques:

TimeToMove: https://github.com/time-to-move/TTM

SteadyDancer: https://github.com/MCG-NJU/SteadyDancer

One-to-all-Animation: https://github.com/ssj9596/One-to-All-Animation

SCAIL: https://github.com/zai-org/SCAIL


Not exactly Wan model, but close enough to work with the code base:

LongCat-Video: https://meituan-longcat.github.io/LongCat-Video/


Examples:
---

WanAnimate:

https://github.com/user-attachments/assets/f370b001-0f98-4c4c-bcb5-cfad0b330697

[ReCamMaster](https://github.com/KwaiVGI/ReCamMaster):


https://github.com/user-attachments/assets/c58a12c2-13ba-4af8-8041-e283dbef197e


TeaCache (with the old temporary WIP naive version, I2V):

**Note that with the new version the threshold values should be 10x higher**

Range of 0.25-0.30 seems good when using the coefficients, start step can be 0, with more aggressive threshold values it may make sense to start later to avoid any potential step skips early on, that generally ruin the motion.

https://github.com/user-attachments/assets/504a9a50-3337-43d2-97b8-8e1661f29f46


Context window test:

1025 frames using window size of 81 frames, with 16 overlap. With the 1.3B T2V model this used under 5GB VRAM and took 10 minutes to gen on a 5090:

https://github.com/user-attachments/assets/89b393af-cf1b-49ae-aa29-23e57f65911e

---


This very first test was 512x512x81

~16GB used with 20/40 blocks offloaded

https://github.com/user-attachments/assets/fa6d0a4f-4a4d-4de5-84a4-877cc37b715f

Vid2vid example:


with 14B T2V model:

https://github.com/user-attachments/assets/ef228b8a-a13a-4327-8a1b-1eb343cf00d8

with 1.3B T2V model

https://github.com/user-attachments/assets/4f35ba84-da7a-4d5b-97ee-9641296f391e



