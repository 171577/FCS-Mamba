import argparse
import importlib.util
import os
import sys

import torch
import time

from option import Options
from model.module.MSDANet import MSDANet


def _load_change_mamba_supported_ops():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    get_flops_path = os.path.join(repo_root, "ChangeMamba-master", "analyze", "get_flops.py")
    if not os.path.isfile(get_flops_path):
        return {}

    spec = importlib.util.spec_from_file_location("change_mamba_get_flops", get_flops_path)
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    sys.modules["change_mamba_get_flops"] = module
    spec.loader.exec_module(module)

    custom_ops = getattr(module, "supported_ops", {})
    try:
        from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS

        merged = dict(_DEFAULT_SUPPORTED_OPS)
        merged.update(custom_ops)
        return merged
    except Exception:
        return custom_ops


def _parse_options():
    opt_builder = Options()
    opt_builder.init()
    opt_builder.parser.add_argument("--profile_device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    opt_builder.parser.add_argument("--profile_batch", type=int, default=1)
    opt_builder.parser.add_argument("--profile_warmup", type=int, default=50)
    opt_builder.parser.add_argument("--profile_iters", type=int, default=200)
    opt_builder.parser.add_argument("--profile_amp", action="store_true")
    opt_builder.parser.add_argument("--profile_no_pretrained", action="store_true")
    opt = opt_builder.parser.parse_args()

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        _id = int(str_id)
        if _id >= 0:
            opt.gpu_ids.append(_id)

    if isinstance(getattr(opt, 'vssm_pretrained', None), str) and opt.vssm_pretrained.strip() == '':
        opt.vssm_pretrained = None

    if isinstance(getattr(opt, 'vssm_depths', None), str):
        depths_str = opt.vssm_depths.strip()
        if depths_str:
            opt.vssm_depths = tuple(int(x.strip()) for x in depths_str.split(',') if x.strip() != '')

    if isinstance(getattr(opt, 'vssm_dims', None), str):
        dims_str = opt.vssm_dims.strip()
        if ',' in dims_str:
            opt.vssm_dims = [int(x.strip()) for x in dims_str.split(',') if x.strip() != '']
        elif dims_str:
            opt.vssm_dims = int(dims_str)

    return opt


def _benchmark_fps(model, x1, x2, device, warmup=50, iters=200, amp=False):
    if iters <= 0:
        return None

    model.eval()

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        with torch.inference_mode():
            for _ in range(max(0, int(warmup))):
                with torch.cuda.amp.autocast(enabled=bool(amp)):
                    _ = model(x1, x2)

            torch.cuda.synchronize()
            start_event.record()
            for _ in range(int(iters)):
                with torch.cuda.amp.autocast(enabled=bool(amp)):
                    _ = model(x1, x2)
            end_event.record()
            torch.cuda.synchronize()

        total_ms = float(start_event.elapsed_time(end_event))
        total_s = total_ms / 1000.0
    else:
        with torch.inference_mode():
            for _ in range(max(0, int(warmup))):
                _ = model(x1, x2)

            t0 = time.perf_counter()
            for _ in range(int(iters)):
                _ = model(x1, x2)
            t1 = time.perf_counter()

        total_s = float(t1 - t0)

    if total_s <= 0:
        return None

    total_pairs = int(iters) * int(x1.shape[0])
    fps = total_pairs / total_s
    ms_per_pair = (total_s / (int(iters))) * 1000.0
    return {
        "total_s": total_s,
        "fps": fps,
        "ms_per_iter": ms_per_pair,
        "iters": int(iters),
        "warmup": int(warmup),
        "amp": bool(amp),
    }


def main():
    opt = _parse_options()

    if opt.profile_device == "cuda":
        device = torch.device("cuda")
    elif opt.profile_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and getattr(opt, "gpu_ids", None):
        try:
            torch.cuda.set_device(int(opt.gpu_ids[0]))
        except Exception:
            pass

    if device.type == "cpu":
        print("Device=cpu. If vmamba/triton ops require CUDA, profiling may fail. Try --profile_device cuda.")

    vssm_pretrained_override = None if opt.profile_no_pretrained else getattr(opt, "vssm_pretrained", None)

    model = MSDANet.from_opt(
        opt,
        input_nc=3,
        output_nc=2,
        use_depthwise=bool(getattr(opt, "use_depthwise", True)),
        use_feedback_decoder=bool(getattr(opt, "use_feedback_decoder", True)),
        vssm_pretrained=vssm_pretrained_override,
    ).to(device)
    model.eval()

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    h = int(getattr(opt, "input_size", 256))
    w = h
    b = int(getattr(opt, "profile_batch", 1))
    x1 = torch.randn(b, 3, h, w, device=device)
    x2 = torch.randn(b, 3, h, w, device=device)

    supported_ops = _load_change_mamba_supported_ops()

    bench = _benchmark_fps(
        model,
        x1,
        x2,
        device=device,
        warmup=int(getattr(opt, "profile_warmup", 50)),
        iters=int(getattr(opt, "profile_iters", 200)),
        amp=bool(getattr(opt, "profile_amp", False)),
    )

    try:
        from fvcore.nn.flop_count import flop_count

        flops_dict, unsupported = flop_count(model=model, inputs=(x1, x2), supported_ops=supported_ops)
        gflops_total = float(sum(flops_dict.values()))
        flops_total = gflops_total * 1e9

        print("==============================")
        print(f"Input: (B=\"{b}\", C=3, H={h}, W={w}) x2")
        print(f"Backbone: {getattr(opt, 'backbone_type', 'N/A')}; Weak source: {getattr(opt, 'weak_feature_source', 'N/A')}; Main branch: {getattr(opt, 'main_branch_type', 'N/A')}")
        print(f"Params(total): {params_total} ({params_total/1e6:.3f} M)")
        print(f"Params(trainable): {params_trainable} ({params_trainable/1e6:.3f} M)")
        print(f"FLOPs (forward): {gflops_total:.6f} GFLOPs ({flops_total:.0f} FLOPs)")
        if bench is not None:
            print(
                f"Speed: {bench['fps']:.3f} FPS (pairs/s), {bench['ms_per_iter']:.3f} ms/iter, "
                f"iters={bench['iters']}, warmup={bench['warmup']}, amp={bench['amp']}, device={device.type}"
            )
        if unsupported:
            print(f"Unsupported ops: {sorted(list(unsupported))}")
        print("==============================")
    except Exception as e:
        print("FLOPs profiling failed:", repr(e))
        print("Params only:")
        print(f"Params(total): {params_total} ({params_total/1e6:.3f} M)")
        print(f"Params(trainable): {params_trainable} ({params_trainable/1e6:.3f} M)")


if __name__ == "__main__":
    main()
