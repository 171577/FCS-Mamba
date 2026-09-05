import os
import json
import logging
from datetime import datetime

import torch
from tqdm import tqdm

from option import Options
from data.cd_dataset import DataLoader
from model.module.MSDANet import MSDANet
from util.metric_tool import ConfuseMatrixMeter


def save_test_results_5(metrics_5, save_path, model_name="MSDANet"):
    results = {
        'model': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {},
    }
    for k, v in metrics_5.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if not isinstance(sub_v, dict):
                    results['metrics'][f"{k}_{sub_k}"] = float(sub_v)
        else:
            results['metrics'][k] = float(v)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {save_path}")


def load_checkpoint_flexible(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    checkpoint_loaded = False
    loaded_keys = 0
    skipped_keys = 0

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return checkpoint_loaded, loaded_keys, skipped_keys

    logging.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    state_dict = MSDANet.remap_legacy_state_dict_keys(state_dict)

    model_state = model.state_dict()

    incompatible_keys = []
    for name, param in state_dict.items():
        if name in model_state:
            try:
                if model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    loaded_keys += 1
                else:
                    skipped_keys += 1
                    incompatible_keys.append(name)
            except Exception:
                skipped_keys += 1
                incompatible_keys.append(name)
        else:
            skipped_keys += 1

    model.load_state_dict(model_state, strict=False)
    checkpoint_loaded = True

    logging.info(f"Checkpoint loaded: {loaded_keys} keys loaded, {skipped_keys} keys skipped/mismatched")
    if incompatible_keys:
        logging.info(
            f"Incompatible keys: {incompatible_keys[:5]}{'...' if len(incompatible_keys) > 5 else ''}"
        )

    return checkpoint_loaded, loaded_keys, skipped_keys


def test_model(opt, checkpoint_path=None):
    logging.basicConfig(level=logging.INFO)

    device = torch.device(
        'cuda'
        if torch.cuda.is_available() and len(getattr(opt, 'gpu_ids', [])) > 0 and opt.gpu_ids[0] >= 0
        else 'cpu'
    )

    logging.info(f"Using device: {device}")

    # Match training-time model configuration to ensure checkpoint compatibility.
    model = MSDANet.from_opt(
        opt,
        input_nc=3,
        output_nc=2,
        use_depthwise=True,
        use_feedback_decoder=True,
    ).to(device)

    checkpoint_loaded = False
    if checkpoint_path:
        try:
            checkpoint_loaded, _, _ = load_checkpoint_flexible(model, checkpoint_path, device)
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)[:120]}")
            checkpoint_loaded = False

    if checkpoint_loaded:
        logging.info("Checkpoint loaded successfully!")
    else:
        logging.warning("No valid checkpoint loaded. Using randomly initialized model.")

    opt.phase = 'test'

    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    logging.info(f"#test images = {len(test_loader)}")

    model.eval()
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    tbar = tqdm(test_data, desc="Testing", ncols=100)
    with torch.no_grad():
        for data in tbar:
            img1 = data['img1'].to(device)
            img2 = data['img2'].to(device)
            label = data['label']

            out = model(img1, img2)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            pred = torch.argmax(logits, dim=1)

            running_metric.update_cm(
                pr=pred.detach().cpu().numpy(),
                gt=label.detach().cpu().numpy(),
            )

            scores = running_metric.get_scores()
            tbar.set_postfix({'F1_1': f"{scores.get('F1_1', 0)*100:.2f}", 'mIoU': f"{scores.get('miou', 0)*100:.2f}"})

    final_scores = running_metric.get_scores()

    metrics_5 = {
        'acc': final_scores.get('acc', 0),
        'miou': final_scores.get('miou', 0),
        'F1_1': final_scores.get('F1_1', 0),
        'precision_1': final_scores.get('precision_1', 0),
        'recall_1': final_scores.get('recall_1', 0),
    }

    print("\n" + "=" * 80)
    print("TEST RESULTS (5 metrics)")
    print("=" * 80)
    print(f"acc         : {metrics_5['acc']*100:6.2f}%")
    print(f"miou        : {metrics_5['miou']*100:6.2f}%")
    print(f"F1_1        : {metrics_5['F1_1']*100:6.2f}%")
    print(f"precision_1 : {metrics_5['precision_1']*100:6.2f}%")
    print(f"recall_1    : {metrics_5['recall_1']*100:6.2f}%")
    print("=" * 80 + "\n")

    return metrics_5


if __name__ == '__main__':
    opt = Options().parse()

    checkpoint_candidates = [
        f"{opt.checkpoint_dir}/{opt.name}/best_model.pth",
        f"{opt.checkpoint_dir}/best_model.pth",
        getattr(opt, 'weight_dir', ''),
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if candidate and os.path.exists(candidate):
            try:
                torch.load(candidate, map_location='cpu')
                checkpoint_path = candidate
                logging.info(f"Found valid checkpoint: {candidate}")
                break
            except Exception as e:
                logging.warning(f"Checkpoint {candidate} is corrupted: {str(e)[:120]}")

    metrics_5 = test_model(opt, checkpoint_path)

    if checkpoint_path:
        result_dir = os.path.dirname(checkpoint_path)
    else:
        result_dir = os.path.join(opt.checkpoint_dir, opt.name)

    result_file = os.path.join(result_dir, 'test_results.json')
    save_test_results_5(metrics_5, result_file, model_name=getattr(opt, 'name', 'MSDANet'))
