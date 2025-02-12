import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode


# FIXME たぶん入力の端は破棄しかされないのでは？（パディングが必須）
# 先にパディングしてからsliding_window_inferenceに渡す？
def sliding_window_inference_edge_discard(
    inputs: torch.Tensor,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    predictor,
    overlap: float = 0.25,
    discard_ratio: float = 0.1,
    mode: BlendMode | str = BlendMode.CONSTANT,
    device: torch.device | str | None = None,
    *args,
    **kwargs,
):
    """
    3D対応のsliding_window_inferenceをベースに、推論マップの端をdiscard_ratioで指定した割合だけ破棄する。
    """
    if isinstance(discard_ratio, float):
        discard_ratio = [discard_ratio] * 3

    assert len(discard_ratio) == 3, "discard_ratio must be a float or a list of 3 floats."

    # roi_weight_mapを作成（中央1.0, 端0.0）
    if not isinstance(roi_size, tuple):
        roi_size = tuple(roi_size)
    roi_weight_map = torch.ones(roi_size, dtype=torch.float32)
    discard_pixels = [int(s * d) for s, d in zip(roi_size, discard_ratio)]
    if discard_pixels[0] != 0:
        roi_weight_map[: discard_pixels[0], :, :] = 0.0
        roi_weight_map[-discard_pixels[0] :, :, :] = 0.0
    if discard_pixels[1] != 0:
        roi_weight_map[:, : discard_pixels[1], :] = 0.0
        roi_weight_map[:, -discard_pixels[1] :, :] = 0.0
    if discard_pixels[2] != 0:
        roi_weight_map[:, :, : discard_pixels[2]] = 0.0
        roi_weight_map[:, :, -discard_pixels[2] :] = 0.0

    # 入力をdiscard_ratioの分だけパディング
    padding = [int(s * d * 2) for s, d in zip(roi_size[::-1], discard_ratio[::-1]) for _ in (1, 1)]
    inputs = torch.nn.functional.pad(inputs, padding)

    outputs = sliding_window_inference(
        inputs=inputs,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
        mode=mode,
        roi_weight_map=roi_weight_map,
        device=device,
        *args,
        **kwargs,
    )

    # 結果からパディングを除去
    if isinstance(outputs, dict):
        for key in outputs.keys():
            if padding[0] != 0:
                outputs[key] = outputs[key][:, :, :, :, padding[0] : -padding[0]]
            if padding[2] != 0:
                outputs[key] = outputs[key][:, :, :, padding[2] : -padding[2], :]
            if padding[4] != 0:
                outputs[key] = outputs[key][:, :, padding[4] : -padding[4], :, :]
    else:
        if padding[0] != 0:
            outputs = outputs[:, :, :, :, padding[0] : -padding[0]]
        if padding[2] != 0:
            outputs = outputs[:, :, :, padding[2] : -padding[2], :]
        if padding[4] != 0:
            outputs = outputs[:, :, padding[4] : -padding[4], :, :]
        # outputs = outputs[:, :, padding[4] : -padding[4], padding[2] : -padding[2], padding[0] : -padding[0]]

    return outputs


# from monai.data.utils import compute_importance_map

# roi_weight_map = compute_importance_map(
#     patch_size=(64, 64, 64),
#     mode="gaussian",  # 端が小さく中心が大きい
# )
# result = sliding_window_inference(
#     inputs=image,
#     roi_size=(64, 64, 64),
#     sw_batch_size=1,
#     predictor=my_model,
#     overlap=0.5,
#     mode="gaussian",
#     roi_weight_map=roi_weight_map,
# )
