import torch
import numpy as np
from typing import Tuple, List


def calculate_overlap_steps(
    total_size: int, window_size: int, discard_size: int, valid_size: int
) -> Tuple[int, List[int]]:
    """
    破棄サイズを考慮してステップサイズとスタート位置のリストを計算
    """
    step_size = (window_size // 2 - discard_size) * 2

    start_positions = []
    current_pos = 0

    while current_pos + valid_size <= total_size:
        start_positions.append(current_pos)
        current_pos += step_size

    if start_positions[-1] + valid_size < total_size:
        start_positions.append(total_size - valid_size)

    return step_size, start_positions


def create_valid_region_mask(
    window_size: Tuple[int, int, int],
    discard_size: Tuple[int, int, int],
    n_channels: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    パッチの有効領域（境界を除いた部分）のマスクを作成
    """
    z_window, y_window, x_window = window_size
    z_overlap, y_overlap, x_overlap = discard_size

    # チャネル次元を含むマスク作成 (C, Z, Y, X)
    mask = torch.ones((n_channels, z_window, y_window, x_window), device=device)

    # 境界部分を0に設定
    mask[:, :z_overlap, :, :] = 0  # 前面
    mask[:, -z_overlap:, :, :] = 0  # 後面
    mask[:, :, :y_overlap, :] = 0  # 上部
    mask[:, :, -y_overlap:, :] = 0  # 下部
    mask[:, :, :, :x_overlap] = 0  # 左側
    mask[:, :, :, -x_overlap:] = 0  # 右側

    return mask


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    window_size: Tuple[int, int, int] = (64, 128, 128),
    discard_ratio: Tuple[float, float, float] = (0.15, 0.15, 0.15),
    c_out: int = 6,
    batch_size: int = 1,
    device: str = "cuda",
    output_dict_key: str = "classmap",
) -> torch.Tensor:
    """
    3Dデータに対してオーバーラップ付きスライディングウィンドウ推論を実行
    オーバーラップした境界部分は破棄し、有効領域のみを使用

    Args:
        model: セグメンテーションモデル
        input_tensor: 入力テンソル (B, C, Z, Y, X)
        window_size: ウィンドウサイズ (z, y, x)
        discard_ratio: 各軸の破棄する割合
        batch_size: バッチサイズ
        device: 推論実行デバイス

    Returns:
        推論結果の統合テンソル (B, C, Z, Y, X)
    """
    input_tensor = input_tensor.to(device)

    B, _, Z, Y, X = input_tensor.shape
    z_window, y_window, x_window = window_size

    # 出力テンソルを初期化（バッチとチャネル次元を含む）
    output = torch.zeros((B, c_out, Z, Y, X), device=device)
    count_matrix = torch.zeros((B, c_out, Z, Y, X), device=device)

    # 破棄サイズを計算
    discard_size = [int(size * ratio) for size, ratio in zip(window_size, discard_ratio)]

    # 破棄サイズの分、入力をパディング
    input_tensor = torch.nn.functional.pad(
        input_tensor,
        (
            discard_size[-1],
            discard_size[-1],
            discard_size[-2],
            discard_size[-2],
            discard_size[-3],
            discard_size[-3],
        ),
        mode="constant",
    )

    # 有効領域サイズを計算
    valid_region_size = [size - 2 * discard for size, discard in zip(window_size, discard_size)]

    # 各軸のステップサイズとスタート位置を計算
    z_step, z_starts = calculate_overlap_steps(Z, z_window, discard_size[0], valid_region_size[0])
    y_step, y_starts = calculate_overlap_steps(Y, y_window, discard_size[1], valid_region_size[1])
    x_step, x_starts = calculate_overlap_steps(X, x_window, discard_size[2], valid_region_size[2])

    for b in range(B):  # バッチ内の各サンプルに対して処理
        # バッチ処理用のリスト
        batch_patches = []
        batch_positions = []

        # スライディングウィンドウで処理
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    # パッチを抽出 (C, z, y, x)
                    patch = input_tensor[b, :, z : z + z_window, y : y + y_window, x : x + x_window]

                    batch_patches.append(patch)
                    batch_positions.append((z, y, x))

                    # バッチサイズに達したら推論実行
                    if len(batch_patches) == batch_size:
                        # バッチ処理
                        batch_tensor = torch.stack(batch_patches)
                        batch_output = model(batch_tensor)
                        if output_dict_key is not None:
                            batch_output = batch_output[output_dict_key]

                        # 結果を統合（有効領域のみ）
                        for patch_output, (z_pos, y_pos, x_pos) in zip(batch_output, batch_positions):
                            patch_output_valid = patch_output[
                                :,
                                discard_size[0] : -discard_size[0],
                                discard_size[1] : -discard_size[1],
                                discard_size[2] : -discard_size[2],
                            ]

                            output[
                                b,
                                :,
                                z_pos : z_pos + valid_region_size[0],
                                y_pos : y_pos + valid_region_size[1],
                                x_pos : x_pos + valid_region_size[2],
                            ] += patch_output_valid

                            count_matrix[
                                b,
                                :,
                                z_pos : z_pos + valid_region_size[0],
                                y_pos : y_pos + valid_region_size[1],
                                x_pos : x_pos + valid_region_size[2],
                            ] += 1

                        # バッチをクリア
                        batch_patches = []
                        batch_positions = []

        # 残りのパッチを処理
        if batch_patches:
            # バッチ処理
            batch_tensor = torch.stack(batch_patches)
            batch_output = model(batch_tensor)
            if output_dict_key is not None:
                batch_output = batch_output[output_dict_key]

            # 結果を統合（有効領域のみ）
            for patch_output, (z_pos, y_pos, x_pos) in zip(batch_output, batch_positions):
                patch_output_valid = patch_output[
                    :,
                    discard_size[0] : -discard_size[0],
                    discard_size[1] : -discard_size[1],
                    discard_size[2] : -discard_size[2],
                ]

                output[
                    b,
                    :,
                    z_pos : z_pos + valid_region_size[0],
                    y_pos : y_pos + valid_region_size[1],
                    x_pos : x_pos + valid_region_size[2],
                ] += patch_output_valid

                count_matrix[
                    b,
                    :,
                    z_pos : z_pos + valid_region_size[0],
                    y_pos : y_pos + valid_region_size[1],
                    x_pos : x_pos + valid_region_size[2],
                ] += 1

    assert (count_matrix == 0).sum() == 0, "Some regions are not covered by any patch."
    # 平均を計算して最終結果を得る
    final_output = output / count_matrix

    return final_output


if __name__ == "__main__":
    # ダミーデータの生成
    input_tensor = torch.randn(1, 1, 184, 630, 630).cuda()

    # モデルの定義
    model = torch.nn.Conv3d(1, 6, 3, padding=1).cuda()

    # スライディングウィンドウ推論
    output = sliding_window_inference(model, input_tensor, batch_size=4, discard_ratio=(0.4, 0.4, 0.4))
    print(output.shape)  # torch.Size([2, 6, 128, 256, 256])
