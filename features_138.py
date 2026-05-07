"""
features_holistic_138.py
=========================
從 wlasl_tsl_subset 影片中提取 138 維特徵。

特徵組成 (138 維)：
  左手  63 維  (21 點 × 3，混合歸一化)
  右手  63 維  (21 點 × 3，混合歸一化)
  左肩   3 維  (相對肩膀中心，以肩距縮放)
  右肩   3 維  (相對肩膀中心，以肩距縮放)
  鼻子   3 維  (相對肩膀中心，以肩距縮放)
  下巴   3 維  (相對肩膀中心，以肩距縮放)
  合計 138 維

Pipeline (與 features_asl_66.py 相同品質)：
  Phase 1 — HolisticLandmarker 逐幀提取原始座標 (144 維暫存)
  Phase 2 — 時序優化：空間過濾 → 插值補幀 → Savitzky-Golay 平滑
  Phase 3 — 混合歸一化後輸出 .npy
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import urllib.request
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
)
from scipy.signal import savgol_filter

# =============================================================================
# 路徑設定
# =============================================================================
VIDEO_SOURCE = "tsl"        
DATA_PATH = "tsl_features_138"   
VIZ_PATH = "tsl_tracking_138" 
MODEL_DIR = "models"
HOLISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "holistic_landmarker.task")

# 手部骨架連線（視覺化用）
HAND_CONNS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# =============================================================================
# 1. 下載模型
# =============================================================================
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print("正在下載 MediaPipe Holistic 模型...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "holistic_landmarker/holistic_landmarker/float16/"
               "latest/holistic_landmarker.task")
        try:
            urllib.request.urlretrieve(url, HOLISTIC_MODEL_PATH)
            print("下載成功。")
        except Exception as e:
            print(f"下載失敗: {e}")
    else:
        print("模型已存在。")

# =============================================================================
# 2. 混合歸一化（手部）
#    手腕(0) → 相對肩膀中心的位置
#    其餘20點 → 相對手腕，以手腕→中指根距離縮放
# =============================================================================
def normalize_hand_local(lm_list, body_center=None, body_dist=1.0):
    if not lm_list:
        return np.zeros(63)
    points = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])
    if np.all(points == 0):
        return np.zeros(63)
    hand_scale = np.linalg.norm(points[0] - points[9])
    if hand_scale < 1e-6:
        hand_scale = 1.0
    res = np.zeros_like(points)
    if body_center is not None:
        res[0] = (points[0] - body_center) / body_dist
    else:
        res[0] = 0.0
    res[1:] = (points[1:] - points[0]) / hand_scale
    return res.flatten()

# =============================================================================
# 3. 提取輔助點原始座標 (4 點：鼻子, 下巴, 左肩, 右肩)
#    順序固定，方便時序過濾取中線
# =============================================================================
def get_extra_features_raw(pose_lm, face_lm):
    """返回 (4, 3)：鼻子(Pose 0), 下巴(Face 152), 左肩(Pose 11), 右肩(Pose 12)"""
    targets = [(pose_lm, 0), (face_lm, 152), (pose_lm, 11), (pose_lm, 12)]
    pts = []
    for lm_set, idx in targets:
        if lm_set and len(lm_set) > idx:
            pts.append([lm_set[idx].x, lm_set[idx].y, lm_set[idx].z])
        else:
            pts.append([0.0, 0.0, 0.0])
    return np.array(pts)

# =============================================================================
# 4. 時序優化
#    原始資料格式 (N, 144)：
#      [左手 0:63, 右手 63:126, 鼻子 126:129, 下巴 129:132, 左肩 132:135, 右肩 135:138, 肩距 138:144 (暫存)]
#    → 實際只對前 138 維做過濾/插值/平滑，肩膀基準獨立平滑
# =============================================================================
def robust_temporal_processing(data_array):
    N, D = data_array.shape
    if N < 11:
        return data_array

    processed = data_array.copy()
    # 左手、右手、輔助點各自處理
    groups = [(0, 63), (63, 126), (126, 138)]

    # 鼻子 x 作中線基準
    nose_x_series = processed[:, 126]
    valid_nose    = nose_x_series[nose_x_series != 0]
    midline_x     = np.median(valid_nose) if len(valid_nose) > 0 else 0.5

    for start, end in groups:
        is_left_hand  = (start == 0)
        is_right_hand = (start == 63)

        group_data = processed[:, start:end]
        has_data   = np.any(group_data != 0, axis=1)

        diff   = np.diff(has_data.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        segs   = [list(p) for p in zip(starts, ends)]

        valid_segs = []
        for s, e in segs:
            seg_len = e - s
            wrist_x = np.mean(processed[s:e, start])
            if is_left_hand  and wrist_x > midline_x + 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0; continue
            if is_right_hand and wrist_x < midline_x - 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0; continue
            if seg_len < 3:
                processed[s:e, start:end] = 0; continue
            valid_segs.append([s, e])

        if not valid_segs:
            continue

        for i in range(len(valid_segs) - 1):
            e1, s2 = valid_segs[i][1], valid_segs[i+1][0]
            dist   = np.linalg.norm(processed[e1-1, start:start+3] - processed[s2, start:start+3])
            if dist < 0.4 and (s2 - e1) <= 5:
                indices = np.arange(e1, s2)
                for d in range(start, end):
                    processed[indices, d] = np.interp(
                        indices, [e1-1, s2],
                        [processed[e1-1, d], processed[s2, d]]
                    )
                valid_segs[i+1][0] = valid_segs[i][0]
                valid_segs[i] = None

        valid_segs = [sg for sg in valid_segs if sg is not None]

        for s, e in valid_segs:
            sub = processed[s:e, start:end]
            if len(sub) < 3:
                continue
            pad  = 10
            temp = np.zeros(((e-s) + 2*pad, end-start))
            temp[pad:pad+(e-s)] = sub
            temp[:pad]          = sub[0]
            temp[pad+(e-s):]    = sub[-1]
            win = min(3, temp.shape[0] if temp.shape[0] % 2 != 0 else temp.shape[0]-1)
            if win >= 3:
                temp = savgol_filter(temp, window_length=win, polyorder=2, axis=0)
            processed[s:e, start:end] = temp[pad:pad+(e-s)]

    processed[np.abs(processed) < 1e-5] = 0
    return processed

# =============================================================================
# 5. 處理單一影片
# =============================================================================
def process_video(video_path, save_path, viz_save_path, holistic_options):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, 0, 0, 0

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(viz_save_path, fourcc, fps, (width, height))

    # 暫存：左手63 + 右手63 + 鼻3 + 下巴3 + 左肩3 + 右肩3 = 138 維
    raw_frames_data = []
    shoulder_params = []
    frame_idx = 0
    first_hl_idx, last_hl_idx, active_detected_count = -1, -1, 0

    # ------------------------------------------------------------------
    # Phase 1：逐幀提取原始座標
    # ------------------------------------------------------------------
    with HolisticLandmarker.create_from_options(holistic_options) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(frame_idx * 1000 / fps)
            results      = holistic.detect_for_video(mp_image, timestamp_ms)

            pose_lm = results.pose_landmarks       if results.pose_landmarks       else None
            lh_lm   = results.left_hand_landmarks  if results.left_hand_landmarks  else None
            rh_lm   = results.right_hand_landmarks if results.right_hand_landmarks else None
            face_lm = results.face_landmarks        if results.face_landmarks        else None

            # 記錄肩膀基準
            shoulder_info = {"center": np.array([0.5, 0.5, 0.0]), "dist": 1.0}
            if pose_lm and len(pose_lm) > 12:
                l_sh = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
                r_sh = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
                shoulder_info["center"] = (l_sh + r_sh) / 2
                shoulder_info["dist"]   = max(1e-6, np.linalg.norm(l_sh - r_sh))
            shoulder_params.append(shoulder_info)

            # 原始座標：左手63 + 右手63 + [鼻,下巴,左肩,右肩]×3 = 138維
            lh_raw    = np.array([[lm.x, lm.y, lm.z] for lm in lh_lm]).flatten() if lh_lm else np.zeros(63)
            rh_raw    = np.array([[lm.x, lm.y, lm.z] for lm in rh_lm]).flatten() if rh_lm else np.zeros(63)
            extra_raw = get_extra_features_raw(pose_lm, face_lm).flatten()  # (4,3)→12維；順序：鼻,下巴,左肩,右肩

            # 存為 138 維（鼻3+下巴3+左肩3+右肩3 = 12維）
            raw_frames_data.append(np.concatenate([lh_raw, rh_raw, extra_raw]))

            if lh_lm or rh_lm:
                active_detected_count += 1
                if first_hl_idx == -1:
                    first_hl_idx = frame_idx
                last_hl_idx = frame_idx
            frame_idx += 1

    if not raw_frames_data:
        cap.release(); out.release()
        return False, 0, 0, 0

    # ------------------------------------------------------------------
    # Phase 2：時序優化
    # ------------------------------------------------------------------
    raw_frames_data  = np.array(raw_frames_data)          # (N, 138)
    refined_raw_data = robust_temporal_processing(raw_frames_data)

    # 平滑肩膀基準點
    if len(shoulder_params) > 11:
        s_centers = np.array([p["center"] for p in shoulder_params])
        s_dists   = np.array([p["dist"]   for p in shoulder_params])
        s_centers = savgol_filter(s_centers, window_length=11, polyorder=2, axis=0)
        s_dists   = savgol_filter(s_dists,   window_length=11, polyorder=2, axis=0)
        for i in range(len(shoulder_params)):
            shoulder_params[i]["center"] = s_centers[i]
            shoulder_params[i]["dist"]   = max(1e-6, s_dists[i])

    # ------------------------------------------------------------------
    # Phase 3：歸一化 + 視覺化
    # ------------------------------------------------------------------
    final_features = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    class MockLM:
        def __init__(self, p): self.x, self.y, self.z = p[0], p[1], p[2]

    for i in range(len(refined_raw_data)):
        success, frame = cap.read()
        if not success:
            break

        raw_feat = refined_raw_data[i]
        info     = shoulder_params[i]
        center   = info["center"]
        dist     = info["dist"]

        # 拆解原始 138 維
        lh_pts   = raw_feat[0:63].reshape(21, 3)
        rh_pts   = raw_feat[63:126].reshape(21, 3)
        # 順序：鼻(126:129), 下巴(129:132), 左肩(132:135), 右肩(135:138)
        nose_pt  = raw_feat[126:129]
        chin_pt  = raw_feat[129:132]
        lsh_pt   = raw_feat[132:135]
        rsh_pt   = raw_feat[135:138]

        # A. 手部：混合歸一化 (63 維 × 2)
        lh_norm = normalize_hand_local([MockLM(p) for p in lh_pts], center, dist)
        rh_norm = normalize_hand_local([MockLM(p) for p in rh_pts], center, dist)

        # B. 輔助點：相對肩膀中心歸一化，以肩距縮放 (3 維 × 4)
        def norm_pt(pt):
            return (pt - center) / dist if np.any(pt) else np.zeros(3)

        nose_norm = norm_pt(nose_pt)
        chin_norm = norm_pt(chin_pt)
        lsh_norm  = norm_pt(lsh_pt)
        rsh_norm  = norm_pt(rsh_pt)

        # C. 串接 → 138 維
        # 順序：左手63 + 右手63 + 左肩3 + 右肩3 + 鼻3 + 下巴3
        feat = np.concatenate([lh_norm, rh_norm, lsh_norm, rsh_norm, nose_norm, chin_norm])
        final_features.append(feat)

        # D. 視覺化
        h, w, _ = frame.shape
        def to_px(p): return (int(p[0] * w), int(p[1] * h))

        for pts, color in [(lh_pts, (0, 255, 0)), (rh_pts, (0, 0, 255))]:
            if np.any(pts):
                for s_c, e_c in HAND_CONNS:
                    cv2.line(frame, to_px(pts[s_c]), to_px(pts[e_c]), color, 2)
                for pt in pts:
                    cv2.circle(frame, to_px(pt), 4, (255, 255, 255), -1)

        # 鼻子 (黃)、下巴 (橘)
        if np.any(nose_pt): cv2.circle(frame, to_px(nose_pt), 5, (0, 255, 255), -1)
        if np.any(chin_pt): cv2.circle(frame, to_px(chin_pt), 5, (0, 165, 255), -1)
        # 肩膀連線 (青)
        if np.any(lsh_pt) or np.any(rsh_pt):
            cv2.circle(frame, to_px(lsh_pt), 6, (255, 255, 0), -1)
            cv2.circle(frame, to_px(rsh_pt), 6, (255, 255, 0), -1)
            cv2.line(frame, to_px(lsh_pt), to_px(rsh_pt), (255, 255, 0), 2)

        out.write(frame)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(final_features))

    cap.release()
    out.release()

    final_rate = (
        (active_detected_count / (last_hl_idx - first_hl_idx + 1)) * 100
        if first_hl_idx != -1 else 0
    )
    return True, (last_hl_idx - first_hl_idx + 1), active_detected_count, final_rate

# =============================================================================
# 6. 主程式
# =============================================================================
def main():
    download_models()

    holistic_options = HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    print("=== 開始提取 138 維特徵 (HolisticLandmarker + 混合歸一化 + 時序優化) ===")
    print("    特徵組成：左手63 + 右手63 + 左肩3 + 右肩3 + 鼻子3 + 下巴3 = 138 維")
    report_lines = [
        "# 138維特徵提取偵測報告\n",
        "| 分類 | 影片名稱 | 偵測率 |",
        "| --- | --- | --- |"
    ]

    for root, _, files in os.walk(VIDEO_SOURCE):
        category = os.path.basename(root)
        if category in (os.path.basename(VIDEO_SOURCE), ""):
            category = "未分類"

        for file in sorted(files):
            if not file.lower().endswith((".mp4", ".mkv", ".mov")):
                continue

            v_path   = os.path.join(root, file)
            rel_path = os.path.relpath(v_path, VIDEO_SOURCE)
            s_path   = os.path.join(DATA_PATH, os.path.splitext(rel_path)[0] + ".npy")
            viz_path = os.path.join(VIZ_PATH,  os.path.splitext(rel_path)[0] + ".mp4")

            if os.path.exists(s_path):
                print(f"  [略過] {file} (已存在)")
                continue

            print(f"  [處理中] {file}...", end="", flush=True)
            success, tot_f, det_f, rate = process_video(
                v_path, s_path, viz_path, holistic_options
            )

            if success:
                print(f" [OK] 偵測率 {rate:.1f}%")
                report_lines.append(f"| {category} | {file} | {rate:.1f}% |")
            else:
                print(" [失敗]")
                report_lines.append(f"| {category} | {file} | 失敗 |")

    with open("detection_report_double_norm.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("\n=== 處理完成 ===")


if __name__ == "__main__":
    main()
