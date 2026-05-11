import cv2
import numpy as np
import os
import mediapipe as mp
import urllib.request
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    HolisticLandmarker, HolisticLandmarkerOptions,
    RunningMode, PoseLandmarksConnections, HandLandmarksConnections, drawing_utils
)
from scipy.signal import savgol_filter
import csv

# --- 設定路徑 ---
VIDEO_SOURCE = "tsl"        
DATA_PATH = "tsl_features_68"   # 更新路徑
VIZ_PATH = "tsl_tracking_68"    # 更新視覺化路徑
MODEL_DIR = "models"
HOLISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "holistic_landmarker.task")

# --- 1. 下載模型函式 ---
def download_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print("正在下載 MediaPipe Holistic 模型...")
        url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
        try:
            urllib.request.urlretrieve(url, HOLISTIC_MODEL_PATH)
            print("下載成功。")
        except Exception as e:
            print(f"下載失敗: {e}")
    else:
        print("模型已存在。")

# --- 2. 幾何特徵提取 ---
def extract_geometric_features(hand_norm_63):
    """
    從歸一化後的 63 維手部點位中提取幾何特徵
    hand_norm_63: (63,) 平鋪陣列，其中 [0:3] 是手腕(相對身體)，其餘是相對手腕且縮放過的座標
    """
    if np.all(hand_norm_63 == 0):
        return np.zeros(1)
    
    pts = hand_norm_63.reshape(21, 3)
    
    # 1. 拇指-食指尖距離 (TI-ED)
    # 由於 pts[4] 和 pts[8] 都是相對手腕的向量，其差值即為兩指尖位移
    ti_ed = np.linalg.norm(pts[4] - pts[8])
    
    return np.array([ti_ed])

# --- 3. 混合歸一化 ---
def normalize_hand_local(lm_list, body_center=None, body_dist=1.0):
    if not lm_list:
        return np.zeros(63)
    
    points = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])
    if np.all(points == 0):
        return np.zeros(63)
    
    # A. 計算局部手部縮放比例 (手腕到中指根部)
    hand_scale = np.linalg.norm(points[0] - points[9])
    if hand_scale < 1e-6: hand_scale = 1.0 
    
    res = np.zeros_like(points)
    
    # B. 處理「位置資訊」：手腕相對於身體中心
    if body_center is not None:
        res[0] = (points[0] - body_center) / body_dist
    else:
        res[0] = 0.0
        
    # C. 處理「手型資訊」：其餘點相對於手腕，並歸一化
    res[1:] = (points[1:] - points[0]) / hand_scale
    
    return res.flatten()

def get_extra_features_raw(pose_lm, face_lm):
    pts = []
    targets = [(pose_lm, 0), (face_lm, 152), (pose_lm, 11), (pose_lm, 12)]
    for lm_set, idx in targets:
        if lm_set and len(lm_set) > idx:
            pts.append([lm_set[idx].x, lm_set[idx].y, lm_set[idx].z])
        else:
            pts.append([0.0, 0.0, 0.0])
    return np.array(pts)

def robust_temporal_processing(data_array):
    """ 
    時序處理邏輯維持不變，但注意維度變為 138 (原始採樣維度)
    """
    N, D = data_array.shape
    if N < 11: return data_array 

    processed = data_array.copy()
    groups = [(0, 63), (63, 126), (126, 138)]
    
    nose_x_series = processed[:, 126]
    valid_nose = nose_x_series[nose_x_series != 0]
    midline_x = np.median(valid_nose) if len(valid_nose) > 0 else 0.5

    for start, end in groups:
        is_left_hand = (start == 0)
        is_right_hand = (start == 63)
        group_data = processed[:, start:end]
        has_data = np.any(group_data != 0, axis=1)
        diff = np.diff(has_data.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        segs = [list(i) for i in zip(starts, ends)]

        valid_segs = []
        for s, e in segs:
            seg_len = e - s
            wrist_x = np.mean(processed[s:e, start])
            if is_left_hand and wrist_x > midline_x + 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0; continue
            if is_right_hand and wrist_x < midline_x - 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0; continue
            if seg_len < 3:
                processed[s:e, start:end] = 0; continue
            valid_segs.append([s, e])

        if not valid_segs: continue

        for i in range(len(valid_segs) - 1):
            e1, s2 = valid_segs[i][1], valid_segs[i+1][0]
            p1 = processed[e1-1, start:start+3]
            p2 = processed[s2, start:start+3]
            dist = np.linalg.norm(p1 - p2)
            if dist < 0.4 and (s2 - e1) <= 5:
                indices = np.arange(e1, s2)
                for d in range(start, end):
                    processed[indices, d] = np.interp(indices, [e1-1, s2], [processed[e1-1, d], processed[s2, d]])
                valid_segs[i+1][0] = valid_segs[i][0]
                valid_segs[i] = None
        
        valid_segs = [seg for seg in valid_segs if seg is not None]

        for s, e in valid_segs:
            sub = processed[s:e, start:end]
            if len(sub) < 3: continue 
            pad = 10
            temp = np.zeros(((e-s) + 2*pad, end-start))
            temp[pad:pad+(e-s)] = sub
            temp[:pad] = sub[0]
            temp[pad+(e-s):] = sub[-1]
            win = min(3, temp.shape[0] if temp.shape[0]%2!=0 else temp.shape[0]-1)
            if win >= 3:
                temp = savgol_filter(temp, window_length=win, polyorder=2, axis=0)
            processed[s:e, start:end] = temp[pad:pad+(e-s)]

    processed[np.abs(processed) < 1e-5] = 0
    return processed

def process_video(video_path, save_path, viz_save_path, holistic_options):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(viz_save_path, fourcc, fps, (width, height))

    HAND_CONNS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]

    raw_frames_data = [] 
    shoulder_params = [] 
    frame_idx = 0
    first_hl_idx, last_hl_idx, active_detected_count = -1, -1, 0

    # Phase 1: 提取原始點位 (138維)
    with HolisticLandmarker.create_from_options(holistic_options) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(frame_idx * 1000 / fps)
            results = holistic.detect_for_video(mp_image, timestamp_ms)

            pose_lm = results.pose_landmarks if results.pose_landmarks else None
            lh_lm = results.left_hand_landmarks if results.left_hand_landmarks else None
            rh_lm = results.right_hand_landmarks if results.right_hand_landmarks else None
            face_lm = results.face_landmarks if results.face_landmarks else None

            shoulder_info = {"center": np.array([0.5, 0.5, 0.0]), "dist": 1.0}
            if pose_lm and len(pose_lm) > 12:
                l_sh = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
                r_sh = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
                shoulder_info["center"] = (l_sh + r_sh) / 2
                shoulder_info["dist"] = max(1e-6, np.linalg.norm(l_sh - r_sh))
            shoulder_params.append(shoulder_info)

            lh_raw = np.array([[lm.x, lm.y, lm.z] for lm in lh_lm]).flatten() if lh_lm else np.zeros(63)
            rh_raw = np.array([[lm.x, lm.y, lm.z] for lm in rh_lm]).flatten() if rh_lm else np.zeros(63)
            extra_pts = get_extra_features_raw(pose_lm, face_lm).flatten() 
            
            raw_frames_data.append(np.concatenate([lh_raw, rh_raw, extra_pts]))

            if lh_lm or rh_lm:
                active_detected_count += 1
                if first_hl_idx == -1: first_hl_idx = frame_idx
                last_hl_idx = frame_idx
            frame_idx += 1

    if not raw_frames_data:
        cap.release(); out.release(); return False, 0, 0, 0

    # Phase 2: 時序優化
    raw_frames_data = np.array(raw_frames_data)
    refined_raw_data = robust_temporal_processing(raw_frames_data)

    if len(shoulder_params) > 11:
        s_centers = np.array([p["center"] for p in shoulder_params])
        s_dists = np.array([p["dist"] for p in shoulder_params])
        s_centers = savgol_filter(s_centers, window_length=11, polyorder=2, axis=0)
        s_dists = savgol_filter(s_dists, window_length=11, polyorder=2, axis=0)
        for i in range(len(shoulder_params)):
            shoulder_params[i]["center"] = s_centers[i]
            shoulder_params[i]["dist"] = max(1e-6, s_dists[i])

    # Phase 3: 歸一化 + 提取幾何特徵
    final_features_72 = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    class MockLM: 
        def __init__(self, p): self.x, self.y, self.z = p[0], p[1], p[2]

    for i in range(len(refined_raw_data)):
        success, frame = cap.read()
        if not success: break
        
        raw_feat = refined_raw_data[i]
        info = shoulder_params[i]
        
        lh_pts = raw_feat[0:63].reshape(21, 3)
        rh_pts = raw_feat[63:126].reshape(21, 3)
        extra_pts = raw_feat[126:138].reshape(4, 3) 
        
        # 1. 執行完整的 63 維歸一化 (為了提取幾何特徵)
        lh_norm_63 = normalize_hand_local([MockLM(p) for p in lh_pts], info["center"], info["dist"])
        rh_norm_63 = normalize_hand_local([MockLM(p) for p in rh_pts], info["center"], info["dist"])
        
        # 2. 提取幾何特徵 (每隻手 1 維：僅 TI-ED)
        lh_geo = extract_geometric_features(lh_norm_63)
        rh_geo = extract_geometric_features(rh_norm_63)
        
        # 3. 保留原始的 11 個關鍵點 (33維)
        keep_points = [0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
        keep_indices = []
        for p in keep_points:
            keep_indices.extend([p*3, p*3+1, p*3+2])
            
        lh_pruned = lh_norm_63[keep_indices]
        rh_pruned = rh_norm_63[keep_indices]
        
        # 4. 串接：(33 + 1) + (33 + 1) = 68 維
        feat_68 = np.concatenate([lh_pruned, lh_geo, rh_pruned, rh_geo])
        final_features_72.append(feat_68)

        # 視覺化繪製 (略...)
        h, w, _ = frame.shape
        def to_px(p): return (int(p[0] * w), int(p[1] * h))
        for pts, color in [(lh_pts, (0, 255, 0)), (rh_pts, (0, 0, 255))]:
            if np.any(pts):
                for start, end in HAND_CONNS:
                    cv2.line(frame, to_px(pts[start]), to_px(pts[end]), color, 2)
                for pt in pts:
                    cv2.circle(frame, to_px(pt), 4, (255, 255, 255), -1)
        out.write(frame)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(final_features_72))
    cap.release(); out.release()
    
    final_rate = (active_detected_count / (last_hl_idx - first_hl_idx + 1)) * 100 if first_hl_idx != -1 else 0
    return True, (last_hl_idx - first_hl_idx + 1), active_detected_count, final_rate

def main():
    download_models()
    holistic_options = HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )
    os.makedirs(DATA_PATH, exist_ok=True)
    report_path = os.path.join(DATA_PATH, "detection_rates_72.csv")
    write_header = not os.path.exists(report_path)
    
    report_lines = ["# 68維特徵提取偵測報告 (66D座標 + 2D幾何)\n", "| 分類 | 影片名稱 | 偵測率 |", "| --- | --- | --- |"]
    
    print(f"=== 開始提取 68 維特徵 (66維座標 + 2維幾何特徵) ===")
    
    with open(report_path, mode='a', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(['Video', 'DetectionRate', 'TotalFrames', 'DetectedFrames'])
            
        for root, _, files in os.walk(VIDEO_SOURCE):
            category = os.path.basename(root)
            if category == os.path.basename(VIDEO_SOURCE) or category == "": category = "未分類"

            for file in files:
                if file.lower().endswith(('.mp4', '.mkv')):
                    v_path = os.path.join(root, file)
                    rel_path = os.path.relpath(v_path, VIDEO_SOURCE)
                    s_path = os.path.join(DATA_PATH, os.path.splitext(rel_path)[0] + ".npy")
                    v_save_path = os.path.join(VIZ_PATH, os.path.splitext(rel_path)[0] + ".mp4")
                    
                    if os.path.exists(s_path): continue
                    
                    print(f"  [處理中] {file}...", end="", flush=True)
                    success, tot_f, det_f, rate = process_video(v_path, s_path, v_save_path, holistic_options)
                    
                    if success: 
                        print(f" [OK] {rate:.1f}%")
                        writer.writerow([file, f"{rate:.2f}%", tot_f, det_f])
                        f_csv.flush()
                        report_lines.append(f"| {category} | {file} | {rate:.1f}% |")
                    else: 
                        print(" [失敗]")
                        report_lines.append(f"| {category} | {file} | 失敗 |")
    
    with open("detection_report_68.md", "w", encoding="utf-8") as f_md:
        f_md.write("\n".join(report_lines) + "\n")
    print("\n=== 處理完成 ===")

if __name__ == "__main__":
    main()
