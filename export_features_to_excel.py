import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
from scipy.signal import savgol_filter

# --- 從 features.py 延用的設定 ---
MODEL_DIR = "models"
HOLISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "holistic_landmarker.task")

# --- 混合歸一化策略 (與 features.py 一致) ---
def normalize_hand_local(lm_list, body_center=None, body_dist=1.0):
    if not lm_list:
        return np.zeros(63)
    
    points = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])
    if np.all(points == 0):
        return np.zeros(63)
    
    # Point 0: 手腕, Point 9: 中指根部
    hand_scale = np.linalg.norm(points[0] - points[9])
    if hand_scale < 1e-6: hand_scale = 1.0 
    
    res = np.zeros_like(points)
    if body_center is not None:
        res[0] = (points[0] - body_center) / body_dist
    else:
        res[0] = 0.0
        
    res[1:] = (points[1:] - points[0]) / hand_scale
    return res.flatten()

def get_extra_features_raw(pose_lm, face_lm):
    """ 提取原始座標：鼻子(Pose 0), 下巴(Face 152), 左肩(Pose 11), 右肩(Pose 12) """
    pts = []
    targets = [(pose_lm, 0), (face_lm, 152), (pose_lm, 11), (pose_lm, 12)]
    for lm_set, idx in targets:
        if lm_set and len(lm_set) > idx:
            pts.append([lm_set[idx].x, lm_set[idx].y, lm_set[idx].z])
        else:
            pts.append([0.0, 0.0, 0.0])
    return np.array(pts)

# --- 延用 features.py 的時序處理 ---
def robust_temporal_processing(data_array):
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

        for s, e in segs:
            seg_len = e - s
            wrist_x = np.mean(processed[s:e, start])
            if is_left_hand and wrist_x > midline_x + 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0
                continue
            if is_right_hand and wrist_x < midline_x - 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0
                continue
            if seg_len < 3:
                processed[s:e, start:end] = 0
                continue

    processed[np.abs(processed) < 1e-5] = 0
    return processed

def extract_and_export(video_path, excel_path):
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print(f"錯誤: 找不到模型文件 {HOLISTIC_MODEL_PATH}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法打開影片: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    raw_frames_data = []
    shoulder_params = []
    frame_idx = 0

    holistic_options = HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )

    print(f"正在分析影片: {video_path}...")
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
            extra_pts = get_extra_features_raw(pose_lm, face_lm).flatten()  # 12 維
            
            raw_frames_data.append(np.concatenate([lh_raw, rh_raw, extra_pts]))
            frame_idx += 1

    if not raw_frames_data:
        print("未偵測到特徵。")
        cap.release()
        return

    # 時序優化
    raw_frames_data = np.array(raw_frames_data)
    refined_raw_data = robust_temporal_processing(raw_frames_data)

    # 平滑肩膀基準
    if len(shoulder_params) > 11:
        s_centers = np.array([p["center"] for p in shoulder_params])
        s_dists = np.array([p["dist"] for p in shoulder_params])
        s_centers = savgol_filter(s_centers, window_length=11, polyorder=2, axis=0)
        s_dists = savgol_filter(s_dists, window_length=11, polyorder=2, axis=0)
        for i in range(len(shoulder_params)):
            shoulder_params[i]["center"] = s_centers[i]
            shoulder_params[i]["dist"] = max(1e-6, s_dists[i])

    # 執行最終歸一化
    final_features = []
    class MockLM: 
        def __init__(self, p): self.x, self.y, self.z = p[0], p[1], p[2]

    print("正在執行歸一化並導出資料...")
    for i in range(len(refined_raw_data)):
        raw_feat = refined_raw_data[i]
        info = shoulder_params[i]
        
        lh_pts = raw_feat[0:63].reshape(21, 3)
        rh_pts = raw_feat[63:126].reshape(21, 3)
        extra_pts = raw_feat[126:138].reshape(4, 3)  # 鼻子, 下巴, 左肩, 右肩
        
        lh_norm = normalize_hand_local([MockLM(p) for p in lh_pts], info["center"], info["dist"])
        rh_norm = normalize_hand_local([MockLM(p) for p in rh_pts], info["center"], info["dist"])
        extra_norm = ((extra_pts - info["center"]) / info["dist"]).flatten()
        
        final_features.append(np.concatenate([lh_norm, rh_norm, extra_norm]))

    # 準備 Excel 欄位名稱
    columns = []
    # 左手 21 點
    for i in range(21):
        for axis in ['x', 'y', 'z']:
            columns.append(f'LH_{i}_{axis}')
    # 右手 21 點
    for i in range(21):
        for axis in ['x', 'y', 'z']:
            columns.append(f'RH_{i}_{axis}')
    # 額外點位 (鼻子, 下巴, 左肩, 右肩)
    for name in ['Nose', 'Chin', 'L_Shoulder', 'R_Shoulder']:
        for axis in ['x', 'y', 'z']:
            columns.append(f'{name}_{axis}')

    df = pd.DataFrame(final_features, columns=columns)
    df.insert(0, 'Frame', range(len(df)))

    df.to_excel(excel_path, index=False)
    print(f"成功導出至: {excel_path} (138 維)")
    cap.release()

if __name__ == "__main__":
    # 使用者可以自行修改此處的路徑
    VIDEO_PATH = r"tsl\你好\你好.mp4"
    OUTPUT_EXCEL = "normalized_features.xlsx"
    
    if os.path.exists(VIDEO_PATH):
        extract_and_export(VIDEO_PATH, OUTPUT_EXCEL)
    else:
        print(f"請確認影片路徑是否存在: {VIDEO_PATH}")
        # 如果不存在，試著找一個現有的
        print("搜尋現有影片中...")
        for root, dirs, files in os.walk("tsl"):
            for f in files:
                if f.endswith(".mp4"):
                    v = os.path.join(root, f)
                    print(f"找到影片: {v}")
                    extract_and_export(v, OUTPUT_EXCEL)
                    exit()
