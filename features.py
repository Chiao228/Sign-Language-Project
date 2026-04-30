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

# --- 設定路徑 ---
VIDEO_SOURCE = "tsl"        
DATA_PATH = "tsl_features_66"   
VIZ_PATH = "tsl_tracking_66" 
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

# --- 2. 混合歸一化：手腕存「位置」，其餘存「手型」 ---
def normalize_hand_local(lm_list, body_center=None, body_dist=1.0):
    """ 
    混合歸一化策略：
    1. 手腕 (Point 0): 儲存相對於 body_center (如肩膀中心) 的位移向量，並以 body_dist 縮放。
    2. 其他點 (Point 1-20): 儲存相對於手腕的位移向量，並以手掌大小 (手腕到中指根部) 縮放。
    """
    if not lm_list:
        return np.zeros(63)
    
    # 轉為 numpy array (21, 3)
    points = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])
    
    # --- 如果所有點都是 0，直接返回 0，徹底消除幽靈座標 ---
    if np.all(points == 0):
        return np.zeros(63)
    
    # A. 計算局部手部縮放比例 (以手腕到中指根部距離為準)
    # Point 0: 手腕, Point 9: 中指根部
    hand_scale = np.linalg.norm(points[0] - points[9])
    if hand_scale < 1e-6: hand_scale = 1.0 
    
    res = np.zeros_like(points)
    
    # B. 處理「位置資訊」：手腕相對於身體中心 (肩膀/臉部)
    if body_center is not None:
        res[0] = (points[0] - body_center) / body_dist
    else:
        res[0] = 0.0 # 若無身體資訊則歸零
        
    # C. 處理「手型資訊」：其餘 20 個點相對於手腕，並用手部大小歸一化
    res[1:] = (points[1:] - points[0]) / hand_scale
    
    return res.flatten()

# --- 3. 提取全局點位 ---
def get_extra_features_raw(pose_lm, face_lm):
    """ 提取原始座標：鼻子(Pose 0), 下巴(Face 152), 左肩(Pose 11), 右肩(Pose 12) """
    pts = []
    # 順序：鼻子, 下巴, 左肩, 右肩
    targets = [(pose_lm, 0), (face_lm, 152), (pose_lm, 11), (pose_lm, 12)]
    
    for lm_set, idx in targets:
        if lm_set and len(lm_set) > idx:
            pts.append([lm_set[idx].x, lm_set[idx].y, lm_set[idx].z])
        else:
            pts.append([0.0, 0.0, 0.0])
            
    return np.array(pts) # 返回 (4, 3) 矩陣

# --- 4. 時間序列處理：內插補點與平滑濾波 ---
def robust_temporal_processing(data_array):
    """
    第五次優化版 (靈敏度微調)：
    1. 空間過濾：放寬緩衝區 (+0.25) 並縮短判斷時間，允許正當的 cross-body 動作。
    2. 位移過濾：放寬距離閾值 (0.4) 以追蹤快速移動的手勢。
    3. 去噪：將門檻降低至 3 幀，撈回極短促的有效動作。
    """
    N, D = data_array.shape
    if N < 11: return data_array 

    processed = data_array.copy()
    groups = [(0, 63), (63, 126), (126, 138)]
    
    # 取得中線基準
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
            # --- 1. 空間邏輯過濾 (放寬限制，僅過濾極短且極不合理的閃現) ---
            wrist_x = np.mean(processed[s:e, start])
            if is_left_hand and wrist_x > midline_x + 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0
                continue
            if is_right_hand and wrist_x < midline_x - 0.25 and seg_len < 5:
                processed[s:e, start:end] = 0
                continue
            
            # --- 2. 雜訊過濾 (降低至 3 幀) ---
            if seg_len < 3:
                processed[s:e, start:end] = 0
                continue
            valid_segs.append([s, e])

        if not valid_segs: continue

        # --- 3. 位移與時間約束插值 (極限壓縮區間至 5 幀) ---
        for i in range(len(valid_segs) - 1):
            e1, s2 = valid_segs[i][1], valid_segs[i+1][0]
            p1 = processed[e1-1, start:start+3]
            p2 = processed[s2, start:start+3]
            dist = np.linalg.norm(p1 - p2)
            
            # 同時滿足位移小與時間短 (<= 5幀) 才會連起來，防止手勢滯後
            if dist < 0.4 and (s2 - e1) <= 5:
                indices = np.arange(e1, s2)
                for d in range(start, end):
                    processed[indices, d] = np.interp(indices, [e1-1, s2], [processed[e1-1, d], processed[s2, d]])
                valid_segs[i+1][0] = valid_segs[i][0]
                valid_segs[i] = None
        
        valid_segs = [seg for seg in valid_segs if seg is not None]

        # --- 4. 平滑 (Edge Padding) ---
        for s, e in valid_segs:
            sub = processed[s:e, start:end]
            if len(sub) < 3: continue 
            pad = 10
            temp = np.zeros(((e-s) + 2*pad, end-start))
            temp[pad:pad+(e-s)] = sub
            temp[:pad] = sub[0]
            temp[pad+(e-s):] = sub[-1]
            
            # 將視窗極大限度縮短為 3，確保手勢變換幾乎即時同步
            win = min(3, temp.shape[0] if temp.shape[0]%2!=0 else temp.shape[0]-1)
            if win >= 3:
                temp = savgol_filter(temp, window_length=win, polyorder=2, axis=0)
            processed[s:e, start:end] = temp[pad:pad+(e-s)]

    processed[np.abs(processed) < 1e-5] = 0
    return processed

# --- 4. 處理單一影片 ---
def process_video(video_path, save_path, viz_save_path, holistic_options):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(viz_save_path, fourcc, fps, (width, height))

    # --- 自定義繪圖連線 (用於 OpenCV 繪製) ---
    HAND_CONNS = [
        (0, 1), (1, 2), (2, 3), (3, 4),             # 大拇指
        (0, 5), (5, 6), (6, 7), (7, 8),             # 食指
        (5, 9), (9, 10), (10, 11), (11, 12),        # 中指
        (9, 13), (13, 14), (14, 15), (15, 16),      # 無名指
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # 小指與手掌
    ]

    raw_frames_data = [] # 儲存每一幀的原始座標 (138維)
    shoulder_params = [] # 儲存每一幀的歸一化基準 (center, dist)
    frame_idx = 0
    first_hl_idx, last_hl_idx, active_detected_count = -1, -1, 0

    # --- 第一階段：快速提取原始點位 ---
    with HolisticLandmarker.create_from_options(holistic_options) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(frame_idx * 1000 / fps)
            results = holistic.detect_for_video(mp_image, timestamp_ms)

            # 1. 取得原始座標 (不歸一化)
            pose_lm = results.pose_landmarks if results.pose_landmarks else None
            lh_lm = results.left_hand_landmarks if results.left_hand_landmarks else None
            rh_lm = results.right_hand_landmarks if results.right_hand_landmarks else None
            face_lm = results.face_landmarks if results.face_landmarks else None

            # 記錄歸一化基準 (肩膀)
            shoulder_info = {"center": np.array([0.5, 0.5, 0.0]), "dist": 1.0}
            if pose_lm and len(pose_lm) > 12:
                l_sh = np.array([pose_lm[11].x, pose_lm[11].y, pose_lm[11].z])
                r_sh = np.array([pose_lm[12].x, pose_lm[12].y, pose_lm[12].z])
                shoulder_info["center"] = (l_sh + r_sh) / 2
                shoulder_info["dist"] = max(1e-6, np.linalg.norm(l_sh - r_sh))
            shoulder_params.append(shoulder_info)

            # 提取 138 維原始座標 (直接存入 0-1 的值)
            lh_raw = np.array([[lm.x, lm.y, lm.z] for lm in lh_lm]).flatten() if lh_lm else np.zeros(63)
            rh_raw = np.array([[lm.x, lm.y, lm.z] for lm in rh_lm]).flatten() if rh_lm else np.zeros(63)
            extra_pts = get_extra_features_raw(pose_lm, face_lm).flatten()  # 12 維 (鼻+下巴+左肩+右肩)
            
            raw_frames_data.append(np.concatenate([lh_raw, rh_raw, extra_pts]))

            if lh_lm or rh_lm:
                active_detected_count += 1
                if first_hl_idx == -1: first_hl_idx = frame_idx
                last_hl_idx = frame_idx
            frame_idx += 1

    if not raw_frames_data:
        cap.release(); out.release(); return False, 0, 0, 0

    # --- 第二階段：時序優化 (核心座標 + 基準點) ---
    raw_frames_data = np.array(raw_frames_data)
    refined_raw_data = robust_temporal_processing(raw_frames_data)

    # 平滑肩膀基準點 (防止基準點抖動導致的點位整體飄移)
    if len(shoulder_params) > 11:
        s_centers = np.array([p["center"] for p in shoulder_params])
        s_dists = np.array([p["dist"] for p in shoulder_params])
        s_centers = savgol_filter(s_centers, window_length=11, polyorder=2, axis=0)
        s_dists = savgol_filter(s_dists, window_length=11, polyorder=2, axis=0)
        for i in range(len(shoulder_params)):
            shoulder_params[i]["center"] = s_centers[i]
            shoulder_params[i]["dist"] = max(1e-6, s_dists[i])

    # --- 第三階段：重新繪製優化後的影片 與 歸一化 ---
    final_normalized_features = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 影片回到開頭
    
    for i in range(len(refined_raw_data)):
        success, frame = cap.read()
        if not success: break
        
        raw_feat = refined_raw_data[i]
        info = shoulder_params[i]
        
        # 拆解 138 維座標
        lh_pts = raw_feat[0:63].reshape(21, 3)
        rh_pts = raw_feat[63:126].reshape(21, 3)
        extra_pts = raw_feat[126:138].reshape(4, 3)  # 鼻子, 下巴, 左肩, 右肩
        
        # A. 執行歸一化 (模型用的)
        class MockLM: 
            def __init__(self, p): self.x, self.y, self.z = p[0], p[1], p[2]
        
        lh_norm = normalize_hand_local([MockLM(p) for p in lh_pts], info["center"], info["dist"])
        rh_norm = normalize_hand_local([MockLM(p) for p in rh_pts], info["center"], info["dist"])
        
        # 只保留關鍵關節 (手腕 + 拇指跟四指的末端兩節)，總共 11 個點 (33維) x 2隻手 = 66維
        # 省略 1(CMC), 2(MCP), 5(MCP), 6(PIP), 9(MCP), 10(PIP), 13(MCP), 14(PIP), 17(MCP), 18(PIP)
        keep_points = [0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
        keep_indices = []
        for p in keep_points:
            keep_indices.extend([p*3, p*3+1, p*3+2])
            
        lh_pruned = lh_norm[keep_indices]
        rh_pruned = rh_norm[keep_indices]
        
        # 捨棄 extra_norm (不將其加入特徵序列)
        final_normalized_features.append(np.concatenate([lh_pruned, rh_pruned]))

        # B. 繪製視覺化 (將優化後的點畫在影片上)
        h, w, _ = frame.shape
        def to_px(p): return (int(p[0] * w), int(p[1] * h))
        
        # 繪製左手 (綠) 與 右手 (紅)
        for pts, color in [(lh_pts, (0, 255, 0)), (rh_pts, (0, 0, 255))]:
            if np.any(pts): # 只要有點位就繪製
                for start, end in HAND_CONNS:
                    cv2.line(frame, to_px(pts[start]), to_px(pts[end]), color, 2)
                for pt in pts:
                    cv2.circle(frame, to_px(pt), 4, (255, 255, 255), -1)
        
        # 繪製面部核心點 (黃) — 鼻子與下巴
        for pt in extra_pts[:2]:
            cv2.circle(frame, to_px(pt), 5, (0, 255, 255), -1)
        
        # 繪製肩膀 (青色) — 左肩與右肩，並連線
        l_sh_pt, r_sh_pt = extra_pts[2], extra_pts[3]
        if np.any(l_sh_pt) or np.any(r_sh_pt):
            cv2.circle(frame, to_px(l_sh_pt), 6, (255, 255, 0), -1)
            cv2.circle(frame, to_px(r_sh_pt), 6, (255, 255, 0), -1)
            cv2.line(frame, to_px(l_sh_pt), to_px(r_sh_pt), (255, 255, 0), 2)

        out.write(frame)

    # 儲存優化後的 .npy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(final_normalized_features))
    
    cap.release()
    out.release()
    
    final_rate = (active_detected_count / (last_hl_idx - first_hl_idx + 1)) * 100 if first_hl_idx != -1 else 0
    return True, (last_hl_idx - first_hl_idx + 1), active_detected_count, final_rate

# --- 5. 主程式 ---
def main():
    download_models()
    holistic_options = HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )

    print(f"=== 開始提取 66 維特徵 (純座標版本 + 精簡手部關節) ===")
    report_lines = ["# 66維雙重歸一化偵測報告\n", "| 分類 | 影片名稱 | 偵測率 |", "| --- | --- | --- |"]
    
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
                    report_lines.append(f"| {category} | {file} | {rate:.1f}% |")
                else:
                    print(" [失敗]")
                    report_lines.append(f"| {category} | {file} | 失敗 |")

    with open("detection_report_double_norm.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("\n=== 處理完成 ===")

if __name__ == "__main__":
    main()