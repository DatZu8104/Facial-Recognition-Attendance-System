import pickle
import face_recognition
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings

# Tắt các cảnh báo không cần thiết của thư viện
warnings.filterwarnings("ignore")

# ================= CẤU HÌNH =================
CACHE_FILE = "encodings_cache.pkl"  # File cache từ app chính
TEST_DIR = "test_dataset"           # Thư mục chứa ảnh để test
TOLERANCE = 0.45                    # PHẢI GIỐNG app chính
# ============================================

def load_known_data():
    print(f"   [DEBUG] Đang tìm file cache tại: {os.path.abspath(CACHE_FILE)}")
    if not os.path.exists(CACHE_FILE):
        print("   [LỖI] Không tìm thấy file encodings_cache.pkl!")
        return [], [], []
    try:
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        return data["encodings"], data["ids"], data["names"]
    except Exception as e:
        print(f"   [LỖI] Không đọc được file cache: {e}")
        return [], [], []

def predict_face(image_path, known_encodings, known_names):
    """Giả lập lại logic của app chính"""
    try:
        image = face_recognition.load_image_file(image_path)
        small_frame = np.array(image) 
        
        # Tìm vị trí khuôn mặt (HOG)
        locs = face_recognition.face_locations(small_frame, model="hog")
        encs = face_recognition.face_encodings(small_frame, locs)

        if not encs:
            return "NoFace" # Không tìm thấy mặt

        # Lấy khuôn mặt đầu tiên tìm thấy
        enc = encs[0]
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
        name = "Unknown"

        if True in matches:
            # Logic lấy face có khoảng cách nhỏ nhất (Best match)
            face_distances = face_recognition.face_distance(known_encodings, enc)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        
        return name
    except Exception as e:
        print(f"   [LỖI] Lỗi khi xử lý ảnh {os.path.basename(image_path)}: {e}")
        return "Error"

def evaluate():
    print("\n>>> BƯỚC 1: ĐANG NẠP DỮ LIỆU ĐÃ HỌC...")
    known_encs, known_ids, known_names = load_known_data()
    
    if not known_encs:
        print(">>> KẾT THÚC SỚM: Không có dữ liệu khuôn mặt mẫu (File cache rỗng hoặc chưa chạy App chính).")
        return

    print(f">>> Đã nạp thành công {len(known_encs)} khuôn mặt mẫu.")

    print("\n>>> BƯỚC 2: KIỂM TRA THƯ MỤC TEST...")
    if not os.path.exists(TEST_DIR):
        print(f">>> LỖI: Không thấy thư mục '{TEST_DIR}'. Hãy tạo thư mục và copy ảnh vào.")
        return

    # Chỉ lấy file ảnh
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f">>> Tìm thấy {len(files)} ảnh trong thư mục test.")

    if len(files) == 0:
        print(">>> LỖI: Thư mục test trống rỗng!")
        return

    y_true = [] # Nhãn thực tế
    y_pred = [] # Nhãn dự đoán

    print("\n>>> BƯỚC 3: BẮT ĐẦU CHẠY ĐÁNH GIÁ TỪNG ẢNH...")
    print(f"{'TÊN FILE':<25} | {'THỰC TẾ':<10} | {'DỰ ĐOÁN':<10} | {'KẾT QUẢ'}")
    print("-" * 65)

    for file in files:
        # --- [LOGIC QUAN TRỌNG: LẤY TÊN THẬT TỪ FILE] ---
        # Quy tắc: TênFile_So.jpg -> Lấy "TênFile"
        true_name = "Unknown"
        
        if "Unknown" in file:
            true_name = "Unknown"
        elif "_" in file:
            # Lấy phần đầu tiên trước dấu gạch dưới (VD: ngoc_01.jpg -> ngoc)
            true_name = file.split('_')[0] 
        else:
            # Nếu không có gạch dưới, lấy toàn bộ tên file (VD: ngoc.jpg -> ngoc)
            true_name = file.split('.')[0]
        # ------------------------------------------------

        full_path = os.path.join(TEST_DIR, file)
        pred_name = predict_face(full_path, known_encs, known_names)
        
        if pred_name == "NoFace":
            print(f"{file:<25} | {true_name:<10} | {'NoFace':<10} | [BỎ QUA]")
            continue
        
        if pred_name == "Error":
            continue

        y_true.append(true_name)
        y_pred.append(pred_name)
        
        status = "ĐÚNG" if true_name == pred_name else "SAI"
        print(f"{file:<25} | {true_name:<10} | {pred_name:<10} | {status}")

    if not y_true:
        print("\n>>> KHÔNG CÓ KẾT QUẢ HỢP LỆ ĐỂ VẼ BIỂU ĐỒ.")
        return

    # --- BÁO CÁO KẾT QUẢ ---
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"ĐỘ CHÍNH XÁC (ACCURACY): {acc:.2%}")
    print("="*40)
    print("\nBÁO CÁO CHI TIẾT:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # --- VẼ CONFUSION MATRIX ---
    try:
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('AI Dự đoán (Predicted)')
        plt.ylabel('Thực tế (Actual)')
        plt.title(f'Ma trận nhầm lẫn (Accuracy: {acc:.2%})')
        plt.show()
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {e}")

if __name__ == "__main__":
    # Đảm bảo thư mục tồn tại
    if not os.path.exists(TEST_DIR): os.makedirs(TEST_DIR)
    evaluate()