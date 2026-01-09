import cv2
import numpy as np
import os
import sqlite3
import threading
import pickle
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime, timedelta
import face_recognition
import warnings
import time
import mediapipe as mp

warnings.filterwarnings("ignore", category=UserWarning)

# ================= CẤU HÌNH HỆ THỐNG =================
DB_PATH = "attendance.db"
DATASET_PATH = "dataset"
CACHE_FILE = "encodings_cache.pkl"
COOLDOWN_SECONDS = 3  # [SỬA] Giảm thời gian chờ xuống 3s
known_encodings = []
known_ids = []
known_names = []

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, type TEXT, time TEXT)")
    conn.commit()
    conn.close()

def reload_face_data():
    global known_encodings, known_ids, known_names
    
    if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH)
    all_files = [f for f in os.listdir(DATASET_PATH) if f.endswith((".jpg", ".png"))]

    if not all_files:
        known_encodings, known_ids, known_names = [], [], []
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        return

    need_recompute = True
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                if len(data["ids"]) == len(all_files):
                    known_encodings = data["encodings"]
                    known_ids = data["ids"]
                    known_names = data["names"]
                    need_recompute = False
                    print(f">>> Đã nạp {len(known_ids)} ảnh từ Cache.")
        except:
            need_recompute = True

    if need_recompute:
        print(">>> Đang tính toán lại đặc trưng khuôn mặt...")
        known_encodings, known_ids, known_names = [], [], []
        for file in all_files:
            try:
                parts = os.path.splitext(file)[0].split("_")
                img = face_recognition.load_image_file(os.path.join(DATASET_PATH, file))
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_encodings.append(enc[0])
                    known_ids.append(int(parts[0]))
                    known_names.append(parts[1])
            except: continue
            
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({"encodings": known_encodings, "ids": known_ids, "names": known_names}, f)
        print(">>> Đã cập nhật xong bộ nhớ hệ thống.")

# ================= GIAO DIỆN & LOGIC =================
class UltimateAttendanceApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hệ Thống Chấm Công")
        self.window.geometry("1200x700")

        init_db()
        reload_face_data()

        #Khởi tạo MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.6)

        self.is_monitoring = True
        self.results = []
        self.lock = threading.Lock()
        self.tracking_info = {} 
        self.frame_count = 0

        self.prev_frame_time = 0
        self.ai_latency = 0
        self.current_fps = 0

        self.notebook = ttk.Notebook(self.window)
        # [MỚI] Gắn sự kiện chuyển tab
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.tab_mon = tk.Frame(self.notebook)
        self.tab_admin = tk.Frame(self.notebook)
        self.notebook.add(self.tab_mon, text=" GIÁM SÁT "); self.notebook.add(self.tab_admin, text=" QUẢN LÝ ")
        self.notebook.pack(fill="both", expand=True)

        self.setup_mon_ui()
        self.setup_admin_ui()

        self.cap = cv2.VideoCapture(0)
        self.update_main()

    # [MỚI] Xử lý tắt/bật cam khi chuyển tab
    def on_tab_change(self, event):
        selected_tab = self.notebook.index(self.notebook.select())
        if selected_tab == 0: # Tab Giám sát
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.is_monitoring = True
            print(">>> Tab Giám sát: Camera ON")
        else: # Tab Quản lý
            self.is_monitoring = False
            if self.cap:
                self.cap.release()
            print(">>> Tab Quản lý: Camera OFF")

    def setup_mon_ui(self):
        self.mon_left = tk.Frame(self.tab_mon, bg="black"); self.mon_left.pack(side="left", fill="both", expand=True)
        self.video_label = tk.Label(self.mon_left, bg="black"); self.video_label.pack(fill="both", expand=True)
        self.mon_right = tk.Frame(self.tab_mon, width=300); self.mon_right.pack(side="right", fill="both", padx=10)
        
        tk.Label(self.mon_right, text="LỊCH SỬ RA VÀO", font=("Arial", 12, "bold")).pack(pady=10)
        self.tree_log = ttk.Treeview(self.mon_right, columns=("t","n","d"), show="headings")
        self.tree_log.heading("t", text="Giờ"); self.tree_log.heading("n", text="Tên"); self.tree_log.heading("d", text="Hướng")
        self.tree_log.column("t", width=80); self.tree_log.column("d", width=60)
        self.tree_log.pack(fill="both", expand=True)

    def setup_admin_ui(self):
        ctrl = tk.Frame(self.tab_admin); ctrl.pack(side="top", fill="x", pady=10)
        tk.Button(ctrl, text="+ ĐĂNG KÝ MỚI", bg="#2ecc71", fg="white", font=("Arial", 10, "bold"), command=self.smart_register).pack(side="left", padx=20)
        tk.Button(ctrl, text="XÓA NHÂN VIÊN", bg="#e74c3c", fg="white", command=self.delete_user).pack(side="right", padx=20)
        self.tree_emp = ttk.Treeview(self.tab_admin, columns=("id","name","img"), show="headings")
        self.tree_emp.heading("id", text="ID"); self.tree_emp.heading("name", text="Tên"); self.tree_emp.heading("img", text="Dữ liệu")
        self.tree_emp.pack(fill="both", expand=True, padx=20, pady=10); self.refresh_table()

    def update_main(self):
        # Chỉ chạy nếu đang ở chế độ giám sát và camera đang mở
        if self.is_monitoring and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                
                self.new_frame_time = time.time()
                diff = self.new_frame_time - self.prev_frame_time
                if diff > 0:
                    self.current_fps = 1 / diff
                self.prev_frame_time = self.new_frame_time

                # Chạy AI mỗi 3 frame để giảm tải (trước là 4)
                if self.frame_count % 3 == 0:
                    threading.Thread(target=self.process_ai, args=(frame.copy(),), daemon=True).start()
                
                self.draw_overlay(frame)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk; self.video_label.configure(image=imgtk)
        
        self.window.after(10, self.update_main)

    def process_ai(self, frame):
        start_time = time.time()
        
        # MediaPipe nhận ảnh RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        face_locations = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                # Chuyển đổi tọa độ MediaPipe (0-1) sang tọa độ pixel
                # face_recognition cần format: (top, right, bottom, left)
                top = int(bboxC.ymin * h)
                left = int(bboxC.xmin * w)
                bottom = int((bboxC.ymin + bboxC.height) * h)
                right = int((bboxC.xmin + bboxC.width) * w)
                
                # Đảm bảo không văng ra ngoài khung hình
                top = max(0, top); left = max(0, left)
                bottom = min(h, bottom); right = min(w, right)
                
                face_locations.append((top, right, bottom, left))

        # Dùng tọa độ từ MediaPipe để encode bằng face_recognition
        encs = face_recognition.face_encodings(rgb_frame, face_locations)
        
        new_res = []
        line_x = frame.shape[1] // 2
        
        for (t,r,b,l), enc in zip(face_locations, encs):
            uname = "Unknown"
            uid = None
            
            if len(known_encodings) > 0:
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.45)
                if True in matches:
                    idx = matches.index(True)
                    uid, uname = known_ids[idx], known_names[idx]
                    
                    curr_x = l + (r-l)//2
                    if uid in self.tracking_info:
                        lx, lt = self.tracking_info[uid]
                        # [SỬA] Dùng biến COOLDOWN_SECONDS mới (3s)
                        if datetime.now() > lt + timedelta(seconds=COOLDOWN_SECONDS):
                            direction = ""
                            if lx < line_x and curr_x >= line_x: direction = "VÀO"
                            elif lx > line_x and curr_x <= line_x: direction = "RA"
                            if direction:
                                self.save_attendance(uid, uname, direction)
                                self.tracking_info[uid] = [curr_x, datetime.now()]
                    else:
                        self.tracking_info[uid] = [curr_x, datetime.now() - timedelta(seconds=COOLDOWN_SECONDS)]

            new_res.append({"box": (t, r, b, l), "name": uname})
        
        end_time = time.time()
        self.ai_latency = (end_time - start_time) * 1000 
        
        with self.lock:
            self.results = new_res

    def draw_overlay(self, frame):
        line_x = frame.shape[1] // 2
        cv2.line(frame, (line_x, 0), (line_x, 600), (0,0,255), 2)
        
        cv2.rectangle(frame, (5, 5), (250, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(self.current_fps)}", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {int(self.ai_latency)}", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        with self.lock:
            for res in self.results:
                t,r,b,l = res["box"]
                color = (0, 255, 0) if res["name"] != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, res["name"], (l, t-10), 1, 1.2, color, 2)

    def save_attendance(self, uid, name, dir):
        # 1. Lấy thời gian hiện tại
        t = datetime.now()
        
        # 2. Hiển thị lên giao diện (Treeview)
        self.tree_log.insert("", 0, values=(t.strftime("%H:%M:%S"), name, dir))
        
        # 3. Lưu vào Database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO attendance (user_id, type, time) VALUES (?,?,?)", 
                  (uid, dir, t.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        
        # HẾT HÀM. 
        # (Đoạn code đăng ký cũ bị thừa ở đây đã được xóa bỏ)

    def smart_register(self):
        # Tạm tắt giám sát
        self.is_monitoring = False
        if self.cap: self.cap.release()
        
        name = simpledialog.askstring("Đăng ký", "Họ tên nhân viên:", parent=self.window)
        if not name: 
            self.on_tab_change(None)
            return 

        # Kiểm tra tên trùng trong DB (Vẫn giữ để tránh lỗi DB)
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("SELECT id FROM employees WHERE name=?", (name,))
        if c.fetchone():
            messagebox.showerror("Lỗi", "Tên nhân viên đã tồn tại!"); 
            conn.close(); self.on_tab_change(None); return

        messagebox.showinfo("Hướng dẫn", 
                            "Hệ thống sẽ vừa chụp vừa kiểm tra an ninh.\n"
                            "- Nếu phát hiện bạn đã đăng ký: Hệ thống sẽ DỪNG NGAY.\n"
                            "- Vui lòng XOAY ĐẦU liên tục để hệ thống lấy đủ góc cạnh.")
        
        temp_cap = cv2.VideoCapture(0)
        new_id = (c.execute("SELECT MAX(id) FROM employees").fetchone()[0] or 0) + 1
        
        count = 0
        last_save = datetime.now()
        last_encoding = None # Lưu encoding của tấm ảnh vừa chụp gần nhất

        while count < 30:
            ret, f = temp_cap.read()
            if not ret: break
            
            # Xử lý mỗi 200ms để tránh lag
            if (datetime.now() - last_save).microseconds > 200000:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                
                # 1. Dùng MediaPipe để tìm mặt (Nhanh)
                results = self.face_detector.process(rgb)
                
                if results.detections:
                    detection = results.detections[0]
                    h, w, _ = f.shape
                    bboxC = detection.location_data.relative_bounding_box
                    top = int(bboxC.ymin * h); left = int(bboxC.xmin * w)
                    bottom = int((bboxC.ymin + bboxC.height) * h); right = int((bboxC.xmin + bboxC.width) * w)
                    
                    # 2. Encode khuôn mặt hiện tại (Để so sánh)
                    # Cần tính toán encoding ngay lập tức cho frame này
                    current_encs = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
                    
                    if current_encs:
                        current_enc = current_encs[0]
                        
                        # === CHECK 1: KIỂM TRA ĐỐI CHIẾU VỚI NGƯỜI KHÁC (GLOBAL) ===
                        # Chống việc che mặt lúc đầu rồi mở ra sau
                        if len(known_encodings) > 0:
                            # tolerance=0.4: Ngưỡng chặt chẽ để tránh nhận nhầm
                            matches = face_recognition.compare_faces(known_encodings, current_enc, tolerance=0.4)
                            if True in matches:
                                # TÌM THẤY TRÙNG LẶP -> DỪNG NGAY
                                first_match_index = matches.index(True)
                                existing_name = known_names[first_match_index]
                                
                                temp_cap.release(); cv2.destroyAllWindows(); conn.close()
                                messagebox.showerror("BÁO ĐỘNG", 
                                                     f"Phát hiện khuôn mặt này ĐÃ ĐĂNG KÝ!\n"
                                                     f"Hệ thống xác định đây là: {existing_name}\n"
                                                     "Quá trình đăng ký bị hủy bỏ.")
                                self.on_tab_change(None)
                                return # Thoát hàm ngay lập tức

                        # === CHECK 2: KIỂM TRA GÓC CHỤP (LOCAL) ===
                        # So với ảnh vừa chụp của chính mình trong phiên này
                        save_flag = True
                        if last_encoding is not None:
                            # Tính khoảng cách sự khác biệt giữa ảnh này và ảnh trước
                            dist = face_recognition.face_distance([last_encoding], current_enc)[0]
                            
                            # Nếu dist < 0.15 nghĩa là góc mặt gần như y hệt ảnh trước -> Bắt xoay
                            if dist < 0.15: 
                                save_flag = False
                                cv2.putText(f, "DA TRUNG GOC - XOAY MAT DI!", (50, 350), 1, 1.5, (0, 0, 255), 3)

                        if save_flag:
                            count += 1
                            # Lưu ảnh
                            cv2.imwrite(f"{DATASET_PATH}/{new_id}_{name}_{count}.jpg", f)
                            last_save = datetime.now()
                            last_encoding = current_enc # Cập nhật để so sánh cho vòng sau
                        
                else:
                    # Không thấy mặt (do che mặt hoặc quay đi chỗ khác)
                    cv2.putText(f, "KHONG THAY MAT!", (50, 350), 1, 1.5, (0, 0, 255), 3)

            # Vẽ giao diện
            cv2.rectangle(f, (50, 400), (590, 430), (50,50,50), -1)
            # Thanh tiến trình màu xanh
            cv2.rectangle(f, (50, 400), (50 + int(count * 18), 430), (0,255,0), -1)
            cv2.putText(f, f"Tien do: {count}/30", (60, 390), 1, 1.2, (255,255,255), 2)
            
            cv2.imshow("Smart Registration (Anti-Cheat)", f)
            if cv2.waitKey(1) & 0xFF == 27: break

        temp_cap.release(); cv2.destroyAllWindows()
        
        # Chỉ lưu vào DB nếu chụp đủ 30 tấm trót lọt
        if count >= 30:
            c.execute("INSERT INTO employees (id, name) VALUES (?,?)", (new_id, name))
            conn.commit(); 
            reload_face_data() # Nạp lại dữ liệu ngay để cập nhật người mới
            self.refresh_table()
            messagebox.showinfo("Thành công", f"Đã đăng ký xong nhân viên: {name}")
        
        conn.close()
        self.on_tab_change(None)
    
    def resume_mon(self): 
        # Hàm này giờ được thay thế bởi logic trong on_tab_change
        pass 

    def refresh_table(self):
        for i in self.tree_emp.get_children(): self.tree_emp.delete(i)
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        for r in c.execute("SELECT * FROM employees").fetchall():
            num = len([f for f in os.listdir(DATASET_PATH) if f.startswith(f"{r[0]}_")])
            self.tree_emp.insert("", "end", values=(r[0], r[1], f"{num} ảnh"))
        conn.close()

    def delete_user(self):
        sel = self.tree_emp.selection()
        if not sel: return
        uid = self.tree_emp.item(sel[0])['values'][0]
        if messagebox.askyesno("Xóa", "Xác nhận xóa triệt để dữ liệu nhân viên này?"):
            conn = sqlite3.connect(DB_PATH); c = conn.cursor()
            c.execute("DELETE FROM employees WHERE id=?", (uid,))
            conn.commit(); conn.close()
            for f in os.listdir(DATASET_PATH):
                if f.startswith(f"{uid}_"): os.remove(os.path.join(DATASET_PATH, f))
            if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
            reload_face_data(); self.refresh_table()

if __name__ == "__main__":
    root = tk.Tk(); app = UltimateAttendanceApp(root); root.mainloop()