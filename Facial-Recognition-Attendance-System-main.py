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

# Tắt cảnh báo lỗi thời của thư viện để console sạch sẽ
warnings.filterwarnings("ignore", category=UserWarning)

# ================= CẤU HÌNH HỆ THỐNG =================
DB_PATH = "attendance.db"
DATASET_PATH = "dataset"
CACHE_FILE = "encodings_cache.pkl"
COOLDOWN_SECONDS = 5
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
    """Đồng bộ hóa dữ liệu từ Dataset, Cache và RAM"""
    global known_encodings, known_ids, known_names
    
    if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH)
    all_files = [f for f in os.listdir(DATASET_PATH) if f.endswith((".jpg", ".png"))]

    # Nếu không có ảnh nào -> Xóa sạch dữ liệu cũ
    if not all_files:
        known_encodings, known_ids, known_names = [], [], []
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        print(">>> Hệ thống trống dữ liệu.")
        return

    # Kiểm tra Cache
    need_recompute = True
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                # Chỉ nạp nếu số lượng ảnh trong cache khớp với thực tế
                if len(data["ids"]) == len(all_files):
                    known_encodings = data["encodings"]
                    known_ids = data["ids"]
                    known_names = data["names"]
                    need_recompute = False
                    print(f">>> Đã nạp {len(known_ids)} ảnh từ Cache.")
        except:
            need_recompute = True

    if need_recompute:
        print(">>> Đang tính toán lại đặc trưng khuôn mặt (Vui lòng đợi)...")
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
        self.window.title("Hệ Thống Chấm Công Thông Minh v8.0")
        self.window.geometry("1200x700")

        init_db()
        reload_face_data()

        self.is_monitoring = True
        self.results = []
        self.lock = threading.Lock()
        self.tracking_info = {} # {id: [last_x, last_time]}
        self.frame_count = 0

        # Tabs
        self.notebook = ttk.Notebook(self.window)
        self.tab_mon = tk.Frame(self.notebook)
        self.tab_admin = tk.Frame(self.notebook)
        self.notebook.add(self.tab_mon, text=" GIÁM SÁT "); self.notebook.add(self.tab_admin, text=" QUẢN LÝ ")
        self.notebook.pack(fill="both", expand=True)

        self.setup_mon_ui()
        self.setup_admin_ui()

        self.cap = cv2.VideoCapture(0)
        self.update_main()

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

    # ---------------- GIÁM SÁT (MULTI-THREADING) ----------------
    def update_main(self):
        if self.is_monitoring:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                # Chạy nhận diện mỗi 4 khung hình để giảm tải CPU
                if self.frame_count % 4 == 0:
                    threading.Thread(target=self.process_ai, args=(frame.copy(),), daemon=True).start()
                
                self.draw_overlay(frame)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk; self.video_label.configure(image=imgtk)
        
        self.window.after(10, self.update_main)

    def process_ai(self, frame):
        # Resize ảnh nhỏ để xử lý nhanh
        small = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Tìm mặt và mã hóa
        locs = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, locs)
        
        new_res = []
        line_x = frame.shape[1] // 2
        
        for (t,r,b,l), enc in zip(locs, encs):
            uname = "Unknown"
            uid = None
            
            if len(known_encodings) > 0:
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.45)
                if True in matches:
                    idx = matches.index(True)
                    uid, uname = known_ids[idx], known_names[idx]
                    
                    # Logic xác định hướng (Tracking)
                    curr_x = l * 5
                    if uid in self.tracking_info:
                        lx, lt = self.tracking_info[uid]
                        if datetime.now() > lt + timedelta(seconds=COOLDOWN_SECONDS):
                            direction = ""
                            if lx < line_x and curr_x >= line_x: direction = "VÀO"
                            elif lx > line_x and curr_x <= line_x: direction = "RA"
                            if direction:
                                self.save_attendance(uid, uname, direction)
                                self.tracking_info[uid] = [curr_x, datetime.now()]
                    else:
                        self.tracking_info[uid] = [curr_x, datetime.now() - timedelta(seconds=COOLDOWN_SECONDS)]

            new_res.append({"box": (t*5, r*5, b*5, l*5), "name": uname})
            
        with self.lock:
            self.results = new_res

    def draw_overlay(self, frame):
        line_x = frame.shape[1] // 2
        cv2.line(frame, (line_x, 0), (line_x, 600), (0,0,255), 2)
        with self.lock:
            for res in self.results:
                t,r,b,l = res["box"]
                color = (0, 255, 0) if res["name"] != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, res["name"], (l, t-10), 1, 1.2, color, 2)

    def save_attendance(self, uid, name, dir):
        t = datetime.now()
        self.tree_log.insert("", 0, values=(t.strftime("%H:%M:%S"), name, dir))
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("INSERT INTO attendance (user_id, type, time) VALUES (?,?,?)", (uid, dir, t.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit(); conn.close()

    # ---------------- ĐĂNG KÝ THÔNG MINH (CHỐNG TRÙNG & AUTO CAPTURE) ----------------
    def smart_register(self):
        self.is_monitoring = False; self.cap.release(); cv2.destroyAllWindows()
        name = simpledialog.askstring("Đăng ký", "Họ tên nhân viên:", parent=self.window)
        if not name: self.resume_mon(); return

        # 1. Kiểm tra trùng TÊN
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("SELECT id FROM employees WHERE name=?", (name,))
        if c.fetchone():
            messagebox.showerror("Lỗi", "Tên này đã tồn tại trong DB!"); self.resume_mon(); return

        messagebox.showinfo("Đăng ký", "Máy sẽ tự chụp 30 ảnh. Hãy NGHIÊNG ĐẦU qua lại.\nHệ thống sẽ tự động kiểm tra nếu bạn đã đăng ký rồi.")
        
        temp_cap = cv2.VideoCapture(0)
        new_id = (c.execute("SELECT MAX(id) FROM employees").fetchone()[0] or 0) + 1
        count = 0; last_save = datetime.now()
        
        while count < 30:
            ret, f = temp_cap.read()
            if not ret: break
            
            # Kiểm tra trùng MẶT ở tấm ảnh đầu tiên
            if count == 0:
                rgb_test = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                enc_test = face_recognition.face_encodings(rgb_test)
                if enc_test and len(known_encodings) > 0:
                    if any(face_recognition.compare_faces(known_encodings, enc_test[0], 0.4)):
                        messagebox.showerror("Lỗi", "Khuôn mặt này đã được đăng ký trước đó!"); break

            # Tự động chụp mỗi 200ms khi thấy mặt
            if (datetime.now() - last_save).microseconds > 200000:
                locs = face_recognition.face_locations(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                if locs:
                    count += 1
                    cv2.imwrite(f"{DATASET_PATH}/{new_id}_{name}_{count}.jpg", f)
                    last_save = datetime.now()

            # Vẽ Progress Bar
            cv2.rectangle(f, (50, 400), (590, 430), (50,50,50), -1)
            cv2.rectangle(f, (50, 400), (50 + int(count * 18), 430), (0,255,0), -1)
            cv2.putText(f, f"Capturing: {count}/30", (60, 390), 1, 1.2, (255,255,255), 2)
            cv2.imshow("Registration - Tilt your head", f)
            if cv2.waitKey(1) & 0xFF == 27: break

        temp_cap.release(); cv2.destroyAllWindows()
        if count >= 30:
            c.execute("INSERT INTO employees (id, name) VALUES (?,?)", (new_id, name))
            conn.commit(); reload_face_data(); self.refresh_table()
            messagebox.showinfo("Thành công", f"Đã đăng ký: {name}")
        
        conn.close(); self.resume_mon()

    def resume_mon(self): self.cap = cv2.VideoCapture(0); self.is_monitoring = True
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
            # Xóa ảnh vật lý
            for f in os.listdir(DATASET_PATH):
                if f.startswith(f"{uid}_"): os.remove(os.path.join(DATASET_PATH, f))
            # BUỘC HỆ THỐNG XÓA CACHE VÀ NẠP LẠI
            if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
            reload_face_data(); self.refresh_table()

if __name__ == "__main__":
    root = tk.Tk(); app = UltimateAttendanceApp(root); root.mainloop()