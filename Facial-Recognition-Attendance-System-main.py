import cv2
import numpy as np
import os
from PIL import Image
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import font as tkfont
from datetime import datetime
import sqlite3
import face_recognition # Thư viện CNN/Dlib

# ================= DATABASE SETUP =================
DB_PATH = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT,
            time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

if not os.path.exists("dataset"):
    os.makedirs("dataset")

# ================= HELPER FUNCTIONS =================

def get_next_user_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM employees")
    r = cursor.fetchone()
    conn.close()
    return 1 if r[0] is None else r[0] + 1

def load_known_encodings():
    """Tải và mã hóa tất cả khuôn mặt trong dataset (Logic CNN)"""
    known_encodings = []
    known_ids = []
    known_names = []
    
    for file in os.listdir("dataset"):
        if file.endswith((".jpg", ".png")):
            path = os.path.join("dataset", file)
            # Tên file định dạng: id_name.jpg
            parts = os.path.splitext(file)[0].split("_")
            if len(parts) < 2: continue
            
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            
            if len(encoding) > 0:
                known_encodings.append(encoding[0])
                known_ids.append(int(parts[0]))
                known_names.append(parts[1])
    return known_encodings, known_ids, known_names

# ================= MAIN LOGIC =================

def register_user():
    # SỬA LỖI HIỆN 2 LẦN: Tạm ẩn root
    root.withdraw() 
    
    name = simpledialog.askstring("Register", "Enter your name:", parent=root)
    if not name:
        root.deiconify()
        return

    face_id = get_next_user_id()
    
    # Mở camera để chụp 1 ảnh duy nhất (CNN không cần 100 ảnh)
    cam = cv2.VideoCapture(0)
    messagebox.showinfo("Register", "Nhìn vào camera và nhấn 'S' để chụp ảnh")
    
    captured = False
    while True:
        ret, img = cam.read()
        if not ret: break
        cv2.imshow("Registering - Press 'S' to Capture", img)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Lưu ảnh theo định dạng id_name.jpg
            img_path = f"dataset/{face_id}_{name}.jpg"
            cv2.imwrite(img_path, img)
            captured = True
            break
        if cv2.waitKey(1) & 0xFF == 27: # ESC để hủy
            break

    cam.release()
    cv2.destroyAllWindows()

    if captured:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO employees (id, name) VALUES (?, ?)", (face_id, name))
        conn.commit()
        conn.close()
        messagebox.showinfo("Success", f"Registered {name} (ID: {face_id})")
    
    root.deiconify()

def recognize_attendance(mode):
    root.withdraw()
    known_encodings, known_ids, known_names = load_known_encodings()
    
    if not known_encodings:
        messagebox.showwarning("Error", "Chưa có dữ liệu khuôn mặt!")
        root.deiconify()
        return

    cam = cv2.VideoCapture(0)
    start_time = datetime.now()
    found = False

    while (datetime.now() - start_time).seconds < 15: # Chờ nhận diện trong 15s
        ret, img = cam.read()
        if not ret: break

        # Tối ưu: Resize ảnh nhỏ lại để CNN chạy nhanh hơn
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Tìm và mã hóa khuôn mặt hiện tại
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            # So sánh với dữ liệu đã biết
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    user_id = known_ids[best_match_index]
                    emp_name = known_names[best_match_index]
                    
                    # Ghi nhận vào DB
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute("INSERT INTO attendance (user_id, type, time) VALUES (?, ?, ?)", (user_id, mode, now))
                    conn.commit()
                    conn.close()
                    
                    messagebox.showinfo(mode, f"Xin chào {emp_name}!")
                    found = True
                    break
        
        if found: break
        cv2.imshow(f"System - {mode} (Press ESC to cancel)", img)
        if cv2.waitKey(1) & 0xFF == 27: break

    cam.release()
    cv2.destroyAllWindows()
    if not found:
        messagebox.showwarning("Fail", "Không nhận diện được khuôn mặt")
    root.deiconify()

# ================= GUI UPDATE =================
root = tk.Tk()
root.geometry("400x350")
root.title("Facial Recognition CNN System")

font_btn = tkfont.Font(size=12, weight="bold")

tk.Label(root, text="Face Recognition System (CNN)", font=("Helvetica", 16, "bold"), fg="darkblue").pack(pady=20)

# Sửa lệnh command: KHÔNG có dấu ngoặc ()
tk.Button(root, text="Register User", font=font_btn, bg="#f0f0f0",
          command=register_user).pack(fill="x", padx=40, pady=10)

tk.Button(root, text="Check In", font=font_btn, bg="#d4edda",
          command=lambda: recognize_attendance("IN")).pack(fill="x", padx=40, pady=10)

tk.Button(root, text="Check Out", font=font_btn, bg="#f8d7da",
          command=lambda: recognize_attendance("OUT")).pack(fill="x", padx=40, pady=10)

root.mainloop()