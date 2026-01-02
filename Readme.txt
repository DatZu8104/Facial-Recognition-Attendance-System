Xóa dữ lieu trên MySQL và dataset trước khi thực hiện
Mở MySQL:

CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY, -- ID nhân viên, tự tăng
    name VARCHAR(255) NOT NULL         -- Tên nhân viên, không được để trống
);
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,      -- ID chấm công, tự tăng
    user_id INT NOT NULL,                   -- ID nhân viên (liên kết tới employees)
    check_in_time DATETIME NOT NULL,        -- Thời gian check-in
    check_out_time DATETIME,                -- Thời gian check-out
    FOREIGN KEY (user_id) REFERENCES employees(id) ON DELETE CASCADE -- Khóa ngoại
);



SELECT * FROM employees;
SELECT * FROM attendance;

DELETE FROM employees;



mở cloud shell đúng đường dẫn: D:\workspace\Facial-Recognition-Attendance-System> cloud-sql-proxy data-aileron-445503-m2:asia-east2:group01 --address 127.0.0.1 --port 3306 --credentials-file="D:\workspace\Facial-Recognition-Attendance-System\data-aileron-445503-m2-104fe217a299.json"

mở visual, terminal đúng đường dẫn: D:\workspace\Facial-Recognition-Attendance-System> python Facial-Recognition-Attendance-System-main.py

host: 127.0.0.1
datvu8104
datvu0961107029
datvu8104

