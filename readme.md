# 🚀 คู่มือการนำ FTMO Trading Bot ไปรันบน VPS (สำหรับรันตลอด 24/5)

เนื่องจากบอทตัวนี้ต้องเชื่อมต่อกับ **MetaTrader 5 (MT5)** ซึ่งรันได้เสถียรสุดบน Windows และไลบรารี `MetaTrader5` ของ Python นั้นบังคับว่า **ต้องรันบนระบบปฏิบัติการ Windows เท่านั้น** การเช่า VPS จึงต้องเลือกเป็นแบบ **Windows Server** ครับ

---

## 1. การเลือกเช่า VPS

เลือกผู้ให้บริการที่เสถียรและควรมี Server Location อยู่ใกล้เคียงกับเซิร์ฟเวอร์ของโบรกเกอร์ (ปกติ FTMO จะอยู่แถวๆ ลอนดอน หรือ แฟรงก์เฟิร์ต หรือนิวยอร์ก)

- **สเปคขั้นต่ำที่แนะนำ**: 2 vCPU, RAM 4GB, SSD 40GB+ (เนื่องจากเรามีรันโมเดล PPO AI ด้วย 2GB อาจจะอืดไปนิดนึง)
- **ระบบปฏิบัติการ**: Windows Server 2019 หรือ 2022
- **ผู้ให้บริการที่นิยมสำหรับ Forex**: Vultr, Contabo (ราคาถูกสเปคคุ้ม), AWS EC2 (Windows), Kamatera

## 2. การเตรียมความพร้อมบน VPS (เมื่อ Remote Desktop เข้าไปแล้ว)

### A. ปิดการ Sleep และ Windows Update อัตโนมัติ (สำคัญมาก)

- ไปที่ **Power & Sleep settings** -> เลือก "Never" ให้หมด เพื่อไม่ให้เครื่องดับ
- ค้นหา **Services** -> หา **Windows Update** -> คลิกขวา Properties -> ปรับ Startup type เป็น "Disabled" (ป้องกัน VPS รีสตาร์ทตัวเองตอนมีอัพเดท)

### B. ติดตั้ง MetaTrader 5

- ดาวน์โหลด MT5 จากเว็บไซต์ [FTMO](https://ftmo.com/en/platforms/)
- ติดตั้งและล็อกอินด้วยหมายเลขบัญชี FTMO ของคุณ
- ไปที่ `Tools -> Options -> Expert Advisors` -> **ติ๊กถูกที่ปุ่ม "Allow Algorithmic Trading"**

### C. ติดตั้ง Python

- ดาวน์โหลด Python (แนะนำเวอร์ชัน 3.9 ถึง 3.11) จาก python.org
- ⚠️ **สำคัญ:** ตอนกดติดตั้ง (หน้าแรกสุด) ให้ติ๊กถูกที่ช่อง **"Add Python to PATH"** ด้วย
- ตรวจสอบโดยเปิด Command Prompt (`cmd`) แล้วพิมพ์ `python --version`

## 3. การนำ Source Code ขึ้น VPS

คุณสามารถนำโปรเจคนี้ (โฟลเดอร์ `ftmo_trading_bot`) ขึ้น VPS ได้ด้วยวิธีดังนี้:

1. การเชื่อมต่อ Github (ถ้าเก็บโค้ดไว้ใน Github) ให้ติดตั้ง Git แล้วรัน `git clone ...`
2. การ Copy/Paste ย้ายผ่านหน้าจอ Remote Desktop โดยตรง (คัดลอกโฟลเดอร์จากเครื่องหลัก ไป Paste วางใน VPS)
3. อัพโหลดขึ้น Google Drive ของตนเอง แล้วไปดาวน์โหลดลงในเครื่อง VPS

## 4. ตั้งค่าระบบเพื่อเทรดจริง

1. **ติดตั้งไลบรารีที่จำเป็น**  
   เปิด Command Prompt ไปยังโฟลเดอร์ของบอท แล้วรัน:

   ```cmd
   pip install -r requirements.txt
   ```

2. **แก้ไขการตั้งค่าใน `config/settings.py`**
   เปิดไฟล์ `config/settings.py` และเปลี่ยนค่าดังนี้:
   ```python
   # ในส่วนของ MT5Config
   terminal_path: str = r"C:\Program Files\FTMO MetaTrader 5\terminal64.exe" # เช็คให้ตรงกับที่ติดตั้งจริง
   login: int = 12345678              # ใส่หมายเลขบัญชีของคุณ
   password: str = "รหัสผ่านFTMO"        # ใส่รหัสผ่านของคุณ
   server: str = "FTMO-Server"        # ใส่เซิร์ฟเวอร์ของคุณ
   live_trading: bool = True          # ⚠️ เปลี่ยนจาก False เป็น True
   ```

## 5. วิธีการตื่นตัวและรัน (Execution)

### ขั้นตอนที่ 1: Train AI จากข้อมูล Mock (ทางเลือกเสริม)

ก่อนที่จะเริ่ม เพื่อให้ AI มีจิตสำนึกเรื่อง FTMO ให้กดรันเทรนโมเดลสัก 1 รอบ (ใช้เวลา 1-2 นาที):

```cmd
python main.py --train-rl
```

### ขั้นตอนที่ 2: สั่งรัน Bot

เปิด Command Prompt ในโฟลเดอร์ของ Bot แล้วรัน:

```cmd
python main.py
```

> [!TIP]
> พอรันคำสั่งปุ๊บ ระบบจะเด้งตรวจสอบ MT5, ดึงค่าน้ำหนักจาก RL Agent และเข้าสู่โหมด Standby ตามเวลาตลาดทันที คุณสามารถย่อหน้าจอ (Minimize) Command Prompt ทิ้งไว้ได้เลย ห้ามกดกากบาท (X) ปิดหน้าต่างเป็นอันขาด

## 6. เครื่องมือที่จะช่วยให้บอทรันได้มั่นคงขึ้น (แนะนำเพิ่มเติม)

ถ้ากลัวว่ารันผ่าน Command Prompt แล้วเผลอกดปิด หรือ Windows เกิดข้อผิดพลาด ขอแนะนำ 2 วิธีนี้:

1. **ใช้ NSSM (Non-Sucking Service Manager)**:
   เป็นโปรแกรมแปลง Script Python ของเราให้กลายเป็นเบื้องหลัง (Windows Service) ซึ่งมันจะสตาร์ทบอทให้เราทำงานตลอดเวลา (รันอัตโนมัติตอนเผลอรีสตาร์ทเครื่อง)
2. **ใช้ `.bat` ไฟล์ควบคู่กับการทำ Auto-Restart**:
   สร้างไฟล์ชื่อ `run_bot.bat` พิมพ์โค้ดดังนี้:
   ```bat
   :START
   python main.py
   echo Bot crashed! Restarting in 10 seconds...
   timeout /t 10
   goto START
   ```
   แล้วกดรันไฟล์ `.bat` ตัวนี้แทน ถ้าบอทมีปัญหาล่ม (Crash) ระบบจะเปิดบอทใหม่ให้อัตโนมัติใน 10 วินาที

> [!WARNING]
> **การปิด VPS**
> เมื่อคุณปิดหน้าต่าง Remote Desktop กลับออกมา **ห้ามกด Shut down หรือ Sign out ใน VPS เด็ดขาด** ให้กดที่ปุ่ม X ที่แถบสีเหลืองด้านบน (Disconnect) ของโปรแกรม Remote Desktop เท่านั้น เพื่อให้ VPS ยังคงทำงานรันบอทอยู่ 24 ชม.
