# 📥 News Inbox

Folder สำหรับ drop CSV export จาก **ForexFactory** — bot จะ auto-import ทุกวันอาทิตย์ **23:30 EET**

## วิธีใช้

1. Export CSV จาก <https://www.forexfactory.com/calendar>
   - ตั้ง Time Zone = **GMT (UTC)**
   - Filter: **Impact = High only**
   - เลือก **Next Week** (สัปดาห์ที่จะมาถึง)
   - กด Export / ⬇ → ได้ไฟล์ `ff_calendar_thisweek.csv`

2. วางไฟล์ CSV ใน folder นี้ (`config/news_inbox/`)
   - ชื่อไฟล์อะไรก็ได้ ลงท้ายด้วย `.csv`
   - ถ้ามีหลายไฟล์ bot จะใช้ไฟล์ที่ชื่อ**มาทีหลัง** (sort alphabetically)

3. วันอาทิตย์ 23:30 EET → bot auto-import:
   - Parse CSV → เขียนทับ `config/news_calendar.json`
   - Move CSV ไป `processed/YYYY-MM-DD_filename.csv` (เก็บประวัติ)
   - Log: `✅ [NewsScheduler] Import เสร็จ: N events`

## ถ้าลืมวาง CSV

- Bot log: `📅 ถึงเวลา import แต่ไม่พบ CSV — ข้าม (ใช้ไฟล์เดิมหรือ fallback hardcoded)`
- `news_calendar.json` เดิมยังอยู่ — ถ้ายังไม่หมดอายุ bot ใช้ต่อได้
- ถ้าหมดอายุ (`valid_until` ผ่านไปแล้ว) → fallback ไปใช้ hardcoded events (ความแม่น ~40-50%)

## รัน manual (ไม่รอ Sunday)

```bash
cd ftmo_trading_bot
python scripts/import_forexfactory_csv.py config/news_inbox/ff_calendar.csv
```

## Folder Structure

```
config/news_inbox/
├── README.md              ← ไฟล์นี้
├── ff_calendar.csv        ← user drop ตรงนี้
└── processed/             ← bot move มาเก็บหลัง import
    ├── 2026-04-19_ff_calendar.csv
    └── 2026-04-26_ff_calendar.csv
```
