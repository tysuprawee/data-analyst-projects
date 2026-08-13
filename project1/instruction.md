# Project 1: Data Cleaning (Jupyter Notebook + GitHub)

ทำใน Jupyter/Colab Notebook

**เป้าหมาย:** มี repo ที่เปิดดู notebook ได้ + ข้อมูลสะอาด (`diabetes_clean.csv`) + README

หลักการเดิม: `คิด · สั่ง · ตรวจ`, ให้ AI เขียนโค้ดแต่ละ cell ให้ได้ หน้าที่เราคือ "อ่านให้เข้าใจ + ตรวจผล" ทุกครั้งก่อนไปขั้นถัดไป

---

## 1. เครื่องมือ

| เครื่องมือ | ใช้ทำอะไร |
|---|---|
| Jupyter / Colab Notebook | เขียนโค้ด + เห็นผลทีละ cell, สลับ markdown (อธิบาย)|
| GitHub | เก็บงาน + version control, commit เป็นระยะ = ประวัติเล่าเรื่องได้|

**Notebook ที่ดี = เล่าเรื่องได้**, สลับ markdown cell (อธิบายว่ากำลังทำอะไร/ทำไม) → code cell (โค้ด pandas ลงมือทำจริง) → output (ผลลัพธ์ที่ต้องตรวจทันที) วนไปเรื่อย ๆ ทุกขั้นตอน

---

## 2. โครงสร้างโปรเจกต์

```
diabetes-cleaning/
├── data/
│   ├── raw/diabetes.csv
│   └── clean/diabetes_clean.csv
├── notebooks/
│   └── cleaning.ipynb
├── README.md
└── .gitignore
```

ตั้ง repo แล้ว commit ครั้งแรกไว้เป็นจุดเริ่ม

เคล็ดลับ: commit เป็นระยะหลังแต่ละขั้นตอน

---

## 3. ชุดข้อมูล

**Pima Indians Diabetes** (`diabetes.csv`, 768 แถว, 9 คอลัมน์), ข้อมูลจริงจาก Kaggle เก็บจากผู้ป่วยหญิงเชื้อสาย Pima Indian อายุ 21 ปีขึ้นไป ใช้ทำนายความเสี่ยงเบาหวาน

คอลัมน์:

| คอลัมน์ | ความหมาย |
|---|---|
| Pregnancies | จำนวนครั้งที่ตั้งครรภ์ |
| Glucose | ระดับน้ำตาลในเลือดจากการทดสอบ Oral Glucose Tolerance Test (mg/dL) |
| BloodPressure | ความดันโลหิตช่วง diastolic (mm Hg) |
| SkinThickness | ความหนาผิวหนังบริเวณ triceps skinfold (mm) |
| Insulin | ระดับอินซูลินในเลือดหลังทดสอบ 2 ชั่วโมง (mu U/ml) |
| BMI | ดัชนีมวลกาย (kg/m²) |
| DiabetesPedigreeFunction | คะแนนความเสี่ยงทางพันธุกรรม คำนวณจากประวัติเบาหวานในครอบครัว |
| Age | อายุ (ปี) |
| Outcome | ผลตรวจเบาหวาน (0 = ไม่เป็น, 1 = เป็น) |

ปัญหาจริงในข้อมูล:

- **ค่าหายแฝงเป็น 0:** `df.isna().sum()` จะรายงานว่าทุกคอลัมน์ไม่มีค่าหายเลย, แต่ Glucose, BloodPressure, SkinThickness, Insulin, BMI เป็นค่าที่ทางสรีรวิทยาเป็น 0 จริงไม่ได้, ค่า 0 ในคอลัมน์เหล่านี้คือ "ค่าหายที่แฝงตัวมา" ไม่ใช่ค่าจริง, Glucose 0 อยู่ 5 คน · BloodPressure 0 อยู่ 35 คน (around 4.6%) · SkinThickness 0 อยู่ 227 คน (around 29.6%) · Insulin 0 อยู่ 374 คน (around 48.7%) · BMI 0 อยู่ 11 คน
- **ค่าโดด:** Insulin สูงสุด 846 (มี 3 คนที่ Insulin เกิน 600: 680, 744, 846) · SkinThickness สูงสุด 99 มม. (เกินช่วงปกติทางกายภาพ)
- **ซ้ำ:** 0 แถว, เป็นผลลัพธ์ที่ถูกต้องแต่ **ต้องตรวจ**

---

## 4. แผนทำความสะอาดข้อมูล 6 ขั้น

ทุกขั้นทำแบบเดิม: สั่ง AI เขียน cell ให้ → อ่านเข้าใจ + ตรวจ output → commit → ไปขั้นต่อไป

### ขั้น 0: เริ่ม notebook: import + โหลดข้อมูล
```python
import pandas as pd
import numpy as np
df = pd.read_csv('data/raw/diabetes.csv')
df.shape   # → (768, 9)
```
ทีละบรรทัด:
- `import pandas as pd` โหลดไลบรารี pandas ใช้ชื่อย่อ `pd`
- `import numpy as np` โหลดไลบรารี numpy ใช้ชื่อย่อ `np` (ใช้ทำ `np.nan` ในขั้นเติมค่าหายทีหลัง)
- `df = pd.read_csv(...)` อ่านไฟล์ csv เข้ามาเป็นตาราง เก็บไว้ในตัวแปร `df`
- `df.shape` เช็คขนาดตาราง (แถว, คอลัมน์) ตรวจว่าตรง 768 แถว 9 คอลัมน์

### ขั้น 1: สำรวจก่อน (เข้าใจความรกก่อนแตะ)
```python
df.info()
df.isna().sum()
df[['Glucose','BMI']].describe()
(df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum()
```
ทีละบรรทัด:
- `df.info()` ดูภาพรวมทุกคอลัมน์: ชื่อคอลัมน์, จำนวนค่าที่ไม่ว่าง, ชนิดข้อมูล
- `df.isna().sum()` นับค่าหาย (NaN) แต่ละคอลัมน์, ผลลัพธ์คือ 0 ทุกคอลัมน์, ยังไม่เห็นปัญหาจริง
- `df[['Glucose','BMI']].describe()` สถิติพื้นฐานเฉพาะ Glucose กับ BMI (count, mean, min, max, quartile)
- `(df[[...]] == 0).sum()` นับจำนวนแถวที่ค่าเป็น 0 เจาะจง 5 คอลัมน์ที่ 0 เป็นไปไม่ได้ทางการแพทย์, บรรทัดนี้เผยค่าหายแฝงที่ `isna()` มองไม่เห็น

จดจำนวนแถวเริ่มต้น 768 ไว้เทียบทีหลัง

### ขั้น 2: ลบแถวซ้ำ
```python
df.duplicated().sum()   # → 0
df = df.drop_duplicates()
```
ทีละบรรทัด:
- `df.duplicated().sum()` นับจำนวนแถวที่ซ้ำกันทั้งแถว, ผลลัพธ์ 0 คือไม่มีซ้ำ
- `df = df.drop_duplicates()` ลบแถวซ้ำออก แล้วเขียนทับ `df` เดิม (แม้ตอนนี้ 0 แถวก็ยังใส่ไว้เป็นขั้นตอนมาตรฐาน)

0 คือผลลัพธ์ที่ถูกต้อง ต้อง "ตรวจ" เอง ไม่ใช่ "เดา" ว่ามีหรือไม่มี

### ขั้น 3: แก้ให้สอดคล้อง (Structural)
```python
df['Outcome_Label'] = df['Outcome'].map({0: 'Negative', 1: 'Positive'})
df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100],
    labels=['20s', '30s', '40s', '50s', '60+'], right=False)
df['Outcome'] = df['Outcome'].astype('category')
```
ทีละบรรทัด:
- `df['Outcome_Label'] = df['Outcome'].map({...})` แปลง Outcome ตัวเลข 0/1 เป็นข้อความอ่านง่าย Negative/Positive เก็บไว้เป็นคอลัมน์ใหม่ (คอลัมน์ Outcome ตัวเลขเดิมยังอยู่ ใช้คำนวณต่อได้)
- `pd.cut(df['Age'], bins=[...], labels=[...])` แบ่งอายุเป็นช่วง (bucket) สร้างคอลัมน์ `AgeGroup` ใหม่ ช่วยดูแนวโน้มตามกลุ่มวัยได้ง่ายขึ้น
- `df['Outcome'] = df['Outcome'].astype('category')` เปลี่ยนชนิดข้อมูลคอลัมน์ Outcome ให้เป็น category

ตรวจว่ามีคอลัมน์ Outcome_Label และ AgeGroup ใหม่ อ่านง่ายขึ้นกว่ารหัสตัวเลข

### ขั้น 4: จัดการค่าโดด (Outliers)
```python
df[df['Insulin'] > 600]          # 3 แถว: 680, 744, 846
df['SkinThickness'].describe()   # → max 99
# ตัดสินใจ: สืบ Insulin>600 ก่อนตัดสิน (อาจจริงในคนไข้ดื้ออินซูลิน)
# SkinThickness=99 มม. น่าสงสัย (เกินช่วงปกติทางกายภาพ)
```
ทีละบรรทัด:
- `df[df['Insulin'] > 600]` กรองดูเฉพาะแถวที่ Insulin เกิน 600 (เจอ 3 แถว: 680, 744, 846)
- `df['SkinThickness'].describe()` ดูสถิติของ SkinThickness ทั้งคอลัมน์ (min, max, ค่าเฉลี่ย ฯลฯ)
- บรรทัด `#` ไม่ใช่โค้ดที่รัน เป็นคอมเมนต์บันทึกเหตุผลการตัดสินใจไว้ในเซลล์

อ่าน+ตัดสินใจ: Insulin สูงมากอาจเป็นค่าจริงในคนไข้ที่ดื้ออินซูลิน ต้องสืบก่อนตัด, SkinThickness 99 มม. เกินช่วงปกติทางกายภาพมาก น่าสงสัยกว่า, ลบเพราะ "ผิด" ไม่ใช่เพราะ "โดด" และต้องจดเหตุผลที่ตัด/เก็บลงใน markdown

### ขั้น 5: จัดการค่าหาย (Missing)
```python
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df.groupby('Outcome')[cols]\
    .transform(lambda x: x.fillna(x.median()))

df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['HasInsulinReading'] = df['Insulin'].notna().astype(int)
```
ทีละบรรทัด:
- `df[cols] = df[cols].replace(0, np.nan)` แปลงค่า 0 ที่เป็นไปไม่ได้ทางสรีรวิทยาใน 4 คอลัมน์ให้เป็น `NaN` (ค่าหายจริง) ก่อน, ถ้าไม่แปลงก่อน pandas จะไม่รู้ว่าค่านี้ "หาย"
- `df.groupby('Outcome')[cols].transform(lambda x: x.fillna(x.median()))` เติมค่าหายด้วย median แยกตามกลุ่ม Outcome (เป็นเบาหวาน/ไม่เป็น) เพราะค่าปกติของแต่ละกลุ่มต่างกัน
- `df['Insulin'] = df['Insulin'].replace(0, np.nan)` แปลง Insulin (0 = หายแฝง) เป็น NaN เช่นกัน
- `df['HasInsulinReading'] = df['Insulin'].notna().astype(int)` แต่ "ไม่เติม" ค่า Insulin เพราะหายถึง around 48.7% เชื่อถือไม่พอจะเดา จึงสร้าง flag บอกว่ามีค่า Insulin จริงให้ใช้หรือไม่ แทนการทิ้งคอลัมน์หรือเดามั่ว

### ขั้น 6: ตรวจผลรวม (Validate)
```python
df.isna().sum()    # → Glucose 0, BloodPressure 0, SkinThickness 0, BMI 0
                    #   Insulin ยังมี NaN อยู่ (ตามตั้งใจ)
df.shape            # → (768, 12)
df[['Glucose','BMI']].describe()
```
ทีละบรรทัด:
- `df.isna().sum()` ตรวจซ้ำว่าค่าหายเหลือเท่าไหร่ในแต่ละคอลัมน์ (คอลัมน์ที่เติมแล้วต้องเป็น 0, ยกเว้น Insulin ที่ตั้งใจไม่เติม)
- `df.shape` ตรวจขนาดตารางหลังทำความสะอาด (แถวไม่ควรหายไปเยอะ, คอลัมน์เพิ่มขึ้นจาก Outcome_Label, AgeGroup, HasInsulinReading ที่สร้างใหม่)
- `df[['Glucose','BMI']].describe()` ตรวจสถิติ Glucose, BMI อีกครั้งว่าช่วงค่ายังสมเหตุผล (ไม่มี 0 หลงเหลือ)

เช็คลิสต์สุดท้าย:
- ค่าหายในคอลัมน์ที่เติมแล้ว = 0 (ยกเว้น Insulin ที่ตั้งใจปล่อย NaN ไว้)
- จำนวนแถวไม่หายเยอะผิดปกติ (ยังคง 768)
- ช่วงค่าสมเหตุผล ไม่มีค่า 0 แฝงหลงเหลือในคอลัมน์ที่ควรเติมแล้ว

---

## 5. จด markdown + บันทึกไฟล์

Markdown cell สรุปการล้างข้อมูล (ตัวอย่าง):
```markdown
## สรุปการทำความสะอาดข้อมูล
- Duplicates: 0 (ตรวจแล้วไม่มี)
- Glucose / BloodPressure / SkinThickness / BMI: ค่า 0 (หายแฝง) แปลงเป็น NaN แล้วเติม median แยกตาม Outcome
- Insulin: หายแฝง around 48.7% เก็บเป็น NaN แล้วทำ flag HasInsulinReading แทนการเดา
```

```python
df.to_csv('data/clean/diabetes_clean.csv', index=False)
```

ทำไมสำคัญ (portfolio): markdown เล่า "การตัดสินใจ" ไม่ใช่แค่โค้ด, แยก raw/clean ไม่ทับข้อมูลดิบ, `index=False` กัน column เกินตอนเซฟ, ใครเปิดอ่านก็เข้าใจกระบวนการทั้งหมด นี่คือสิ่งที่แยกงานมือใหม่กับมืออาชีพ

---

## 6. Commit & Push ขึ้น GitHub

ทำ `.gitignore` กัน `.ipynb_checkpoints/` ไม่ให้ขึ้น repo แล้ว commit ("clean: dedup + structural + impute hidden-zero missing") จากนั้น push

ควรมีใน repo: `notebooks/cleaning.ipynb`, `data/raw/diabetes.csv`, `data/clean/diabetes_clean.csv`, `README.md`, `.gitignore`

ไม่แน่ใจ? ให้ Claude Code ช่วย push ได้

---

## 7. ทำ repo ให้ "ปัง" ใน Portfolio

- **README ที่เล่าเรื่อง**, ปัญหา / ข้อมูลรกยังไง / แก้ยังไง / ผลลัพธ์
- **notebook เปิดดูบนเว็บได้**, GitHub เรนเดอร์ `.ipynb` ให้เลย (หรือใช้ nbviewer)
- **โชว์ before/after**, ตาราง/ตัวเลขเทียบ ให้เห็นว่าสะอาดขึ้นจริง
- **อธิบายการตัดสินใจ**, ทำไมถือว่า 0 คือค่าหาย ทำไม flag Insulin แทนการเดา

---

## 8. ระวัง! กับดักตอนล้างข้อมูล

- **ล้างมากไป (over-clean)**, ลบ outlier ที่จริง ๆ เป็นของจริง = ทำข้อมูลบิด
- **เชื่อ AI ทันที**, โค้ด/ผลอาจผิด ต้องรัน cell แล้ว "ตรวจ output" เสมอ
- **ทับไฟล์ดิบ**, เก็บ raw แยกไว้เสมอ พังแล้วย้อนได้
- **ลบสิ่งที่ไม่เข้าใจ**, คอลัมน์งง ๆ ให้สืบก่อน อย่าลบทิ้งมั่ว
- **เชื่อ `isna()` เพียงอย่างเดียว**, ในข้อมูลสุขภาพค่าหายมักแฝงเป็น 0, ต้องรู้ context ว่าคอลัมน์ไหน 0 จริงไม่ได้

---

## 9. ลงมือทำ (เป้าหมายในคาบ)

1. สร้าง `cleaning.ipynb` ล้าง `diabetes.csv` ให้ครบ 6 ขั้น (cell + ตรวจทุกขั้น)
2. ใส่ markdown อธิบายการตัดสินใจแต่ละขั้น
3. เซฟ `diabetes_clean.csv` แยกไว้ใน `data/clean/`
4. commit เป็นระยะ แล้ว push ขึ้น GitHub พร้อม README

ติดตรงไหน: ให้ AI เขียน/อธิบาย cell ได้เลย, แต่ต้อง "รันแล้วตรวจ output" ทุกครั้งก่อนไปต่อ

---

## 10. สรุป

- ล้างข้อมูลจริงครบ 6 ขั้นในโน้ตบุ๊ก: สำรวจ → ลบซ้ำ → สอดคล้อง → outlier → ค่าหาย → validate
- ได้ผลงานชิ้นแรก: notebook + clean.csv + README + GitHub repo
- ใช้ mindset เดิม: AI เขียน cell, เรา "อ่านเข้าใจ + ตรวจ output"
- เข้า Portfolio ได้ทันที: ลิงก์ GitHub เดียว เปิดดู notebook ได้เลย

**การบ้าน:** ทำ README ให้สวย + commit/push ให้ครบ · เตรียม dataset ที่อยากใช้ใน Project ถัดไป

**ต่อไป:** Project 2: SQL Analysis (ดึง + วิเคราะห์จากฐานข้อมูล)
