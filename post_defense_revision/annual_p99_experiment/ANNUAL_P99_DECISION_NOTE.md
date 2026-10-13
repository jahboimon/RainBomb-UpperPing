# Annual-P99 Decision Note

## Decision

**คง frozen Def5/P99 เป็น main target และจัด Annual-P99 เป็น sensitivity only — ไม่ promote เป็น final target ในตอนนี้**

## เหตุผล

1. Annual-P99 เปลี่ยน event membership อย่างมีนัยสำคัญ: 1,006 events, เพิ่ม 41, ตัดออก 158 และ Jaccard กับ frozen เพียง 0.829
2. station-years ที่ sample น้อยทำให้ percentile ไม่นิ่ง: 17 station-years มี wet hours ต่ำกว่า 100 และบาง threshold ต่ำผิดปกติ แม้มี floor 20 mm/h
3. ผล Evaluation 2025 เป็น trade-off ไม่ใช่ improvement ที่สอดคล้องกัน: AP ลด 0.005980 → 0.004771 และ recall ลดเล็กน้อย แม้ ROC AUC, precision, specificity และ F1 ดีขึ้น พร้อม false positives ลดลง
4. Annual-P99 ของปี y ใช้ข้อมูลครบทั้งปี y จึงเป็น retrospective label definition และไม่ใช่ threshold ที่ deploy ได้แบบ realtime โดยตรง
5. Rolling prior-years comparator มี operational logic ที่ดีกว่าและใกล้ frozen มากกว่า (Jaccard 0.884) แต่ยังมี cold-start station-years ที่ prior wet hours ไม่ถึง 100 จึงต้องกำหนด fallback policy ก่อนจึงจะทดลอง model อย่างเป็นธรรมได้

## สิ่งที่ควรรายงานในวิทยานิพนธ์

- รายงาน Annual-P99 เป็น robustness/sensitivity experiment ที่ตอบข้อเสนออาจารย์โดยตรง
- แสดง event overlap, added/removed events, sample-size caveat และ model metrics side-by-side
- ระบุชัดว่า Annual-P99 เป็น retrospective และห้ามใช้ถ้อยคำสื่อว่า threshold รู้ล่วงหน้า
- คง frozen Def5 conclusions เดิมจนกว่าจะมี pre-registered operational rolling/cold-start design และ independent validation

## หากจะทำขั้นต่อไป

กำหนด rolling policy ล่วงหน้า เช่น all-prior-years P99 + minimum 100 wet hours และ fallback เป็น frozen station P99 เมื่อ history ไม่พอ จากนั้น freeze policy ก่อน train/validation ห้ามใช้ Evaluation 2025 เลือก policy และควรประเมินกับ holdout ปีใหม่ที่ไม่เคยถูกเปิดเพื่อการตัดสินใจ
