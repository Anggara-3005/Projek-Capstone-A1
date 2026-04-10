import cv2
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import os

model = YOLO(r".\runs\detect\runs\screw_detection_v2\weights\best.pt")

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

history      = deque(maxlen=5)
total_baut   = 0       # ← total keseluruhan, tidak pernah turun
batch_terakhir = 0     # ← jumlah baut di batch sebelumnya

# Folder simpan foto
SAVE_DIR = r".\captures"
os.makedirs(SAVE_DIR, exist_ok=True)

frame_skip = 0
annotated  = None

print("Tekan R: reset total | C: capture foto | Q: keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak bisa membaca kamera!")
        break

    frame_skip += 1
    if frame_skip % 2 == 0:
        if annotated is not None:
            cv2.imshow("Screw Counter - R:Reset | C:Capture | Q:Quit", annotated)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            total_baut     = 0
            batch_terakhir = 0
            history.clear()
            print("Total direset!")
        elif key == ord('c') and annotated is not None:
            # Simpan foto dengan timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
            cv2.imwrite(filename, annotated)
            print(f"Foto disimpan: {filename}")
        continue

    results = model(frame, conf=0.69, verbose=False)
    jumlah  = len(results[0].boxes)
    history.append(jumlah)
    jumlah_stabil = max(set(history), key=history.count)

    # ── Logic counter tidak turun ──────────────────────
    # Kalau jumlah sekarang LEBIH dari batch terakhir
    # berarti ada batch baru → tambahkan ke total
    if jumlah_stabil > batch_terakhir:
        tambahan       = jumlah_stabil - batch_terakhir
        total_baut    += tambahan
        batch_terakhir = jumlah_stabil

    # Kalau baut dihilangkan semua (jumlah = 0)
    # reset batch_terakhir → siap hitung batch berikutnya
    if jumlah_stabil == 0:
        batch_terakhir = 0

    # ── Gambar UI ──────────────────────────────────────
    annotated = results[0].plot()

    # Background counter
    cv2.rectangle(annotated, (0, 0), (380, 110), (0, 0, 0), -1)

    # Jumlah baut sekarang (bisa turun, ini normal)
    cv2.putText(
        annotated,
        f"Sekarang : {jumlah_stabil}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2
    )

    # Total keseluruhan (tidak pernah turun)
    cv2.putText(
        annotated,
        f"Total    : {total_baut}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 255), 2
    )

    cv2.imshow("Screw Counter - R:Reset | C:Capture | Q:Quit", annotated)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        total_baut     = 0
        batch_terakhir = 0
        history.clear()
        print(f"Total direset!")
    elif key == ord('c'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, annotated)
        print(f"Foto disimpan: {filename}")

cap.release()
cv2.destroyAllWindows()