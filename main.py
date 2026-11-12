import cv2
import mediapipe as mp
import math

# Инициализация MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Путь к файлу модели
MODEL_PATH = "face_landmarker.task"

# Настройки детектора
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

landmarker = FaceLandmarker.create_from_options(options)

# Индексы точек глаз
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Пороги
EAR_THRESHOLD = 0.22  # EAR ниже этого = глаз закрыт
CLOSED_FRAMES_LIMIT = 30  # кадров подряд с закрытыми глазами → предупреждение


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(eye_points):
    if len(eye_points) != 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_eye_points(landmarks, indices, width, height):
    """Извлечение координат точек глаза из результата FaceLandmarker."""
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append((int(lm.x * width), int(lm.y * height)))
    return points


# Запуск камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Не удалось открыть камеру!")
    exit(1)

closed_frames_counter = 0
warning_active = False
frame_timestamp = 0  # таймстамп в миллисекундах для VIDEO-режима

print("[INFO] Запуск DriverEyeGuard MVP. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Конвертация в Image для MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Детекция (VIDEO-режим требует таймстамп)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if frame_timestamp_ms == 0:
        frame_timestamp += 33  # ~30 FPS
        frame_timestamp_ms = frame_timestamp

    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    eyes_closed = False

    if result.face_landmarks and len(result.face_landmarks) > 0:
        landmarks = result.face_landmarks[0]

        # Получаем точки глаз
        left_eye_pts = get_eye_points(landmarks, LEFT_EYE, w, h)
        right_eye_pts = get_eye_points(landmarks, RIGHT_EYE, w, h)

        # Считаем EAR для каждого глаза
        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)

        # Рисуем точки глаз на кадре
        for pt in left_eye_pts + right_eye_pts:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        # Оба глаза закрыты?
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            eyes_closed = True
            closed_frames_counter += 1
        else:
            closed_frames_counter = 0
            warning_active = False

        # Выводим EAR на кадр
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Лицо не найдено — сбрасываем счётчик
        closed_frames_counter = 0
        warning_active = False

    # Проверка: предупреждение
    if closed_frames_counter >= CLOSED_FRAMES_LIMIT:
        warning_active = True

    # Отрисовка предупреждения
    if warning_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(frame, "PROSNIS!", (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
        cv2.putText(frame, f"Closed frames: {closed_frames_counter}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('DriverEyeGuard MVP', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("[INFO] Завершено.")
