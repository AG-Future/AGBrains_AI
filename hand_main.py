from typing import Tuple
import cv2
import mediapipe as mp
from send_request import send_request, set_port

# 사용자 입력 초기 설정
send_request_true = 0
hands_num = 1
camera_set = 0
temp_default = True

try:
    print("Input stayLength(float number): ", end='')
    stayLength = input()
    if stayLength == 'default' or stayLength == 'Default' or stayLength == 'def' or stayLength == 'DEF':
        stayLength = 5000
        temp_default = False
    else:
        stayLength = float(stayLength)
except ValueError:
    stayLength = 5000
    print(f"Only numbers are allowed, now stayLength is default({stayLength})")

if temp_default:
    # send_request 여부 설정
    print("send_request(y/n): ", end='')
    temp = input()
    if temp in ['y', 'Y']:
        send_request_true = 1
        # 포트 번호 설정
        try:
            print("port set: ", end='')
            port = int(input())
            set_port(port)
        except ValueError:
            print("Please enter a number, now port is default(80)")

    if temp not in ['y', 'Y', 'n', 'N']:
        print("Please enter y or n, now send_request is default(n)")


    # 손 개수 설정
    try:
        print("max_num_hands(1 or 2): ", end='')
        hands_num = int(input())
        if hands_num not in [1, 2]:
            print("Please enter 1 or 2, now hands is default(1)")
    except ValueError:
        print("Please enter a number, now hands is default(1)")

    # 카메라 설정
    try:
        print("camera set(0 - a): ", end='')
        camera_set = int(input())
    except ValueError:
        print("Please enter a number, now camera is default(0)")



# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

HAND_SIZE_THRESHOLD = 3200
HAND_SIZE_MINI = 2700

# 비디오 설정
cap = cv2.VideoCapture(camera_set)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1250)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 전체화면 여부 변수
is_fullscreen = False

def calculate_hand_size():
    wrist = hand_landmarks.landmark[0]
    middle = hand_landmarks.landmark[9]
    height, width = frame.shape[:2]

    hand_size = length_of((middle.x, middle.y), (wrist.x, wrist.y))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    warning_text = "****WARNING: The distance is too far to be recognized.***"
    caution_text = "Caution: Too far from camera"

    def draw_centered_text(text, y_offset):
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2  # 가로 중앙
        text_y = (height + text_size[1]) // 2 + y_offset  # 세로 중앙 + y_offset
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    if hand_size < HAND_SIZE_MINI:
        draw_centered_text(warning_text, 0)  # 화면 중앙에 출력
    elif hand_size < HAND_SIZE_THRESHOLD:
        draw_centered_text(caution_text, -50)  # 경고 문구를 중앙보다 위에 출력

def get_direction(p1: Tuple[float, float], p2: Tuple[float, float]):
    if length_of(p1, p2) < stayLength:
        return "stay"
    tilt = get_tilt(p1, p2)
    if tilt == 1j or -1 >= tilt or tilt >= 1:
        return "down" if p1[1] > p2[1] else "up"
    else:
        return "right" if p1[0] > p2[0] else "left"

def get_actual_position(point: Tuple[float, float]) -> Tuple[float, float]:
    return point[0] * cap.get(cv2.CAP_PROP_FRAME_WIDTH), point[1] * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

def length_of(p1: Tuple[float, float], p2: Tuple[float, float]):
    p1 = get_actual_position(p1)
    p2 = get_actual_position(p2)
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2

def get_tilt(p1: Tuple[float, float], p2: Tuple[float, float]):
    p1 = get_actual_position(p1)
    p2 = get_actual_position(p2)
    return (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else 1j

def x_y():
    x1, y1 = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
    x2, y2 = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
    direction = get_direction((x1, y1), (x2, y2))

    if send_request_true:
        send_request(direction)

    cv2.putText(frame, f'direction : {direction}',
                (int(hand_landmarks.landmark[9].x * frame.shape[1]),
                 int(hand_landmarks.landmark[9].y * frame.shape[0]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)  # 창 모드를 명시적으로 설정

# 메인 실행 루프
with mp_hands.Hands(max_num_hands=hands_num, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                calculate_hand_size()
                x_y()

        cv2.imshow('frame', frame)

        # 키 입력 처리
        key = cv2.waitKey(1)
        if key == ord('q'):  # 종료
            break
        elif key == ord('1'):  # 숫자 '1'을 누르면 전체화면 전환
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, prop)

cap.release()
cv2.destroyAllWindows()
