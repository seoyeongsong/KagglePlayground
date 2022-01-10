# Facial Landmark - OpenCv, dlib

# cv2 라이브러리는 영상을 처리하기 위해 사용
# 얼굴인식, 물체 식별, 이미지 결합 등의 작업에 사용
# 설치 시 커맨드창에 'pip install opencv-python'
import cv2
# dlib 라이브러리는 얼굴인식(recognition) 모듈을 제공
# 얼굴의 landmark를 찾아내는 작업에 사용
# dlib 설치 중 에러 발생 시, Visual Studio(for C++) 설치 후 cmake와 dlib 설치
# 'pip install cmake', 'pip install opencv-contrib-python dlib'
import dlib
# imutils 라이브러리는 이미지의 처리를 위해 사용
# 회전, 사이즈 조정, 변형, 뼈대 구하기(skeletonization) 등에 사용
# 
from imutils import face_utils, resize

# numpy 라이브러리는 array를 처리하기 위해 사용
# 이미지 데이터를 numpy 배열로 변환에 사용
import numpy as np

# 실행할 코드의 파일과 같은 경로에 오렌지 이미지를 저장한 후 이미지 불러오기
# cv2.imread(fileName, flag) : 이미지 읽기
# 읽어온 이미지는 numpy의 n차 배열

# 이미지 파일 경로는 절대 경로로 설정해야 하며, escape 주의
pre_path = 'C:\\Users\\seoyeong\\Desktop\\study\\KagglePlayGround\\1.AnnoyingOrange'
img_file_path = pre_path + '\\orange.jpg'
orange_img = cv2.imread(img_file_path)

# orange_img.shape 으로 배열 확인가능
print(orange_img.ndim)  # array의 차원 (3차원)
print(orange_img.shape) # array의 크기

# 이미지 파일이 너무 클 경우 resize
orange_img = cv2.resize(orange_img, dsize=(512, 512))

# 이미지를 띄워 확인, waitKey를 주어야 창이 그대로 출력되고 없으면 즉시 close됨
# cv2.imshow('1', orange_img)
# cv2.waitKey(0)

# dlib의 얼굴 인식기 사용
# 1. detector를 사용해 얼굴을 먼저 찾는다.
detector = dlib.get_frontal_face_detector()
# 2. predictor를 사용해 얼굴의 landmark를 예측한다.
# predictor를 사용하기 위해서는 shape_predictor_68_face_landmarks.dat 파일이 필요하므로 미리 다운로드해둔다.
# 출처 : http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor(pre_path+'\\shape_predictor_68_face_landmarks.dat')

# VideoCapture : 동영상 입력을 관리하는 함수
# 인수 : 파일명 또는 연결된 장치
# cv2.VideoCapture(0) : 웹캠을 통한 비디오 입력
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(pre_path + '\\woman.mp4')

# VideoCapture 객체가 생성되어 있는 동안 아래 코드를 실행
while cap.isOpened():

    # 영상을 한 프레임씩 읽어 정상이면 ret=True, 실패하면 False
    # img에는 읽은 프레임을 저장
    ret, img = cap.read()

    # 프레임이 없으면 while문 중단
    if not ret:
        break

    # 읽은 프레임에서 얼굴을 인식
    faces = detector(img)
    # result에 orange 이미지의 복사본을 저장
    result = orange_img.copy()

    # 얼굴이 존재할 경우 아래 코드를 실행
    if len(faces) > 0:
        # 여러 얼굴이 존재할 경우, 하나의 얼굴을 사용하기 위해 첫번째 인덱스만 담기
        face = faces[0]
        
        # 얼굴 영역만 자르기 위해 좌표를 구한다.
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        # 프레임에서 얼굴 영역만을 잘라 복사본을 face_img 변수에 담는다.
        face_img = img[y1:y2, x1:x2].copy()

        # Face Landmark 68개의 점 구하기
        shape = predictor(img, face)
        shape = face_utils.shape_to_np(shape)

        # 자른 얼굴 영역을 확인하기
        # for p in shape:
        #     cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        # eyes
        # 왼쪽눈 영역(36~39, 37~41)
        le_x1 = shape[36, 0]
        le_y1 = shape[37, 1]
        le_x2 = shape[39, 0]
        le_y2 = shape[41, 1]
        # 0.18의 margin을 부여
        le_margin = int((le_x2 - le_x1) * 0.18)

        # 오른쪽눈 영역 ()
        re_x1 = shape[42, 0]
        re_y1 = shape[43, 1]
        re_x2 = shape[45, 0]
        re_y2 = shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.18)

        # 위에서 얻은 눈 영역 좌표와 margin을 적용해 프레임에서 추출한 눈 저장
        left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy()
        right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()
        # 눈 영역 resize
        left_eye_img = resize(left_eye_img, width=100)
        right_eye_img = resize(right_eye_img, width=100)

        # 이미지 합성에서 블렌딩과 마스킹을 쉽게 도와주는 함수
        # 1. 왼쪽 눈 합성
        result = cv2.seamlessClone(
            left_eye_img,   # 입력이미지(전경)
            result,         # 대상이미지(배경)
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),   #mask(합성하고자 하는 영역 : 255, 나머지 : 0)
            (120, 230),     # 입력이미지가 위치할 배경에서의 좌표
            cv2.MIXED_CLONE # 합성 방식 (MIXED_CLONE : 경계가 blur 처리되어 합성)
        )

        # 2. 오른쪽 눈 합성
        result = cv2.seamlessClone(
            right_eye_img,
            result,
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            (255, 230),
            cv2.MIXED_CLONE
        )

        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        mouth_img = resize(mouth_img, width=250)

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (190, 350),
            cv2.MIXED_CLONE
        )
        # 각 영역 crop한 결과를 확인
        # cv2.imshow('left', left_eye_img)
        # cv2.imshow('right', right_eye_img)
        # cv2.imshow('mouth', mouth_img)
        # cv2.imshow('face', face_img)

        # 최종 결과물인 Annoying Orange를 확인
        cv2.imshow('result', result)

    # cv2.imshow('img', img)
    # 영상이 뜨고, q를 누르기 전까지 창이 활성화
    if cv2.waitKey(1) == ord('q'):
        break