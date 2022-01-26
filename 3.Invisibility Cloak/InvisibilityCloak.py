# ==============================
# 라이브러리 import
# ==============================
import cv2
import numpy as np
# python에서는 CLI argument를 처리하기 위해 argparse 모듈을 제공
import time, argparse

# ==============================
# 저장된 동영상 파일 불러오기
# ==============================

# 터미널에서 python 파일을 실행할 때 명령어에 대한 옵션을 parameter로 전달할 때 사용
# 인자값을 받아 처리하는 로직
parser = argparse.ArgumentParser()
# 입력받을 인자값을 등록
# add_argument의 help 옵션은 인자가 하는 일에 대한 간단한 설명을 의미한다.
# command line에서 python 파일을 실행할 때, [python 파일명].py --video [video 파일 경로] 입력 시
# 저장된 동영상 파일을 불러와서 작업할 수 있다.
parser.add_argument("--video", help='Input video path')
# 입력받은 인자값을 args에 저장
args = parser.parse_args()

# command line에서 입력받은 경로에 영상이 존재하면 사용하고 없을 경우 웹캠을 사용한다.
cap = cv2.VideoCapture(args.video if args.video else 0)

# 웹캠을 사용할 경우, 웹캠이 켜지는데 시간이 소요되므로 3초간 timesleep을 준다.
time.sleep(3)

# 사전 준비 : 동영상의 앞 2~5초 정도는 사람없이 배경만 존재하도록 한다.
# Grap background image from first part of the video
# 60 frame에 대해서 사람이 없는 배경을 저장한다.
for i in range(60):
  # 읽은 프레임을 background에 저장하고, 
  # 비디오 프레임을 제대로 읽은 경우 ret = True, 아니면 False
  ret, background = cap.read()
print(background.shape) # (480, 852, 3)

# ==============================
# 결과 동영상 저장 객체 생성하기
# ==============================

# 영상을 저장하기 위해 cv2.VideoWriter 객체를 생성
# fourcc는 Codec 정보를 저장 (MP4V는 MPEG-4의 약자)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

# cv2.VideoWriter([저장 파일명], Codec정보, 초당 저장될 frame, 저장될 사이즈)
# 카메라 또는 영상의 속성을 확인하기 위해 cap.get(id) 사용
# cv2.CAP_PROP_FPS : 초당 프레임 수
# 영상 저장될 사이즈(list) 예 : (640, 480)
out1 = cv2.VideoWriter('result.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))
out2 = cv2.VideoWriter('remove_red.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))
#out3 = cv2.VideoWriter('mask.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))

# ==============================
# 연속된 동영상 읽기
# ==============================

# 동영상의 첫 frame만 cap에 담기게 되는데, cap객체가 지정한 파일로 정상적으로 초기화가 된 경우
# while 문을 통해 무한루프를 돌면서 연속된 프레임을 읽을 수 있고
# frame을 정상적으로 읽었다면 ret은 True, frame은 img 변수에 담는다. 
while(cap.isOpened()):
  ret, img = cap.read()
  if not ret:
    break
  
  # ==============================
  # HSV Color System
  # openCV에서는 Hue [0,179] / Saturation, Value [0,255]로 정의한다.
  # ==============================

  # Convert the color space from BGR to HSV
  # 컬러 시스템에서 HSV가 사람이 인식하는 것과 유사하다.
  # cv2.cvtColor() : 색상 공간을 변환할 때 사용
  # cv2.cvtColor([입력이미지], [색상변환코드]), 결과는 matrix
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  #print("==hsv==",hsv)

  # Generate mask to detect red color
  # 빨간색을 추출하여 mask를 생성한다.
  # HSV 시스테에서는 빨간색 영역이 2개이므로 mask를 각각 생성한 뒤 더한다.
  lower_red = np.array([0, 120, 70])
  upper_red = np.array([10, 255, 255])
  # cv2.inRange 특정 색상 영역을 추출
  # cv2.inRange([입력행렬], [하한값 행렬], [상한값 행렬])
  mask1 = cv2.inRange(hsv, lower_red, upper_red)

  lower_red = np.array([170, 120, 70])
  upper_red = np.array([180, 255, 255])
  mask2 = cv2.inRange(hsv, lower_red, upper_red)

  mask1 = mask1 + mask2

  # lower_black = np.array([0, 0, 0])
  # upper_black = np.array([255, 255, 80])
  # mask1 = cv2.inRange(hsv, lower_black, upper_black)

  '''
  # Refining the mask corresponding to the detected red color
  https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
  '''
  # Remove noise
  # 모폴로지 연산 : 다양한 영상 처리 시스템에서 전처리 또는 후처리 형태
  mask_cloak = cv2.morphologyEx(mask1, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
  mask_cloak = cv2.dilate(mask_cloak, kernel=np.ones((3, 3), np.uint8), iterations=1)
  mask_bg = cv2.bitwise_not(mask_cloak)

  cv2.imshow('mask_cloak1', mask_cloak)

  # Generate the final output
  res1 = cv2.bitwise_and(background, background, mask=mask_cloak)
  res2 = cv2.bitwise_and(img, img, mask=mask_bg)
  # 두 개의 이미지를 더한다.
  result = cv2.addWeighted(src1=res1, alpha=1, src2=res2, beta=1, gamma=0)

  cv2.imshow('res1', res1)

  # cv2.imshow('ori', img)
  cv2.imshow('result', result)
  out1.write(result)      # 최종결과 영상
  out2.write(res1)        # Red 영역 지운 영상
  #out3.write(mask_cloak)  # mask 

  # 'q' 키가 눌리면 정지
  if cv2.waitKey(1) == ord('q'):
    break

out1.release()
out2.release()
#out3.release()
cap.release()