# 명함 검출 및 인식

# 일반적인 명함 사진의 조건
# - 명함은 흰색 배경에 검정색 글씨이다
# - 명함은 충분히 크게 촬영되었다
# - 명함은 각진 사각형 모양이다

# 명함 검출 및 인식 진행 과정
# 이진화 -> 외곽선 검출&다각형 근사화 -> 투영 변환 -> OCR

import sys
import cv2

src = cv2.imread('./000_opencv/namecard/namecard1.jpg')
# src = cv2.imread('./000_opencv/namecard/namecard2.jpg') # namecard1 보다 밝게 찍힌 사진
# src = cv2.imread('./000_opencv/namecard/namecard3.jpg') # namecard1 보다 밝게 찍힌 사진

if src is None:
    print("image load failed")
    sys.exit()

# src = cv2.resize(src, (640, 480))
src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)

# convert color to gray scale
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# binarization 
# _, src_bin = cv2.threshold(src_gray, 130, 255, cv2.THRESH_BINARY)
# using OTSU
th, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# OTSU로 자동 임계값을 설정할 것이기 때문에 기존에 줬던 threshold 값 130을 0으로 변경하면 된다
# threshold가 0인 것이 아니라 자동으로 설정되기 때문에 값을 주지 않는 것이라고 보면 된다
# 130을 줘도 무시하고 자동으로 설정되지만 소스코드를 이해하기 더 쉽게끔 이렇게 만들어주는 것이 좋다
print(th) # 151.0, 자동으로 적용된 임계값

contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contours))


cv2.imshow('src', src)
cv2.imshow('src_gray', src_gray)
cv2.imshow('src_bin', src_bin)

cv2.waitKey()
cv2.destroyAllWindows()

# 1. 이진화
# 영상의 이진화(Binarization)란?
# - 영상의 픽셀 값을 0(black) 또는 1(255)(white)로 만드는 연산
# - 배경(Background) vs. 객체(Object)
# - 관심 영역 vs. 비관심 영역

# 그레이스케일 영상의 이진화
# g(x, y) = 0 (if f(x, y) <= T)  /  255 (if f(x, y) > T) -> T : 임계값, threshold

# 임계값 함수
# cv2.threshold(src, thresh, maxval, type, dst=None)->retval, dst
# - src : 입력 영상(다채널, 8비트 또는 32비트 실수형)
# - thresh: 임계값
# - maxval : THRESH_BINARY 또는 THRESH_BINARY_INV 방법을 사용할 때의 최댓값 지정(cv.ThresholdTypes)
# - retval : 사용된 임계값
# - dst : (출력) 임계값 영상(src와 동일 크기, 동일 타입)

# cv2.ThresholdTypes
# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# 아래의 타입들은 완전한 이진화는 아님
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_MASK
# 특히 이 아래 2개는 임계값 자동 결정 방법에 속함
# cv2.TRHESH_OTSU : Otsu 알고리즘으로 임계값 결정
# cv2.THRESH_TRIANGLE : 삼각 알고리즘으로 임계값 결정

# 이진화 : 임계값 결정 방법
# 자동 임계값 결정 방법 : Otsu 방법
# - 입력 영상이 배경과 객체 두 개로 구성되어 있다고 가정 -> bimodal histogram
# - 두 픽셀 분포의 분산의 합이 최소가 되는 임계값을 선택(Minimize within-class variance)
# - 효과적인 수식 전개와 재귀식을 이용하여 빠르게 임계값을 결정
# Otus 가 무조건 best 임계값을 설정해주는 건 아니다
# 경우에 따라 Otus에 의해서 정해진 임계값에 우리가 임의로 10을 더하거나 빼줌으로서 오히려 더 좋은 결과가 나올 수도 있다
# 경험치가 중요

# 객체 단위 분석
# - 객체 단위 분석 : 흰색으로 표현된 객체를 분할하여 특징을 분석, 객체 위치 및 크기 정보, ROI 추출
# 1) 레이블링(Connected Component Labeling)
# - cv2.connectedComponent(), cv2.connectedComponentWithStats()
# - 서로 연결되어 있는 객체 픽셀에 고유한 번호를 지정
# - 각 객체의 바운딩 박스, 무게 중심 좌표로 함께 반환
# 2) 외곽선 검출(Contour Tracing)
# - cv2.findContours()
# - 각 객체의 외곽선 좌표를 모두 검출

# 레이블링이 좀 더 빠르다
# 객체가 어느 위치에 있는지 또는 크기가 어떤지 또는 무게중심 좌표를 알고 싶다면 
# 레이블링을 사용하면 된다
# 그게 아니라 어떤 객체를 찾고 그 객체가 어떤 모양이고 그 모양 안에 홀이라는 것이 있는지 없는지를 
# 파악하고 싶다면 외곽선 검출, findContours를 쓰면 된다

# 사각형의 명함을 찾을 것이기 때문에 외곽선의 모양이 사각형에 가까운지를 알아보기 위해
# 외곽선 검출을 해 볼 것이다

# 외곽선 검출 : OpenCV API
# 외곽선 검출(Cont)
# cv2.findContours(image, mode, method, contours = None, hierarchy = None, offset = None) -> contours, hierarchy
# - image : 입력 영상, non-zero 픽셀을 객체로 간주함 / 보통 이진화 되어있는 영상으로 준다
# - mode : 외곽선 검출 모드
# - 1) RETR_EXTERNAL : 가장 바깥쪽 외곽선만 검출(hierarchy[i][2] = hierarchy[i][3]=-1)
# - 2) RETR_LIST : 계층에 관계없이 모든 외곽선 검출(hierarchy[i][2] = hierarchy[i][3]=-1)
# - 3) RETR_CCOMP : 2레벨 계층 구조로 외곽선 검출, 상위 레벨은 (흰색) 객체 외곽선, 하위 레벨은 (검정색) 구멍(hole) 외곽선
# - 4) RETR_TREE : 계층적 트리 구조로 모든 외곽선 검출
# - 1, 2 : 계층 정보 X, 3, 4 : 계층 정보 O
# - mode를 선택하는 것이 중요!

# - method : 외곽선 근사화 방법
# - 1) CHAIN_APPROX_NONE : 근사화 없음
# - 2) CHAIN_APPROX_SIMPLE : 쉭선, 수평선, 대각선에 대해 끝점만 사용하여 압축
# - 3) CHAIN_APPROX_TC89_L1 : Teh & Chin L1 근사화
# - 4) CHAIN_APPROX_TC89_KCOS : Teh & Chin k cos 근사화
# - 3, 4는 거의 안 쓰고 주로 1 아님 2를 사용
# - 깊이 들어갈 생각이 없으면 NONE을 쓰는 것이 가장 무난하다

# - contours : 검출된 외곽선 좌표. numpy.ndarray로 구성된 리스트, len(contours) = N, contours[i].shape = (K, 1, 2)
# - hierarchy : 외곽선 계층 정보, numpy.ndarray, shape = (1, N, 4), hierarchy[0, i, 0~3]가 순서대로
# next, prev, child, parent 외곽선을 가리킴, 해당 외곽선이 없으면 -1을 가진다

# 외곽선 검출이란?
# - 객체의 외곽선 좌표를 모두 추출하는 작업
# - 바깥쪽 & 안쪽(홀) 외곽선
# - 외곽선의 계층 구조도 표현 가능
# 객체 하나의 외곽선 표현 방법
# - numpy.ndarray
# - shape = (K, 1, 2), dtype = int32(K는 외곽선 좌표 개수)
# 여러 객체의 외곽선 표현 방법
# - "객체 하나의 외곽선"을 원소로 갖는 리스트
# - 리스트의 길이 = 외곽선 개수

# 외곽선 관련 OpenCV API
# 1. 면적 구하기
# - cv2.contourArea(contour, oriented = None) -> retval
# - contour : 외곽선 좌표. numpy.ndarray.shape=(K, 1, 2)
# - oriented : True이면 외곽선 진행 방향에 따라 부호 있는 면적을 반환
# - retval : 외곽선으로 구성된 면적

# 2. 외곽선 길이 구하기
# - cv2.arcLength(curve, closed) -> retval
# - curve : 외곽선 좌표. numpy.ndarray.shape=(K, 1, 2)
# - closed : True이면 폐곡선으로 간주
# - retval : 외곽선의 길이

# 3. 바운딩박스 (외곽선을 외접하여 둘러싸는 가장 작은 사각형) 구하기
# - cv2.boundingRect(array) -> retval
# - array : 외곽선 좌표, numpy.ndarray.shape = (K, 1, 2)
# - retval : 사각형(x, y, w, h)

# 4. 바운딩 서클(외곽선을 외접하여 둘러싸는 가장 작은 원) 구하기
# - cv2.minEnclosingCircle(points) -> center, radius
# - points : 외곽선 좌표, numpy.ndarray.shape = (K, 1, 2)
# - center : 바운딩 서클 중심 좌표(x, y)
# - radius : 바운딩 서클 반지름

# 5. 외곽선 근사화
# - cv2.approxPolyDP(curve, epsilon, closed, approxCurve = None) -> approxCurve
# - curve : 입력 곡선 좌표, numpy.ndarray.shape=(K, 1, 2)
# - epsilon : 근사화 정밀도 조절, 입력 곡선과 근사화 곡선 간의 최대 거리(e.g. 외곽선 전체 길이 * 0.02)
# - closed : Ture를 전달하면 폐곡선으로 간주
# - approxCurve : 근사화된 곡선 좌표 numpy.ndarray.shpae = (K', 1, 2)
# ** DP == Douglas-Peucker algorithm 외곽선 근사화 알고리즘인데, opencv에서 제공
# 근사화 : 점이 되게 많은데 그 점들을 단순화 시키겠다는 의미
