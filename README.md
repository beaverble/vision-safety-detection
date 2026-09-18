# 실시간 화재·침입 탐지 (비전 AI)

현장 CCTV 영상에서 **화재(불꽃·연기)** 와 **무단 침입**을 실시간 탐지하고 경보를 전송하는 비전 AI 파이프라인이다.
YOLOv9 계열 객체탐지 모델을 기반으로 하며, 데이터·모델 가중치(`*.pt`)는 포함하지 않는다.

<br>

## 구성

- **다중 스트림 탐지 클라이언트** (`msc.py`)
  - 여러 CCTV 스트림을 동시에 처리, 탐지 결과를 **ZeroMQ로 경보 전송**
- **다중 타깃 처리** (`mtp.py`)
  - 관심 객체 크롭·저장 및 후처리, 로깅
- **모델 학습/검증/추론** (YOLOv9 계열)
  - `train.py`, `train_dual.py`, `train_triple.py`
  - `val.py`, `val_dual.py`, `val_triple.py`
  - `detect.py`, `export.py`

<br>

## 메모

- 학습 데이터·모델 가중치(`*.pt`)·현장 영상은 저장소에 포함하지 않는다.
- 클래스 정의(`names`)는 학습된 가중치에서 로드된다.
- 경로·서버 주소는 상대경로/플레이스홀더로 정리했다.

<br>

## 기술 스택

Python · PyTorch · YOLOv9 · OpenCV · ZeroMQ
