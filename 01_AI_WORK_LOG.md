# VibePoser AI Work Log

Last updated: 2026-06-17

## 현재 상태
- 이 저장소는 `VibePoser`라는 Python/Tkinter 기반 GUI 툴을 담고 있다.
- 메인 실행 파일은 `pose_app.py`이며, `VibePoser.bat`와 Pixi 환경(`pixi.toml`, `pixi.lock`)을 통해 실행하는 구조로 보인다.
- 현재 인수인계 문서는 `AGENT.md`에 있으며, 아키텍처, 메모리 이슈, GitHub 개인 백업 정책, 안정화 패치 메모가 정리되어 있다.
- `00_ai_guidelines.txt`는 읽기 전용 가이드라인이다. 수정하지 않는다.

## 툴 개요
VibePoser는 단일 이미지에서 사람의 3D 포즈를 추출하고, 이를 SMPL/SMPL-X 기반 포즈 데이터로 변환한 뒤 OSC로 외부 프로그램에 송출하기 위한 로컬 GUI 도구다.

주요 흐름:
1. 사용자가 Tkinter GUI에서 이미지를 불러온다.
2. SAM 3D Body 계열 모델이 이미지에서 3D 신체 포즈와 메쉬 정보를 추론한다.
3. MHR/SMPL 변환 단계에서 추론 결과를 SMPL-X 포즈 파라미터와 조인트 회전값으로 변환한다.
4. 변환된 포즈 데이터를 OSC UDP 메시지로 언리얼 엔진, VRChat, 기타 수신 프로그램에 보낼 수 있다.

## 주요 구성
- `pose_app.py`: GUI, 모델 로딩, 추론, 변환, OSC 송출, 로그 출력이 들어 있는 중심 파일.
- `sam-3d-body/`: SAM 3D Body 관련 소스 코드.
- `sam-3d-body-dinov3/`: SAM 3D Body/DINOv3 계열 모델 가중치 위치.
- `MHR-main/`: MHR 및 SMPL 변환 관련 코드.
- `smplx/`: SMPL-X 모델 데이터 위치.
- `logs/`: 실행 중 생성되는 VibePoser 로그 파일 위치.
- `AGENT.md`: 현재 상세 아키텍처 및 인수인계 문서.

## 기술 스택
- Python 3.12
- Tkinter GUI
- PyTorch/CUDA
- OpenCV
- NumPy/SciPy
- Trimesh
- Matplotlib
- python-osc
- smplx
- Pixi 환경 관리

## 현재 알려진 설계 이슈
- SAM3D/DINOv3/PyTorch 모델을 같은 Python 프로세스 안에서 반복 로딩/삭제하면 RAM/VRAM 누수가 발생했던 이력이 있다.
- 현재는 모델을 한 번 로딩한 뒤 앱이 켜져 있는 동안 계속 재사용하는 방식으로 타협한 상태다.
- 이 방식은 반복 실행 시 메모리 증가를 막지만, 앱 실행 중에는 약 5-6GB 수준의 VRAM을 계속 점유할 수 있다.
- 2026-06-17 기준 Tkinter UI 갱신, 웹캠 루프 중복 방지, 슬라이더 OSC debounce 안정화 패치가 적용되었다.
- 2026-06-17 기준 SMPL 변환 결과는 GPU 텐서가 아니라 CPU 캐시로 보관하고, preview/OSC 계산 시에만 임시로 모델 디바이스에 올린다.
- 2026-06-17 기준 preview mesh 경량화가 적용되었다. mesh는 face stride로 일부만 그리고, 회전 슬라이더 드래그 중에는 skeleton만 갱신한다.

## 다음 할 일
- 큰 구조 개선을 한다면 `pose_app.py`의 무거운 PyTorch 추론/변환 로직을 별도 워커 프로세스로 분리한다.
- GUI 메인 프로세스는 가볍게 유지하고, 추론 버튼 클릭 시 별도 프로세스를 띄운 뒤 결과만 Queue/Pipe/파일 등으로 돌려받는 구조를 검토한다.
- MHR -> SMPL 빠른 모드를 만들려면 외부 라이브러리 성격의 `MHR-main/tools/mhr_smpl_conversion` 내부에 iterations 인자를 뚫어야 하므로, 안정 버전에서는 보류한다.
- CUDA OOM, 모델 로딩 실패, OSC 송출 실패가 UI 전체 크래시로 이어지지 않도록 에러 전달/표시 방식을 정리한다.
- 작업 전에는 항상 이 파일과 `AGENT.md`를 읽고, 이전 실패 이력을 반복하지 않는다.

## 오류 및 해결 이력
- 2026-05-11: `00_ai_guidelines.txt`를 PowerShell 기본 인코딩으로 읽으면 한글이 깨져 보였다.
  - 해결: `Get-Content -Encoding UTF8`로 다시 읽어 정상 확인했다.
- 이전 이력: 모델을 매번 `del`로 삭제하고 재로딩하는 방식에서 실행 1회당 약 665MB씩 메모리가 증가하는 누수 문제가 있었다.
  - 현재 대응: 모델 삭제/재로딩을 피하고, 앱 실행 중 모델을 캐시하여 재사용한다.
  - 장기 해결 방향: PyTorch 모델 작업을 별도 프로세스로 격리하고 작업 종료 시 프로세스를 종료해 OS 레벨에서 메모리를 회수한다.

## 이번 세션 기록
- 2026-06-17: GitHub 원격 `https://github.com/dmustud/vibePoser.git`를 연결하고 개인 백업용 첫 커밋을 푸시했다.
- 2026-06-17: 대형 외부 자산과 런타임 산출물을 `.gitignore`에 제외했다.
  - 제외 대상: `.pixi/`, `logs/`, `backup/`, `__pycache__/`, `sam-3d-body/`, `MHR-main/`, `sam-3d-body-dinov3/`, `smplx/`, ML 모델 파일 확장자.
- 2026-06-17: `AGENT.md`에 GitHub 백업 정책과 외부 모델/라이브러리 운용 방식을 추가했다.
- 2026-06-17: 1차 안정화 패치를 진행했다.
  - Tkinter UI 갱신용 `run_on_ui(...)`, `set_status(...)` 헬퍼를 추가했다.
  - 로그, 로딩 애니메이션, 모델 초기화 상태, 이미지 표시, 3D plot 갱신을 메인 UI 스레드로 우회시켰다.
  - 웹캠 전환 시 이전 루프가 중복 실행되지 않도록 세대 번호와 lock/thread 관리를 추가했다.
  - 웹캠 프레임 UI 예약이 무한히 쌓이지 않도록 frame pending 플래그를 추가했다.
  - 추론 시작 시 `current_image` 복사본을 사용해 웹캠 갱신과의 경합을 줄였다.
  - 슬라이더 변화에 따른 OSC 전송을 150ms debounce로 묶고, UI 값 캡처 후 별도 스레드에서 OSC를 송출하도록 바꿨다.
  - `python -m py_compile pose_app.py`로 문법 확인을 통과했다.
- 2026-06-17: 2차 메모리 최적화 패치를 진행했다.
  - `self.smpl_result`를 GPU 텐서 clone이 아니라 CPU 캐시로 보관하도록 변경했다.
  - `cache_smpl_result_on_cpu(...)`, `smpl_result_to_device(...)`, `smpl_result_to_numpy(...)` 헬퍼를 추가했다.
  - SMPL preview와 OSC grounding 계산 시에만 CPU 캐시를 임시로 모델 디바이스에 올리도록 바꿨다.
  - OSC payload 생성은 CPU/NumPy 데이터를 사용하도록 정리했다.
  - `python -m py_compile pose_app.py`로 문법 확인을 통과했다.
- 2026-06-17: SMPL 변환 체감 지연 원인 확인을 위해 성능 계측 로그를 추가했다.
  - `[PERF] SMPL convert/input vertices to device`
  - `[PERF] SMPL convert/convert_mhr2smpl`
  - `[PERF] SMPL convert/cache result on CPU`
  - `[PERF] SMPL convert/params back to device for preview`
  - `[PERF] SMPL convert/smplx preview forward`
  - `[PERF] SMPL convert/preview arrays to CPU`
  - `[PERF] SMPL convert/auto OSC send`
  - `[PERF] OSC/...`
  - CUDA 비동기 타이밍 왜곡을 줄이기 위해 측정 전 `torch.cuda.synchronize()`를 호출한다.
- 2026-06-17: 현재 사용 버전용 preview 경량화를 적용했다.
  - `메시 보기 (Light)`로 UI 문구를 바꾸고, mesh preview face를 `mesh_preview_face_stride = 8`로 일부만 렌더링한다.
  - 회전 슬라이더 드래그 중에는 mesh 렌더링을 생략하고 skeleton만 갱신한다.
  - 슬라이더 버튼을 놓으면 mesh preview를 다시 갱신한다.
  - 중복으로 pack되던 `손 모드` 체크박스 호출을 제거했다.
- 2026-06-17: MHR -> SMPL 변환 옵션을 조사했다.
  - 공개 `convert_mhr2smpl(...)` API에는 반복 횟수/품질 preset 인자가 없다.
  - 내부 `PyTorchSMPLFitting._optimize_smpl(...)` 기본 반복 수는 `((40, 80, 40), 300)`이다.
  - 빠른 모드는 가능해 보이지만 `MHR-main/tools/mhr_smpl_conversion` 내부 API 수정이 필요하므로 현재 안정 버전에는 적용하지 않았다.

## 이전 세션 기록
- `00_ai_guidelines.txt`를 읽고 프로젝트 작업 규칙을 확인했다.
- 기존 `handover.md`, `pose_app.py`, `pixi.toml`을 확인해 VibePoser의 목적과 현재 구조를 요약했다.
- 신규 작업 로그 파일 `01_AI_WORK_LOG.md`를 생성했다.
- `pose_app.py`에서 SMPL 변환 관련 UI 동작을 수정했다.
  - 핫키 기본값을 `\`에서 `-`로 변경했다.
  - 핫키 입력 시 기존 원샷 흐름 대신 `2. SMPL 변환 (+전송)` 버튼 동작을 직접 호출하도록 연결했다.
  - 포즈 데이터가 없어 자동 추출을 먼저 실행한 뒤 `포즈 자동 추출 완료 -> SMPL 변환 중 0%` 메시지가 표시되도록 했다.
  - SMPL 변환 진행 콜백에서 상태 문구에 현재 퍼센트를 표시하도록 했다.
  - SMPL 변환 완료 시 Windows 알림음을 울리도록 했다.
  - `pixi run python -m py_compile pose_app.py`로 문법 확인을 통과했다.
