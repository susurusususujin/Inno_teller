# Phase 2: ZSInvert 기반 스토리텔링 파이프라인

이 모듈은 `InnoWhitespaceExtractor` 프로젝트의 2단계를 담당합니다. 발굴된 "기술 공백(Whitespace)" 좌표 공간(고차원 임베딩 차원)을 ZSInvert (Zero-Shot Inversion) 알고리즘을 사용하여 사람이 읽을 수 있는 텍스트로 역추산(Reverse-engineer)합니다. 이를 통해 현재의 특허나 논문 생태계에서 연구되지 않고 누락된 유망 기술 주제를 시맨틱(의미론적) 형태로 찾아낼 수 있습니다.

---

## 💻 실행 환경 및 요구사항

이 파이프라인을 원활하게 구동하기 위해 다음 환경을 권장합니다:
* **하드웨어**: 다중 GPU 환경 필수 (Multi-GPU 분산 처리 요구, 예: RTX 4070 12GB + RTX 3070 Ti 8GB)
* **프레임워크**: `torch`, `transformers`, `sentence-transformers`, `accelerate` (멀티 GPU 필수 자원 관리), `pandas`
* **필수 로드 모델**: 
  - 인코더 (Encoder): `thenlper/gte-large`
  - 대형 언어 모델 (LLM): `Qwen/Qwen2.5-7B-Instruct` (약 14.5GB의 파라미터 VRAM 요구)
  - 오토인코더 가중치: `neurips_gte_ae_*.pth`

---

## 🏗️ 주요 코드 및 아키텍처 변경 내역

### 1. `InnoWhitespaceExtractor` 파이프라인 구조 최적화

**A. 타겟 영역(Zone) 필터링 로직 추가 (`run_zsinvert.py`)**
초기 코드는 657개의 배경 매트릭스 좌표 전체를 무작정 순회하며 디코딩하는 비효율성이 있었습니다. 실행 시간을 단축하고 통계적으로 의미 있는 군집만 골라낼 수 있도록 동적 필터링(예: Zone A, B, C 지정) 로직을 추가했습니다.
```python
# 전체 좌표 연산의 비효율성을 줄이고 주요 타겟 영역만 동적으로 필터링
target_zones = ['A', 'B', 'C']
vacancies_df = vacancies_df[vacancies_df['zone'].str.upper().isin(target_zones)]
```

**B. 생성 프롬프트 엔지니어링 강화 (`zeroshot.py`)**
ZSInvert 모델은 시작 컨텍스트(Prefix)에 매우 민감하게 반응하여 외계어나 한자를 쏟아내는 메모리 파괴 폭주(Collapse) 위험이 있었습니다. 범용적인 `prompt='tell me a story'` 지시문을 학술 영문 생성을 강제하는 최소 단위 닻(Anchor) 형태로 교체했습니다.
```python
# 빔 서치가 쓸데없는 단어를 찾지 않고 영어 학술 논문 컨텍스트에 갇혀 생성하도록 강제
attack.run_decoding(
    prompt='English research paper abstract:',
    ...
)
```

**C. 싱글톤 형태의 LLM 로딩 로직 분리 (`run_zsinvert.py` & `zeroshot.py`)**
기존 파이프라인은 수백 번 도는 `for` 반복문 내부에서 매번 거대한 7B 모델을 새로 인스턴스화하여 메모리 누수를 유발했습니다. 이를 방지하기 위해 로직을 모듈화(`init_zsinvert`, `zsinvert_decode`)하여, **거대 모델을 루프 바깥에서 단 1회만 메모리에 적재**하고 루프 간 파편화 방지(`torch.cuda.empty_cache()`) 처리를 수행하도록 개선했습니다.

---

### 2. `adversarial_decoding` 라이브러리 엔진 개선

**A. 멀티 GPU 분산 로딩 적용 (`emb_inv_decoding.py`)**
단독 모델 크기가 14.5GB에 달하는 LLM을 12GB 메모리의 단일 하드웨어에 강제 할당(`.to(device)`)할 경우 로딩 단계에서부터 즉각적인 OOM(Out of Memory) 셧다운이 발생했습니다. 
라이브러리 내부의 로딩 코드를 Hf `accelerate` 생태계 호환형으로 변경하여, 다중 GPU 환경에서 파라미터들을 안전하게 잘라서 분산 배치(Sharding)하도록 개선했습니다.
```python
# 단일 VRAM 풀에 억지로 고정하던 `.to(device)` 코드를 제거하고 Auto Map 채택
self.model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto" # 가용 중인 모든 GPU 자원을 동원하여 파라미터를 자동 분산
).eval()
```

**B. VRAM 폭파 방지를 위한 빔 서치(Beam Search) 그래프 제약 (`run_zsinvert.py`)**
토큰 선택 과정의 깊이가 길어질수록 KV 캐시 크기가 기하급수적으로 폭발하는 현상을 통제하기 위해, 탐색 공간을 명시적으로 제한했습니다.
* `beam_width=5` (기존 10 하드코딩에서 절반으로 하향 조절)
* `max_steps=32` (토큰 탐색 길이를 제약하여 모델 생성 한계선 확보)

---

## 📈 결과 파일 해석 가이드 (Output Analysis)

ZSInvert의 결과는 `5_inversion_results/zsinvert_results.csv` 경로에 점진적으로 저장됩니다.

* **주의사항**: 결과로 생성된 텍스트("An accurate abstraction could transform raw datasets... **enhanchsing accuracc**")에는 오타나 거친 문법 오류들이 존재할 수 있습니다.
  이는 ZSInvert 알고리즘 고유의 특성 때문입니다. ZSInvert는 사람이 읽기 예쁜 문장 생성 규칙을 일부 무력화하고, 역산해야 할 목표 임베딩 1024차원 좌표값에 **수학적 거리가 가장 가까운 토큰 위치만을 쫓아가는 탐욕(Greedy) 방식의 알고리즘**입니다(실제로 코사인 유사도 >0.93 근접).
* **Next Action**: 파이프라인이 뱉어낸 이 "매우 거칠지만 정교하게 타겟된 원시 뼈대 키워드"들을 묶어, 챗GPT나 별도의 다듬기 모델 프롬프트로 한 번 덧입혀주는(Refinement) 후처리 작업을 진행하시면 매끄러운 형태의 완전한 사업화 아이디어 초록이 튀어나오게 됩니다!
