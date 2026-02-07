# ZKP 테스트 결과

---

# Groth16 단위 테스트 결과

## 실행 환경

| 항목 | 값 |
|------|-----|
| Python | 3.13.3 |
| pytest | 9.0.2 |
| OS | macOS (Darwin 25.2.0) |
| 실행일 | 2026-02-08 |

## 실행 방법

```bash
cd /Users/kevin/dev/interactive-zkp-study
source .venv/bin/activate
pytest tests/ -v
```

## 전체 결과 요약

```
148 passed, 91 warnings in 52.00s
```

| 지표 | 값 |
|------|-----|
| 총 테스트 수 | 148 |
| 통과 | 148 |
| 실패 | 0 |
| 경고 | 91 (ast.Num deprecation) |
| 소요 시간 | ~52초 |

## 모듈별 테스트 결과

### test_determinant.py (33 tests) - ALL PASSED

순수 Python 행렬 연산 함수 16개에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestZerosMatrix | 2 | PASSED |
| TestIdentityMatrix | 2 | PASSED |
| TestCopyMatrix | 1 | PASSED |
| TestTranspose | 2 | PASSED |
| TestMatrixAddition | 2 | PASSED |
| TestMatrixSubtraction | 2 | PASSED |
| TestMatrixMultiply | 3 | PASSED |
| TestMultiplyMatrices | 1 | PASSED |
| TestCheckMatrixEquality | 4 | PASSED |
| TestDotProduct | 2 | PASSED |
| TestUnitizeVector | 2 | PASSED |
| TestCheckSquareness | 2 | PASSED |
| TestDeterminantRecursive | 3 | PASSED |
| TestDeterminantFast | 3 | PASSED |
| TestCheckNonSingular | 2 | PASSED |

> **참고**: `determinant_fast`는 대각선이 0인 경우 `1.0e-18`을 대입하는 로직이 있어, singular matrix에서도 정확히 0을 반환하지 않는다. `check_non_singular` 테스트는 이 동작을 반영한다.

### test_code_to_r1cs.py (25 tests) - ALL PASSED

AST 파싱 및 R1CS 제약조건 변환 함수 13개에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestParse | 2 | PASSED |
| TestExtractInputsAndBody | 3 | PASSED |
| TestFlatten | 6 | PASSED |
| TestSymbol | 2 | PASSED |
| TestInsertVar | 4 | PASSED |
| TestGetVarPlacement | 1 | PASSED |
| TestFlatcodeToR1cs | 1 | PASSED |
| TestGrabVar | 3 | PASSED |
| TestAssignVariables | 1 | PASSED |
| TestCodeToR1csWithInputs | 2 | PASSED |

> **주의사항**: `code_to_r1cs.py`의 전역 심볼 카운터(`next_symbol`)는 `autouse` fixture에서 매 테스트 전 `initialize_symbol()`로 초기화된다.

### test_qap_creator.py (20 tests) - ALL PASSED

다항식 연산 및 QAP 변환 함수 11개에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestMultiplyPolys | 3 | PASSED |
| TestAddPolys | 3 | PASSED |
| TestSubtractPolys | 2 | PASSED |
| TestDivPolys | 2 | PASSED |
| TestEvalPoly | 3 | PASSED |
| TestMkSingleton | 1 | PASSED |
| TestLagrangeInterp | 2 | PASSED |
| TestTranspose | 1 | PASSED |
| TestR1csToQap | 1 | PASSED |
| TestCreateSolutionPolynomials | 1 | PASSED |
| TestCreateDivisorPolynomial | 1 | PASSED |

### test_qap_creator_lcm.py (13 tests) - ALL PASSED

LCM(행렬식) 변형 QAP 특화 함수에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestKMatrix | 4 | PASSED |
| TestR1csToQapTimesLcm | 2 | PASSED |
| TestCreateSolutionPolynomialsLcm | 1 | PASSED |
| TestTransposeLcm | 1 | PASSED |
| TestPolyOpsLcm | 5 | PASSED |

### test_poly_utils.py (18 tests) - ALL PASSED

FR(BN128 curve order) 필드 위 다항식 연산 함수 17개에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestFRPolynomials | 5 | PASSED |
| TestVectorOps | 2 | PASSED |
| TestUtilFunctions | 2 | PASSED |
| TestFRConversion | 3 | PASSED |
| TestPolyEvalFunctions | 5 | PASSED |
| TestHxr | 1 | PASSED |

### test_setup.py (19 tests) - ALL PASSED

EC(타원곡선) sigma 생성 함수 7개에 대한 테스트. `full_pipeline_data` session-scoped fixture 활용.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestSigma11 | 4 | PASSED |
| TestSigma12 | 2 | PASSED |
| TestSigma13 | 3 | PASSED |
| TestSigma14 | 3 | PASSED |
| TestSigma15 | 1 | PASSED |
| TestSigma21 | 4 | PASSED |
| TestSigma22 | 2 | PASSED |

### test_proving.py (8 tests) - ALL PASSED

EC proof 생성 함수 4개에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestProofA | 2 | PASSED |
| TestProofB | 2 | PASSED |
| TestProofC | 2 | PASSED |
| TestBuildRpubEnum | 2 | PASSED |

### test_verifying.py (6 tests) - ALL PASSED

EC 페어링 검증 함수 3개 + 변조 proof 거부 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestLhs | 2 | PASSED |
| TestRhs | 1 | PASSED |
| TestVerify | 3 | PASSED |

> **검증 항목**: 올바른 proof 승인, 변조된 proof_A 거부, 변조된 proof_C 거부.

### test_integration.py (6 tests) - ALL PASSED

Groth16 전체 파이프라인 E2E 테스트.

| 테스트 클래스 | 테스트 수 | 결과 |
|--------------|----------|------|
| TestR1csSatisfaction | 1 | PASSED |
| TestQapCancellation | 1 | PASSED |
| TestPolynomialIdentity | 2 | PASSED |
| TestE2EVerification | 2 | PASSED |

> **검증 파이프라인**: Code -> R1CS -> QAP -> Setup -> Proving -> Verifying

## 테스트 데이터

테스트에 사용된 기준 데이터:

```python
# 테스트 코드
def qeval(x):
    y = x**3
    return y + x + 5

# 입력
input_vars = [3]

# 기대 결과 벡터
r = [1, 3, 35, 9, 27, 30]

# Toxic waste (setup 단계)
alpha=3926, beta=3604, gamma=2971, delta=1357, x_val=3721

# Prover random
r=4106, s=4565

# Public indices
pub_r_indexs = [0, 1]
```

## 경고 사항

91건의 `DeprecationWarning`이 발생하며, 모두 `code_to_r1cs.py`에서 사용하는 `ast.Num`과 `ast.Num.n` 관련이다. Python 3.14에서 제거 예정이므로 `ast.Constant`로 마이그레이션이 필요하다.

## 디렉토리 구조

```
tests/
├── __init__.py                 # 패키지 마커
├── conftest.py                 # 공유 fixture (session-scoped)
├── test_determinant.py         # 33 tests - 행렬 연산
├── test_code_to_r1cs.py        # 25 tests - AST 파싱 + R1CS
├── test_qap_creator.py         # 20 tests - 다항식 + QAP
├── test_qap_creator_lcm.py     # 13 tests - LCM 변형 QAP
├── test_poly_utils.py          # 18 tests - FR 필드 다항식
├── test_setup.py               # 19 tests - EC sigma 생성
├── test_proving.py             #  8 tests - EC proof 생성
├── test_verifying.py           #  6 tests - EC 페어링 검증
└── test_integration.py         #  6 tests - E2E 파이프라인
```

---

# PLONK 단위 테스트 결과

## 실행 환경

| 항목 | 값 |
|------|-----|
| Python | 3.13.3 |
| pytest | 9.0.2 |
| OS | macOS (Darwin 25.2.0) |
| 실행일 | 2026-02-08 |

## 실행 방법

```bash
cd /Users/kevin/dev/interactive-zkp-study
source .venv/bin/activate
pytest tests/plonk/ -v
```

## 전체 결과 요약

```
321 passed in 269.99s (0:04:29)
```

| 지표 | 값 |
|------|-----|
| 총 테스트 수 | 321 |
| 통과 | 321 |
| 실패 | 0 |
| 소요 시간 | ~270초 (4분 29초) |

## 모듈별 테스트 결과

### test_foundation.py (118 tests) - ALL PASSED

FR 필드 산술, 타원곡선 연산, 다항식, FFT, 유틸리티 함수에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 | 커버리지 |
|--------------|----------|------|----------|
| TestFR | 15 | PASSED | FR 산술 (add/sub/mul/div/pow/modular), 페르마 소정리 |
| TestEC | 13 | PASSED | ec_mul, ec_add, ec_neg, ec_pairing (bilinearity), Z1 |
| TestRootsOfUnity | 10 | PASSED | get_root_of_unity, get_roots_of_unity, primitive root, 에러 케이스 |
| TestPolynomialBasic | 14 | PASSED | 생성, trim, degree, is_zero, len, repr, zero/one |
| TestPolynomialArithmetic | 16 | PASSED | add/sub/mul/neg/eq/scale, scalar 연산, radd/rsub/rmul |
| TestPolynomialEvaluate | 6 | PASSED | 상수/선형/이차, zero poly, int input |
| TestDivideByVanishing | 2 | PASSED | 정확한 나눗셈, ValueError |
| TestVanishing | 3 | PASSED | roots에서 0, degree, leading coeff |
| TestFromEvaluations | 2 | PASSED | roundtrip, 상수 다항식 |
| TestFFT | 7 | PASSED | fft/ifft 단일/기본/전체점/roundtrip |
| TestPolyDiv | 5 | PASSED | 정확한 나눗셈, 나머지, degree 비교, zero 나눗셈, 항등식 |
| TestLagrangeBasis | 2 | PASSED | Kronecker delta, roots of unity 도메인 |
| TestVanishingPolyEval | 3 | PASSED | root/off-root/polynomial 일치 |
| TestLagrangeBasisEval | 3 | PASSED | Kronecker delta, off-domain, sum-to-one |
| TestPublicInputPolynomial | 3 | PASSED | empty, single, multiple |
| TestPublicInputPolyEval | 2 | PASSED | polynomial 일치, empty |
| TestCosetFFT | 4 | PASSED | coset 평가, roundtrip, default k |
| TestPadding | 7 | PASSED | next_power_of_2, pad_to_power_of_2 |

### test_circuit.py (68 tests) - ALL PASSED

회로 게이트, 순열 인수, Fiat-Shamir 트랜스크립트에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 | 커버리지 |
|--------------|----------|------|----------|
| TestGate | 12 | PASSED | 생성(FR/int), check 유효/무효 (mul/add/const/public_input/zero/general) |
| TestCircuit | 16 | PASSED | 빈 회로, 4종 게이트 추가, copy constraint, selector polynomials, build_copy_constraints |
| TestX3PlusXPlus5Eq35 | 13 | PASSED | 반환값, 4개 게이트, witness 값, 게이트 만족, public inputs, copy constraint 검증 |
| TestBuildPermutationPolynomials | 5 | PASSED | 항등 순열, 스왑, x3 회로, 출력 길이, 코셋 값 검증 |
| TestComputeAccumulator | 6 | PASSED | z[0]=1, 길이, 항등순열 곱=1, 전체 곱=1, 다른 챌린지, 잘못된 witness 검출 |
| TestTranscript | 17 | PASSED | 결정론성, 다른 입력/레이블/순서, 체이닝, FR 타입, append_scalar/point/None, PLONK 라운드 시뮬레이션 |

> **참고**: `x³ + x + 5 = 35` 회로(x=3)는 4개 게이트(mul, mul, add, constant+add)로 구성되며, copy constraint가 x, x², x³, sum 공유를 강제한다.

### test_crypto.py (52 tests) - ALL PASSED

SRS 생성, KZG 다항식 커밋먼트, 회로 전처리에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 | 커버리지 |
|--------------|----------|------|----------|
| TestSRS | 12 | PASSED | deterministic 생성, g1/g2 powers 길이, max_degree, seed 재현성, 생성자 검증 |
| TestKZGCommit | 6 | PASSED | 상수/선형 다항식 커밋, 영다항식, 차수 초과 에러, max_degree 경계, 스칼라곱 호환 |
| TestKZGLinearity | 3 | PASSED | commit(a+b) == commit(a)+commit(b), 다른 차수, 영다항식 덧셈 |
| TestKZGOpening | 10 | PASSED | 상수/선형/이차/고차 valid opening, x=0, 잘못된 eval/point/commitment invalid 검증 |
| TestPreprocessor | 18 | PASSED | PreprocessedData, 도메인 2^n, ω^n=1, 셀렉터/순열 다항식+커밋먼트, σ 유효 순열, 멱등성 |
| TestCryptoIntegration | 3 | PASSED | 전처리된 셀렉터/순열 다항식 opening 증명, evaluations에서 복원한 다항식 커밋+열기 |

### test_prover.py (52 tests) - ALL PASSED

Prover 5라운드 및 전체 prove() 함수에 대한 테스트.

| 테스트 클래스 | 테스트 수 | 결과 | 커버리지 |
|--------------|----------|------|----------|
| TestProofAndProverState | 4 | PASSED | Proof/ProverState 초기화, 필드 기본값, build_proof() |
| TestRound1 | 8 | PASSED | witness 보간, 블라인딩(차수 증가, 도메인 값 보존), KZG 커밋먼트 일치, 랜덤 블라인딩 |
| TestRound2 | 9 | PASSED | β/γ 챌린지, z_poly, z(ω⁰)=1, z_comm, 블라인딩 |
| TestRound3 | 7 | PASSED | α 챌린지, t_lo/t_mid/t_hi, Z_H 나눗셈 검증, 커밋먼트 일치, 차수 바운드 |
| TestRound4 | 10 | PASSED | ζ 챌린지, 6개 평가값이 polynomial.evaluate(ζ)와 일치 |
| TestRound5 | 5 | PASSED | v 챌린지, r_eval 일관성(t_eval·Z_H(ζ)), 선형화 다항식 직접 재구성 검증 |
| TestProveFunction | 10 | PASSED | 전체 prove() 반환값, 모든 커밋먼트 on-curve, 평가값 FR 타입, 블라인딩 비결정성 |
| TestInvalidWitness | 1 | PASSED | 잘못된 witness로 Round 3에서 ValueError 발생 확인 |

> **핵심 검증**: Round 4에서 a(ζ), b(ζ), c(ζ), S_σ1(ζ), S_σ2(ζ), z(ζ·ω) 6개 평가값이 실제 다항식 evaluate()와 정확히 일치하는지 확인. Round 5에서 `r_eval = t_eval · Z_H(ζ)` 관계 검증.

### test_e2e.py (31 tests) - ALL PASSED

Verifier 및 전체 파이프라인 E2E 통합 테스트.

| 테스트 클래스 | 테스트 수 | 결과 | 커버리지 |
|--------------|----------|------|----------|
| TestE2EPipeline | 6 | PASSED | x³+x+5=35, 덧셈 회로, 곱셈 회로, proof 필드 확인, 결정론성, 다른 SRS seed |
| TestSoundnessScalarTampering | 7 | PASSED | a/b/c_eval, s_sigma1/2_eval, z_omega_eval, r_eval 각각 변조 → 검증 실패 |
| TestSoundnessCommitmentTampering | 9 | PASSED | a/b/c/z/t_lo/t_mid/t_hi/W_zeta/W_zeta_omega_comm 각각 변조 → 검증 실패 |
| TestPublicInputHandling | 2 | PASSED | 잘못된/빈 PI로도 검증 통과 (q_C 인코딩 동작 문서화) |
| TestCrossCircuitSoundness | 2 | PASSED | 다른 회로의 전처리 데이터로 교차 검증 실패 확인 |
| TestGateConstraints | 3 | PASSED | 각 회로의 게이트 제약 만족 확인 |
| TestMultipleTampering | 2 | PASSED | 여러 필드 동시 변조 실패 확인 |

> **Soundness 검증**: 16개 proof 요소(7개 스칼라 + 9개 커밋먼트)를 각각 변조하여 모두 검증 실패를 확인. 교차 회로 soundness도 검증.
>
> **Public Input 참고**: 현재 구현은 공개 입력을 q_C 셀렉터에 직접 인코딩하므로 PI(x)=0이다. 따라서 verify()의 public_inputs 파라미터가 검증에 영향을 주지 않는다.

## 테스트 데이터

테스트에 사용된 기준 회로:

```python
# x³ + x + 5 = 35 (x=3)
# Gate 0: x * x = x²       (multiplication)
# Gate 1: x * x² = x³      (multiplication)
# Gate 2: x³ + x = sum     (addition)
# Gate 3: sum + 5 = 35     (constant addition)

# Witness
a_vals = [3, 3, 27, 30]    # left inputs
b_vals = [3, 9, 3, 5]      # right inputs
c_vals = [9, 27, 30, 35]   # outputs
public_inputs = [35]

# SRS seed (테스트용)
seed = 42
max_degree = 30
```

## 디렉토리 구조

```
tests/plonk/
├── __init__.py                 # 패키지 마커
├── test_foundation.py          # 118 tests - FR, Polynomial, FFT, Utils
├── test_circuit.py             #  68 tests - Gate, Circuit, Permutation, Transcript
├── test_crypto.py              #  52 tests - SRS, KZG, Preprocessor
├── test_prover.py              #  52 tests - Prover Round 1~5
└── test_e2e.py                 #  31 tests - Verifier, E2E, Soundness
```
