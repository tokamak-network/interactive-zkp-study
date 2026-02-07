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
