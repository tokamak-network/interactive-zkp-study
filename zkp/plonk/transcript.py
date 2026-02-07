"""
PLONK Fiat-Shamir Transcript
==============================

비대화식(non-interactive) 변환을 위한 Fiat-Shamir 해싱 구현.

**Fiat-Shamir 변환이란?**
  원래 PLONK는 대화식(interactive) 프로토콜이다:
  - Prover가 커밋먼트를 보내면
  - Verifier가 랜덤 챌린지를 보내고
  - Prover가 응답한다

  Fiat-Shamir 변환은 이 대화를 해시 함수로 시뮬레이션한다:
  - Prover가 지금까지의 모든 메시지를 해시하여 챌린지를 직접 생성
  - Verifier도 같은 방식으로 챌린지를 재구성하여 일치 여부 확인
  - 해시의 랜덤 오라클 모델 하에서 보안성이 보장됨

**PLONK의 5개 챌린지**:
  Round 1 → β, γ (순열 제약용)
  Round 2 → α (제약 결합용)
  Round 3 → ζ (평가 점)
  Round 4 → v (일괄 열기)
  Round 5 → u (결합 페어링)

사용 예시:
    >>> t = Transcript()
    >>> t.append_point(b"a_comm", commitment)
    >>> beta = t.challenge_scalar(b"beta")
"""

import hashlib
from py_ecc.fields import bn128_FQ as FQ
from zkp.plonk.field import FR, CURVE_ORDER


class Transcript:
    """SHA-256 기반 Fiat-Shamir 트랜스크립트.

    해시 상태를 누적하여 결정론적이면서 예측 불가능한 챌린지를 생성한다.
    Prover와 Verifier가 동일한 순서로 데이터를 추가하면
    동일한 챌린지가 생성된다.

    속성:
        state: 현재까지 누적된 해시 입력 바이트열

    보안 주의:
        - 모든 데이터는 레이블(label)과 함께 추가하여 도메인 분리(domain separation) 보장
        - 트랜스크립트 순서가 다르면 다른 챌린지가 생성됨
    """

    def __init__(self, label=b"plonk"):
        """트랜스크립트를 초기화한다.

        Args:
            label: 프로토콜 도메인 분리용 레이블 (기본값: b"plonk")
        """
        self.state = bytearray()
        self.state.extend(label)

    def append_scalar(self, label, scalar):
        """FR 스칼라 값을 트랜스크립트에 추가한다.

        Args:
            label: 바이트열 레이블 (예: b"a_eval")
            scalar: FR 원소

        예시:
            >>> t.append_scalar(b"a_eval", FR(42))
        """
        self.state.extend(label)
        # FR 원소를 32바이트 빅엔디안으로 직렬화
        val = int(scalar) % CURVE_ORDER
        self.state.extend(val.to_bytes(32, "big"))

    def append_point(self, label, point):
        """타원곡선 점(G1)을 트랜스크립트에 추가한다.

        G1 점은 (x, y) 좌표의 FQ 원소 쌍이다.
        무한원점(None)은 특별히 처리한다.

        Args:
            label: 바이트열 레이블 (예: b"a_comm")
            point: G1 점 (FQ 튜플) 또는 None (무한원점)

        예시:
            >>> t.append_point(b"a_comm", ec_mul(G1, 5))
        """
        self.state.extend(label)
        if point is None:
            # 무한원점: 64바이트의 0
            self.state.extend(b"\x00" * 64)
        else:
            x, y = point
            self.state.extend(int(x).to_bytes(32, "big"))
            self.state.extend(int(y).to_bytes(32, "big"))

    def challenge_scalar(self, label):
        """트랜스크립트로부터 챌린지 스칼라를 생성한다.

        현재 상태를 SHA-256으로 해싱하여 FR 원소를 도출한다.
        생성된 챌린지는 자동으로 트랜스크립트에 추가된다 (체이닝).

        Args:
            label: 바이트열 레이블 (예: b"beta")

        Returns:
            FR: 챌린지 스칼라

        예시:
            >>> beta = t.challenge_scalar(b"beta")
            >>> gamma = t.challenge_scalar(b"gamma")
            # beta와 gamma는 서로 다른 값 (상태가 업데이트되므로)
        """
        self.state.extend(label)
        # SHA-256 해시 → 256비트 값 → FR 필드 원소로 축소
        h = hashlib.sha256(bytes(self.state)).digest()
        challenge_int = int.from_bytes(h, "big") % CURVE_ORDER
        challenge = FR(challenge_int)

        # 챌린지를 상태에 추가 (체이닝: 다음 챌린지에 영향)
        self.state.extend(h)

        return challenge
