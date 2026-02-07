"""
PLONK Structured Reference String (SRS)
=========================================

범용(universal) 신뢰 설정(trusted setup)을 생성한다.

**SRS란?**
  KZG 다항식 커밋먼트 스킴에 필요한 공개 파라미터이다.
  비밀 값 τ ("toxic waste")를 사용하여 생성되며,
  생성 후 τ는 반드시 폐기되어야 한다.

  SRS = {
      G1 powers: [G1, τ·G1, τ²·G1, ..., τ^d·G1]
      G2 powers: [G2, τ·G2]
  }

**범용(Universal) 설정**:
  Groth16과 달리 PLONK의 SRS는 회로에 독립적이다.
  한 번 생성하면 최대 차수 d 이하의 모든 회로에 재사용 가능하다.

**보안**:
  τ를 아는 사람은 임의의 거짓 증명을 만들 수 있다.
  실제 시스템에서는 MPC(Multi-Party Computation)로 τ를 생성하여
  모든 참여자 중 한 명이라도 정직하면 안전성이 보장된다.
  여기서는 교육용으로 seed에서 결정론적으로 생성한다.

사용 예시:
    >>> srs = SRS.generate(max_degree=16, seed=42)
    >>> len(srs.g1_powers)  # 17 (0차부터 16차까지)
"""

import hashlib
from zkp.plonk.field import FR, G1, G2, ec_mul, CURVE_ORDER


class SRS:
    """Structured Reference String: KZG 커밋먼트용 공개 파라미터.

    속성:
        g1_powers: [G1, τ·G1, τ²·G1, ..., τ^d·G1]
        g2_powers: [G2, τ·G2]
        max_degree: 지원하는 최대 다항식 차수 d
    """

    def __init__(self, g1_powers, g2_powers, max_degree):
        self.g1_powers = g1_powers
        self.g2_powers = g2_powers
        self.max_degree = max_degree

    @classmethod
    def generate(cls, max_degree, seed=None):
        """SRS를 생성한다.

        Args:
            max_degree: 지원할 최대 다항식 차수.
                        PLONK에서 필요한 최대 차수는 약 3n+5 (n: 게이트 수).
            seed: 결정론적 생성을 위한 시드 (교육용).
                  실제 시스템에서는 MPC를 사용해야 한다.

        Returns:
            SRS: 생성된 구조화 참조 문자열

        예시 (x³+x+5=35 회로, n=4 게이트):
            >>> srs = SRS.generate(max_degree=20, seed=1234)
            >>> # 20차까지의 다항식을 커밋할 수 있음
        """
        # toxic waste τ 생성
        if seed is not None:
            h = hashlib.sha256(str(seed).encode()).digest()
            tau_int = int.from_bytes(h, "big") % CURVE_ORDER
        else:
            # 랜덤 시드 사용 (실제로는 MPC)
            import secrets
            tau_int = secrets.randbelow(CURVE_ORDER - 1) + 1
        tau = FR(tau_int)

        # G1 powers: [G1, τ·G1, τ²·G1, ..., τ^d·G1]
        g1_powers = []
        tau_power = FR(1)  # τ^0 = 1
        for _ in range(max_degree + 1):
            g1_powers.append(ec_mul(G1, tau_power))
            tau_power = tau_power * tau

        # G2 powers: [G2, τ·G2]
        g2_powers = [G2, ec_mul(G2, tau)]

        return cls(g1_powers, g2_powers, max_degree)
