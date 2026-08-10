# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021-2026 Antoine COLLET

from pyesmda._inversion import ESMDAInversionType


def test_inversion_type() -> None:
    for t in ESMDAInversionType.to_list():
        assert t.value == ESMDAInversionType(t.value)

    assert not "test" == ESMDAInversionType.NAIVE
    assert "test" != ESMDAInversionType.NAIVE
    assert not 2.0 == ESMDAInversionType.NAIVE
