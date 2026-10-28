from laskin import erotus, summa, tulo


def test_summa():
    assert summa(2, 3) == 5


def test_erotus():
    assert erotus(5, 2) == 3


def test_tulo():
    assert tulo(2, 4) == 8
