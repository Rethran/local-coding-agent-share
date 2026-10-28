class Tuote:
    def __init__(self, nimi: str, hinta: float, varastosaldo: int):
        self.nimi = nimi
        self.hinta = hinta
        self.varastosaldo = varastosaldo
    
    def lisaa_varastoon(self, maara: int):
        """Lisää tuotetta varastoon"""
        if maara < 0:
            raise ValueError("Määrän täytyy olla positiivinen")
        self.varastosaldo += maara
    
    def vahenna_varastosta(self, maara: int):
        """Vähentää tuotteen varastosaldoa"""
        if maara < 0:
            raise ValueError("Määrän täytyy olla positiivinen")
        if self.varastosaldo - maara < 0:
            raise ValueError("Varastosaldo ei voi olla negatiivinen")
        self.varastosaldo -= maara

    def hae_varastosaldo(self):
        """Palauttaa tuotteen varastosaldon"""
        return self.varastosaldo

class Ostoskori:
    def __init__(self):
        self.tuotteet = []

    def lisaa_tuote(self, tuote: Tuote):
        """Lisää tuotteen ostoskoriin ja vähentää varastosaldoa"""
        if tuote.varastosaldo <= 0:
            raise ValueError("Tuotetta ei ole varastossa")
        tuote.vahenna_varastosta(1)
        self.tuotteet.append(tuote)

import unittest

class TestTuote(unittest.TestCase):
    def test_tuotteen_lisääminen_vähentää_varastosaldoa(self):
        """Testataan että tuotteen lisääminen vähentää varastosaldoa"""
        # Luodaan tuote
        tuote = Tuote("Maito", 3.50, 10)

        # Luodaan ostoskori ja lisätään tuote siihen
        kori = Ostoskori()
        kori.lisaa_tuote(tuote)

        # Tarkistetaan että saldo on vähentynyt
        self.assertEqual(tuote.hae_varastosaldo(), 9)

    def test_tyhjän_tuotteen_lisääminen_nostaa_valueerrorin(self):
        """Testataan että tyhjän tuotteen lisääminen nostaa ValueErrorin"""
        # Luodaan tuote ilman varastosaldoa
        tuote = Tuote("Maito", 3.50, 0)

        # Luodaan ostoskori ja yritetään lisätä tuote siihen (tämä pitäisi nostaa virheen)
        kori = Ostoskori()
        with self.assertRaises(ValueError):
            kori.lisaa_tuote(tuote)

if __name__ == "__main__":
    unittest.main()