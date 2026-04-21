# AIprojekt-SafeSteps

Naprava za pomoč slepim in slabovidnim pri premikanju v prostoru. 

## Ključne funkcionalnosti
- Zaznavanje razdalje
- Prepoznavanje ovir z umetno inteligenco - analiza slik
- Povratne informacije uporabniku

## Delovanje
1. Merjenje razdalje
   - ToF senzor neprestano meri razdaljo do najbližje ovire (do 2 metra). Ko je ovira znotraj vnaprej določenega območja, STM32 sproži opozorilo.
2. Prepoznavanje ovir
   - Kamera zajema slike okolice. Ovire bodo obdelane s pomočjo že naučene nevronske mreže.
3. Povratna informacija
   - *sestava informacije še ni določena, najverjetneje zvočno opozorilo, sestavljeno iz podatkov ToF senzorja in kamere*
  
## Tehnologije
- **Hardware**: STM32F411, ToF senzor, kamera
- **Software & AI**: Python, nevronska mreža (ni še določena, verjetno YOLO)

## Zagon 
```sh
# navodila za namestitev in zagon bodo dodana kasneje
```
## Avtorji
Lajla Suljić, Lara Podgoršek, Ema Horvat
