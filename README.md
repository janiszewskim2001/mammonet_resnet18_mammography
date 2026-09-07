# MammoNet vs ResNet-18

Kod do pracy *Analiza porównawcza autorskiej sieci konwolucyjnej MammoNet z modelem
uczenia transferowego ResNet-18 w klasyfikacji obrazów mammograficznych z wykorzystaniem
Explainable AI*, Michał Janiszewski

Zadanie: Rozróżnienie zwapnienia i guzów na wycinkach zmian ogniskowych ze zbioru CBIS-DDSM.
Porównanie obejmuje skuteczność klasyfikacji, koszt obliczeniowy oraz charakter map
atrybucji generowanych metodą Grad-CAM.

## Dane

Repozytorium nie zawiera obrazów. Zbiór CBIS-DDSM jest publicznie dostępny i należy pobrać
go samodzielnie; użyto eksportu JPEG rozpowszechnianego w serwisie Kaggle
(`awsaf49/cbis-ddsm-breast-cancer-image-dataset`). Po rozpakowaniu katalog powinien
zawierać podkatalogi `csv/` i `jpeg/`.

Ścieżki w plikach opisowych wskazują na pliki DICOM, podczas gdy obrazy JPEG leżą
w katalogach nazwanych identyfikatorami serii. Wycinek zmiany i jego maska często dzielą
ten sam katalog i rozróżnia je wyłącznie pole `SeriesDescription` w `dicom_info.csv`.
Skrypt `prepare_data.py` rozwiązuje to powiązanie.

## Instalacja

```bash
pip install -r requirements.txt
```

Kod korzysta z akceleratora MPS (Apple Silicon) lub CUDA, jeśli są dostępne, w przeciwnym
razie działa na CPU.

## Kolejność uruchamiania

```bash
# przygotowanie danych do struktury ImageFolder
python prepare_data.py --raw_dir /sciezka/do/CBIS-DDSM

# trening obu modeli
python train.py --arch mammonet
python train.py --arch resnet18

# metryki, testy istotności, koszt obliczeniowy
python evaluate.py

# ilościowa charakterystyka map atrybucji
python analyze_attribution.py

# skuteczność w podgrupach klinicznych
python analyze_clinical.py --raw_dir /sciezka/do/CBIS-DDSM

# ryciny w EPS i PNG
python make_figures.py
```

Domyślne katalogi: `data/` (obrazy), `data_masks/` (maski), `results/` (modele i wyniki),
`figures/` (ryciny). Każdy skrypt przyjmuje odpowiedni argument, jeśli układ ma być inny.

## Odtwarzalność

Ziarno generatora liczb losowych ustalone jest w `config.py` i stosowane we wszystkich
etapach. Uruchomienie powyższej sekwencji odtwarza wyniki przedstawione w pracy:

| | MammoNet | ResNet-18 |
|---|---|---|
| Dokładność | 0,832 | 0,936 |
| AUC | 0,919 | 0,977 |
| Parametry | 93 476 | 11 177 538 |

Test McNemara: χ² = 49,371; p = 2,1 · 10⁻¹². Różnica AUC: 0,058 (95% CI: 0,040–0,076).

Czas inferencji zależy od sprzętu i będzie się różnił między maszynami.

## Uwagi metodologiczne

`models.check_attention_gradient` sprawdza przed treningiem, czy gradient dociera do warstwy
uwagi przestrzennej. Warstwa utworzona wewnątrz `forward()` zamiast w konstruktorze nie trafia
do `model.parameters()` i nigdy nie jest trenowana, a sieć uczy się bez żadnego komunikatu
o błędzie.

Mapy Grad-CAM pobierane są z warstwy `pool3` w MammoNet (rozdzielczość 16×16) oraz `layer4`
w ResNet-18 (7×7). Różnica rozdzielczości przekłada się bezpośrednio na ostrość mapy i została
uwzględniona w interpretacji wyników.

Ocena zgodności lokalizacyjnej map przez porównanie z maskami segmentacji okazała się
niewykonalna: w wykorzystanym eksporcie maski i wycinki klasyfikacyjne nie są zapisane
w spójnym układzie współrzędnych. `analyze_attribution.py` wyznacza miary opisujące sam
rozkład mapy, niezależne od masek.

## Struktura

```
config.py                 ścieżki, ziarno, hiperparametry
prepare_data.py           porządkowanie surowego zbioru
data.py                   wczytywanie i transformacje
models.py                 MammoNet, ResNet-18, kontrola gradientu
train.py                  trening
gradcam.py                mapy atrybucji
evaluate.py               metryki, Wilson, McNemar, bootstrap, benchmark
analyze_attribution.py    energia brzegowa, stereotypowość, centroidy
analyze_clinical.py       podgrupy subtelności i gęstości utkania
make_figures.py           ryciny publikacyjne
```

## Licencja

MIT — patrz `LICENSE`. Licencja obejmuje wyłącznie kod. Warunki korzystania ze zbioru
CBIS-DDSM określają jego twórcy.
