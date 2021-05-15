# Snelle bewegingsdetectie uit camerabeelden
Dit is een bachelorproef waarbij algoritmes geïmplementeerd zijn die een robuustebewegingsdetectie op camerabeelden kunnen uitvoeren op basis van twee of drie frames.


Bachelorproef Science in de industriële wetenschappen: elektronica-ICT \
Auteurs: Arno Heirman, Dieter Van doorsselaere, Laurens Van Goethem \
Promotor: Peter Veelaert \
Begeleiders: Gianni Allebosch, Stefaan Lambrecht \
Academiejaar: 2018-2019

## Inhoud

- **Images**: map met datasets van changedetection.net
- **Video**: map voor videobeelden
- **main.py**: hoofdbestand om de methoden te evalueren met de testbench
- **test_all.py**: test alle methoden op alle datasets en slaat scores op in test_all.csv
- **testbench.py**: code om de methoden te evalueren
- **Traffic_count.py**: een toepassing van de bewegingsdetectie om auto's te tellen
- **difference_method.py**: bevat de verschilgebaseerd methoden
- **Edge_based_method.py**: bevat de randgebaseerd methoden
- **combined_method.py**: bevat methoden die verschil- en randgebaseerde methoden combineren
- **segmentatie.py**: code om een randenbeeld op te vullen

## Gebruik

Het project gebruikt python 3.7 \
FFmpeg is nodig voor de video in Traffic_count.py \
De packages (zie versie controle) kunnen geïnstalleerd worden met pip:

    pip install -r requirements.txt


**Main.py**

Hoofdbestand om de methoden te evalueren op de datasets van changedetection.net

argumenten:
- method: naam van de methode
    - diff_color: verschilgebaseerde methode voor twee frames
    - edge_based_method: randgebaseerde methode voor twee frames
    - combined_method: gecombineerde methode met twee frames
    - diff_color_3: verschilgebaseerde methode voor drie frames
    - edge_based_method_3: randgebaseerde methode voor drie frames
    - combined_method_3: gecombineerde methode met drie frames
- dataset: naam van de dataset in de map Images

gebruik:

    python main.py --method diff_color_3 --dataset pedestrians
    
**Traffic_count.py**
  
  Toepassing van bewegingsdetectie om auto's te tellen in een video.
  Om de video in te lezen moet **FFmpeg** geïnstalleerd zijn.
  De wagens die door de twee rechthoeken rijden worden geteld.
  De rechthoeken worden ingesteld in een globale variabele bovenaan in het bestand.
  De visualisatie wordt opgeslaan als traffic.avi.
  Het programma kan worden gestopt door de Q-toets in te drukken.
  
argumenten:
- filename: naam van het videobestand in de map Video

gebruik:

    python Traffic_count.py --filename GOPR0021.MP4
  
**test_all.py**

Script om alle methoden op alle datasets te testen (ingesteld bovenaan in het bestand). 
Scores worden opgeslaan in test_all.csv

gebruik:

    python test_all.py

## Versie controle

OS: Linux Mint 19.1 Cinnamon

IDE: JetBrains Pycharm 2019.1.1 (Professional Edition)

FFmpeg: version 3.4.6-0ubuntu0.18.04.1

Python 3.7.1

Packages
- numpy 1.16.3
- matplotlib 3.0.3
- opencv-python 4.1.0.25
- scikit-image 0.15.0
- scikit-video 1.1.11
- numba 0.43.1
