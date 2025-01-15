# Tirocinio

Uso di tecniche di machine learning per attribuzione di valori mancanti e generazione di dati sintetici in ambito veterinario.

## Prerequisiti
Assicurati di avere installato [Anaconda](https://www.anaconda.com/products/individual) sul tuo sistema. Anaconda è una distribuzione di Python che include una serie di pacchetti scientifici e strumenti di gestione degli ambienti virtuali.

## Configurazione dell'ambiente

1. **Apri il terminale**

2. **Crea un nuovo ambiente**
   Specificare la versione di Python che desideri utilizzare (ad esempio, `python=3.8`).

    ```bash
    conda create --name ml_models python=3.8
    ```

3. **Attiva il nuovo ambiente**

    ```bash
    conda activate ml_models
    ```

4. **Installa i pacchetti necessari**
   Installa le dipendenze con [pip](https://pypi.org/project/pip/)

    ```bash
    pip install numpy
    pip install pandas
    pip install sklearn
    pip install logging
    pip install matplotlib
    pip install ipython
    pip install seaborn
    pip install tensorflow
    pip install keras
    ```