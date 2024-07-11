# Tirocinio

Uso di tecniche di machine learning per attribuzione di valori mancanti e generazione di dati sintetici in ambito veterinario.

## Prerequisiti
Assicurati di avere installato Anaconda sul tuo sistema. Anaconda è una distribuzione di Python che include una serie di pacchetti scientifici e strumenti di gestione degli ambienti virtuali.

## Configurazione dell'ambiente
1. **Apri il terminale**.
   
2. **Aggiorna Conda all'ultima versione** (opzionale ma consigliato):

    ```bash
    conda update conda
    ```

3. **Crea un nuovo ambiente**. Sostituisci `myenv` con il nome desiderato per il tuo ambiente. Puoi anche specificare la versione di Python che desideri utilizzare (ad esempio, `python=3.8`).

    ```bash
    conda create --name myenv python=3.8
    ```

4. **Attiva il nuovo ambiente**:

    ```bash
    conda activate myenv
    ```

5. **Installa i pacchetti necessari**. Ad esempio, se hai bisogno di `numpy`, `pandas` e `scikit-learn`, esegui:

    ```bash
    conda install numpy pandas scikit-learn
    ```

    Puoi anche installare pacchetti specifici da conda-forge:

    ```bash
    conda install -c conda-forge nome_pacchetto
    ```
