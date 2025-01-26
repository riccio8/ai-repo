import pandas as pd
import matplotlib.pyplot as plt

# Leggi il file CSV
file_csv = 'btc.csv'  
df = pd.read_csv(file_csv)

# Estrai i dati dalla prima colonna
dati = df.iloc[:, 0]  # Prima colonna

# Genera le coordinate x (indici dei dati)
x = range(len(dati))

# Crea il grafico a dispersione
plt.scatter(x, dati, color='blue', label='Dati')

# Aggiungi etichette e titolo
plt.xlabel('Indice')
plt.ylabel('btc')
plt.title('Grafico a dispersione dei dati')
plt.legend()

# Mostra il grafico
plt.show()
