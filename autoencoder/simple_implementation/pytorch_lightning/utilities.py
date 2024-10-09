import pandas as pd
import matplotlib.pyplot as plt

logspath = './logs/autoencoder_exp/'
version = 4

df = pd.read_csv(logspath + 'version_%i/metrics.csv' %version)
nan_index = df.loc[pd.notna(df['train_loss']), :].index

train_loss = df.loc[nan_index,'train_loss']
epoch = df.loc[nan_index,'epoch']

print(epoch, train_loss)
plt.figure(1)
plt.plot(epoch, train_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()