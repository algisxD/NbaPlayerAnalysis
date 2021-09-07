import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

###################################################
# Konstantos
###################################################

#Iveskite norima klasteriu skaiciu
stulpelis1 = 'GP'
stulpelis2 = 'MPG'
stulpelis3 = 'ORPM'
stulpelis4 = 'DRPM'
stulpelis5 = 'RPM'
stulpelis6 = 'WINS'
klasteriuSkaicius = 2



###################################################
# Metodai
###################################################

def klasterizuoti(df, stulpelis1, stulpelis2, klasteriuSkaicius):
    plt.scatter(df[stulpelis1], df[stulpelis2])
    plt.xlabel(stulpelis1)
    plt.ylabel(stulpelis2)
    plt.title('Pries klasterizavima')
    plt.show()

    km = KMeans(n_clusters=klasteriuSkaicius)
    y_spejimas = km.fit_predict(df[[stulpelis1, stulpelis2]])
    df['klasteris'] = y_spejimas
    dfs = []
    for x in range(klasteriuSkaicius):
        dfTarp = df[df.klasteris==x]
        print("Klasteris ", x + 1, " turi ", dfTarp.shape[0])
        dfs.append(dfTarp)
    print()
    
    if klasteriuSkaicius == 2:
        plt.scatter(dfs[0][stulpelis1], dfs[0][stulpelis2], color="red")
        plt.scatter(dfs[1][stulpelis1], dfs[1][stulpelis2], color="green")
    elif klasteriuSkaicius == 3:
        plt.scatter(dfs[0][stulpelis1], dfs[0][stulpelis2], color="red")
        plt.scatter(dfs[1][stulpelis1], dfs[1][stulpelis2], color="green")
        plt.scatter(dfs[2][stulpelis1], dfs[2][stulpelis2], color="blue")
    elif klasteriuSkaicius == 4:
        plt.scatter(dfs[0][stulpelis1], dfs[0][stulpelis2], color="red")
        plt.scatter(dfs[1][stulpelis1], dfs[1][stulpelis2], color="green")
        plt.scatter(dfs[2][stulpelis1], dfs[2][stulpelis2], color="blue")
        plt.scatter(dfs[3][stulpelis1], dfs[3][stulpelis2], color="black")
    else:
        print("Netinkamas klasteriu skaicius")
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', label='centroid')
    plt.xlabel(stulpelis1)
    plt.ylabel(stulpelis2)
    plt.legend()
    plt.title('Po klasterizavima')
    plt.show()

def klasterizuotKiekvienaSuKiekvienu(df, stulpeliai, klasteriuSkaicius):
    for x in stulpeliai:
        for y in stulpeliai:
            klasterizuoti(df, x, y, klasteriuSkaicius)


###################################################
# Vykdymas
###################################################

df = pd.read_csv('data2.csv')

scaler = MinMaxScaler()

df[[stulpelis1]] = scaler.fit_transform(df[[stulpelis1]])
df[[stulpelis2]] = scaler.fit_transform(df[[stulpelis2]])
df[[stulpelis3]] = scaler.fit_transform(df[[stulpelis3]])
df[[stulpelis4]] = scaler.fit_transform(df[[stulpelis4]])
df[[stulpelis5]] = scaler.fit_transform(df[[stulpelis5]])
df[[stulpelis6]] = scaler.fit_transform(df[[stulpelis6]])

#Klasterizuoja du stulpelius
klasterizuoti(df, stulpelis4, stulpelis5, 3)

stulpeliai = []
stulpeliai.append(stulpelis1)
stulpeliai.append(stulpelis2)
stulpeliai.append(stulpelis3)
stulpeliai.append(stulpelis4)
stulpeliai.append(stulpelis5)
stulpeliai.append(stulpelis6)

#Klasterizuoja kiekviena stulpeli su kiekvienu
klasterizuotKiekvienaSuKiekvienu(df, stulpeliai, 3)

