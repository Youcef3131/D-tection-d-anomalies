# TP - Detection danomalies

## Objectif

Comparer plusieurs methodes de detection danomalies, construire une regle de seuillage sans utiliser les etiquettes, puis evaluer les resultats avec des mesures adaptees.

Le notebook principal est [code.ipynb](./code.ipynb). Les fonctions reutilisables sont regroupees dans [utils.py](./utils.py).

## 1. Donnees Mouse

Le fichier mouse.txt contient 500 observations decrites par deux variables numeriques.

python
X, y_reference = load_mouse_data("mouse.txt", n_reference_outliers=10)


Resultats de la verification :

text
X.shape = (500, 2)
Valeurs manquantes = 0
Observations normales = 490
Anomalies de reference = 10


Les dix dernieres observations sont les anomalies de reference indiquees par lenonce. Le vecteur y_reference contient donc :

- 0 pour les 490 premieres observations ;
- 1 pour les 10 dernieres observations.

Ces etiquettes ne sont pas utilisees pour entrainer Isolation Forest ou LOF. Elles servent uniquement a levaluation et a la comparaison avec les points detectes.

## 2. Isolation Forest et LOF

La contamination est fixee a :

python
contamination = 10 / 500 = 0.02


Cela indique aux modeles quenviron 2 % des observations peuvent etre des anomalies.

### Isolation Forest

Isolation Forest construit des arbres aleatoires. Une observation facilement isolee est consideree comme plus anormale.

Parametres utilises :

python
IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)


Les labels retournes sont :

- 1 : observation normale ;
- -1 : anomalie.

Le score affiche dans le notebook est :

python
scores = -model.decision_function(X)


Le signe moins permet dutiliser la convention suivante :

- score faible : observation plutot normale ;
- score eleve : observation plus anormale.

### LOF

LOF (*Local Outlier Factor*) compare la densite locale dun point avec celle de ses voisins.

Deux scores sont affiches :

1. le score calcule sur les observations dapprentissage avec novelty=False ;
2. le score prevu pour de nouvelles observations avec novelty=True.

Ces deux scores ne sont pas exactement definis de la meme facon et sont donc presentes separement.

### Resultats avec le seuil par defaut

| Methode | Nombre doutliers detectes |
|---|---:|
| Isolation Forest | 10 |
| LOF - apprentissage | 10 |
| LOF - nouveaute | 10 |

Le nombre detecte est coherent avec la contamination fixee a 2 %. Cela ne garantit toutefois pas que les points detectes soient exactement les anomalies de reference.

## 3. Choix dun seuil sans etiquettes

### Regle IQR

Pour chaque methode, le seuil est calcule uniquement a partir des scores :

text
IQR = Q3 - Q1
seuil = Q3 + 1.5 * IQR


Une observation est declaree anomalie lorsque :

python
score >= seuil


Le multiplicateur 1.5 est fixe avant levaluation et ne depend pas de y_reference. Cette regle est une heuristique et ne garantit pas le meilleur F1.

### Comparaison des seuils

La matrice de confusion est presentee selon la convention :

text
[[vrais negatifs, faux positifs],
 [faux negatifs, vrais positifs]]


La classe positive est la classe anomalie.

#### Isolation Forest

| Seuil | Valeur | Detectes | F1 anomalie |
|---|---:|---:|---:|
| IQR | -0.027165 | 21 | 0.6452 |
| Defaut | interne au modele | 10 | 0.9000 |

Matrice IQR :

text
[[479, 11],
 [  0, 10]]


Le seuil IQR detecte les 10 anomalies, mais signale aussi 11 observations normales. Il ny a aucune anomalie manquee, mais le nombre de faux positifs diminue le F1.

Matrice avec le seuil par defaut :

text
[[489, 1],
 [  1, 9]]


Le seuil par defaut obtient un meilleur F1, car il ne produit quun faux positif et manque une seule anomalie.

#### LOF

| Seuil | Valeur | Detectes | F1 anomalie |
|---|---:|---:|---:|
| IQR | 1.225625 | 37 | 0.4255 |
| Defaut | interne au modele | 10 | 1.0000 |

Matrice IQR :

text
[[463, 27],
 [  0, 10]]


Le seuil IQR detecte les 10 anomalies, mais signale a tort 27 observations normales. Cela explique le F1 faible.

Matrice avec le seuil par defaut :

text
[[490, 0],
 [  0, 10]]


Dans cette execution, LOF detecte correctement les 10 anomalies de reference sans faux positif. Le F1 vaut donc 1.0000.

### Observation generale

Le seuil IQR est plus permissif : il detecte davantage de points, mais genere davantage de faux positifs. Le seuil par defaut est meilleur sur cet echantillon, notamment parce que la contamination a ete fixee a 10 / 500, ce qui correspond au nombre danomalies de reference.

Il ne faut pas conclure que le seuil par defaut est toujours superieur. Le seuil IQR a lavantage de ne pas utiliser les etiquettes et de ne pas imposer directement un nombre danomalies.

## 4. AP et AUC-ROC

Resultats obtenus :

| Methode | AP | AUC-ROC |
|---|---:|---:|
| Isolation Forest | 0.9614 | 0.9990 |
| LOF | 1.0000 | 1.0000 |

LAP (*Average Precision*) resume la courbe precision-rappel. LAUC-ROC mesure la capacite du score a classer les anomalies avant les observations normales.

Modifier le seuil change les labels predits, donc la matrice de confusion et le F1. En revanche, lAP et lAUC-ROC utilisent directement les scores continus et evaluent le classement sur une gamme de seuils. Modifier uniquement le seuil final ne change donc ni les scores ni leur ordre : lAP et lAUC-ROC restent inchangees.

Ces resultats decrivent la detection sur les 500 observations Mouse observees. Ils ne constituent pas une estimation des performances sur des donnees futures.

## 5. Synthese

| Methode | Seuil | F1 anomalie | AP | AUC-ROC |
|---|---|---:|---:|---:|
| Isolation Forest | IQR | 0.6452 | 0.9614 | 0.9990 |
| Isolation Forest | Defaut | 0.9000 | 0.9614 | 0.9990 |
| LOF | IQR | 0.4255 | 1.0000 | 1.0000 |
| LOF | Defaut | 1.0000 | 1.0000 | 1.0000 |

Sur cet echantillon, LOF obtient le meilleur classement avec lAP et lAUC-ROC. Avec son seuil par defaut, il obtient egalement le meilleur F1.

## 6. Reproductibilite

Les graines sont fixees avec :

python
SEED = 42
np.random.seed(SEED)
random_state = 42


Versions utilisees :

text
Python : 3.13.5
NumPy : 2.3.4
pandas : 2.3.3
scikit-learn : 1.7.2


Pour reproduire les resultats, executer les cellules de [code.ipynb](./code.ipynb) dans lordre depuis le dossier contenant mouse.txt, utils.py et code.ipynb.


## 7. Detection de nouveautes avec LOF

### 7.1 Separation apprentissage/test

Pour simuler un systeme de detection de nouveautes, les 490 observations normales sont separees aleatoirement avec `random_state=42` :

- 392 observations normales pour l apprentissage ;
- 98 observations normales reservees au test ;
- les 10 anomalies de reference sont toutes reservees au test.

Le modele LOF est ajuste uniquement sur les 392 observations normales. Le test contient donc 108 points.

```text
Apprentissage : 392 normaux
Test : 98 normaux + 10 anomalies
```

La selection des normaux utilise `y_reference` pour construire ce protocole. Les labels ne sont ensuite pas utilises pour ajuster LOF ni pour calculer le seuil. Ils servent uniquement a evaluer les predictions sur le test.

### 7.2 Resultats sur les nouveautes

Le seuil IQR est calcule uniquement avec les scores des 392 observations normales d apprentissage.

| Regle | Nombre detecte | Matrice de confusion | F1 anomalie |
|---|---:|---|---:|
| IQR appris sur normaux | 13 | `[[95, 3], [0, 10]]` | 0.8696 |
| LOF seuil par defaut | 11 | `[[97, 1], [0, 10]]` | 0.9524 |

Le seuil IQR detecte les 10 anomalies, mais signale 3 normaux comme anomalies. Le seuil par defaut detecte aussi les 10 anomalies et ne produit qu un seul faux positif.

### 7.3 Faut-il ajouter des lignes dans Mouse ?

Non. Il ne faut pas ajouter de nouvelles lignes dans `mouse.txt`.

La detection de nouveautes ne signifie pas que l on doit modifier la base de donnees. Elle signifie que l on divise les observations existantes en deux ensembles :

```text
donnees normales connues -> apprentissage du modele
nouvelles observations -> test et prediction
```

Dans ce TP, les observations reservees au test jouent le role de nouvelles observations. Elles sont deja presentes dans `mouse.txt`, mais elles ne sont pas montrees au modele pendant son apprentissage. Cette separation est faite uniquement pour simuler le fonctionnement futur du systeme.

Dans une application reelle, les nouvelles lignes arriveraient plus tard, apres l entrainement du modele. Le modele conserverait son apprentissage normal et appliquerait simplement :

```python
predictions = model.predict(X_new)
scores = -model.score_samples(X_new)
```

Les donnees `X_new` ne seraient pas utilisees pour reajuster le modele avant la prediction.

### 7.4 Difference entre outliers et nouveautes

La detection d outliers analyse un ensemble qui contient potentiellement des normaux et des anomalies. Le modele utilise la structure globale et locale de cet ensemble pour trouver les observations atypiques.

La detection de nouveautes suppose que l apprentissage contient uniquement des observations normales. Le modele apprend une description du comportement normal, puis score des observations nouvelles sans les integrer a l apprentissage.

La construction de l apprentissage normal avec les labels apporte une information supplementaire : nous savons quels points sont normaux. Le protocole est donc semi-supervise dans sa preparation, meme si LOF reste non supervise pendant son ajustement. Cette information n etait pas disponible dans la detection d outliers de la partie precedente.

### 7.5 Limites

Il n y a que 10 anomalies. Une seule erreur modifie fortement le F1, car le nombre de vrais positifs est faible. Les performances peuvent aussi varier selon la separation aleatoire des observations normales et selon le choix du nombre de voisins. Ces resultats decrivent cet echantillon Mouse et ne garantissent pas les performances sur de futures donnees.
