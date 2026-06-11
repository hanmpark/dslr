# DSLR - Maths expliquees simplement

Ce fichier explique les notions mathematiques du projet sans partir directement dans les grosses formules.

L'idee generale du projet est :

```text
notes des eleves -> modele -> maison Hogwarts predite
```

Les notes sont les **features**.
La maison est le **label** a predire.

---

# 1. Les mots importants

## Feature

Une feature est une information donnee au modele.

Dans DSLR, les features sont les notes :

```text
Astronomy
Herbology
Potions
Charms
Flying
...
```

Exemple :

```text
Eleve A :
Astronomy = 12
Potions = 8
Charms = 15
```

Ces valeurs sont donnees au modele pour l'aider a deviner la maison.

## Label / target

Le label est ce que le modele doit predire.

Dans DSLR :

```text
Hogwarts House
```

Exemple :

```text
features = notes de l'eleve
label = Gryffindor
```

Pendant l'entrainement, le modele voit les features ET la vraie maison.
Pendant la prediction, il voit seulement les features et doit deviner la maison.

## Modele

Un modele est une fonction qui apprend une relation entre les entrees et la sortie.

Dans ce projet :

```text
entrees : notes
sortie : maison
```

Le modele apprend par exemple que certaines combinaisons de notes ressemblent plus a Ravenclaw, Gryffindor, Hufflepuff ou Slytherin.

## Poids

Un poids indique l'importance d'une feature dans la decision.

Exemple imaginaire pour un modele Gryffindor :

```text
Potions a un poids fort
Flying a un poids faible
```

Cela veut dire que `Potions` influence plus la prediction que `Flying` pour ce modele.

## Biais

Le biais est une valeur de base ajoutee au calcul du modele.

Tu peux le voir comme un point de depart avant de regarder les features.

Dans le code, le biais est :

```python
w[0]
```

---

# 2. Statistiques descriptives dans `describe.py`

Le but de `describe.py` est de comprendre les donnees avant de faire du machine learning.

Il affiche :

```text
Count
Mean
Std
Min
25%
50%
75%
Max
```

## Count

`Count` est le nombre de valeurs disponibles dans une colonne.

Exemple :

```text
Potions : 10, 12, vide, 15
Count = 3
```

La valeur vide ne compte pas.

## Mean

`Mean` est la moyenne.

Recette :

```text
on additionne les valeurs
on divise par le nombre de valeurs
```

Exemple :

```text
10, 12, 14
somme = 36
nombre de valeurs = 3
moyenne = 36 / 3 = 12
```

La moyenne donne une valeur centrale.

## Std, ou ecart type

`Std` veut dire standard deviation, en francais **ecart type**.

Il mesure si les valeurs sont proches ou loin de la moyenne.

Exemple 1 :

```text
10, 10, 10
moyenne = 10
ecart type = 0
```

Toutes les valeurs sont exactement sur la moyenne.
Il n'y a aucune dispersion.

Exemple 2 :

```text
5, 10, 15
moyenne = 10
ecart type environ 4.08
```

Les valeurs sont autour de la moyenne, mais elles s'en eloignent.
L'ecart type resume cette dispersion.

Pourquoi on met les ecarts au carre ?

Parce que les ecarts positifs et negatifs s'annulent sinon.

```text
valeurs : 5, 10, 15
moyenne : 10
ecarts : -5, 0, +5
somme des ecarts = 0
```

Si on additionne directement, on croit a tort qu'il n'y a pas de dispersion.

Donc on utilise les carres :

```text
(-5)^2 = 25
0^2 = 0
5^2 = 25
```

Puis on fait une racine carree pour revenir dans la meme unite que les notes.

Ce qu'il faut retenir :

```text
ecart type petit -> valeurs regroupees
ecart type grand -> valeurs dispersees
```

## Min et Max

`Min` est la plus petite valeur.
`Max` est la plus grande valeur.

Exemple :

```text
5, 10, 15
Min = 5
Max = 15
```

## 25%, 50%, 75%

Ces valeurs sont les quartiles.

Elles servent a comprendre comment les donnees sont reparties.

On trie les valeurs :

```text
2, 4, 6, 8, 10, 12, 14, 16
```

Puis on regarde :

```text
25% -> environ le premier quart
50% -> le milieu, aussi appele mediane
75% -> environ les trois quarts
```

La mediane (`50%`) est utile parce qu'elle est moins sensible aux valeurs extremes que la moyenne.

Exemple :

```text
10, 11, 12, 13, 1000
```

La moyenne est tiree vers `1000`.
La mediane reste autour du centre reel des donnees.

---

# 3. Histogrammes dans `histogram.py`

Un histogramme montre la distribution d'une feature.

Il repond a la question :

```text
Combien de valeurs tombent dans chaque intervalle ?
```

Exemple :

```text
notes entre 0 et 5   -> 4 eleves
notes entre 5 et 10  -> 20 eleves
notes entre 10 et 15 -> 18 eleves
notes entre 15 et 20 -> 6 eleves
```

Dans le projet, `histogram.py` superpose les distributions des 4 maisons pour chaque cours.

La question du sujet est :

```text
Quel cours a une distribution homogene entre les 4 maisons ?
```

Une distribution homogene veut dire :

```text
les maisons se ressemblent sur cette feature
```

Si les 4 maisons ont presque le meme histogramme pour `Astronomy`, alors `Astronomy` ne permet pas beaucoup de les differencier.

Ce qu'il faut retenir :

```text
feature homogene -> les maisons se ressemblent
feature non homogene -> les maisons sont plus faciles a separer
```

---

# 4. Scatter plot et correlation de Pearson

## Pourquoi deux features ?

Un scatter plot est un graphique en 2D.

Il a deux axes :

```text
axe horizontal X = une feature
axe vertical Y = une autre feature
```

Exemple :

```text
X = Potions
Y = Charms
```

Chaque eleve devient un point :

```text
Eleve A : Potions = 10, Charms = 12 -> point (10, 12)
Eleve B : Potions = 15, Charms = 16 -> point (15, 16)
```

On prend donc deux features parce qu'un point dans un graphique 2D a deux coordonnees.

## Correlation de Pearson

La correlation de Pearson mesure si deux features bougent ensemble de maniere lineaire.

Elle donne un nombre entre `-1` et `1`.

```text
proche de 1  -> les deux features montent ensemble
proche de -1 -> une feature monte quand l'autre descend
proche de 0  -> pas de relation lineaire claire
```

Exemple avec correlation positive :

```text
Potions : 5, 10, 15
Charms  : 6, 11, 16
```

Quand `Potions` augmente, `Charms` augmente aussi.
La correlation est proche de `1`.

Exemple avec correlation negative :

```text
Potions : 5, 10, 15
Flying  : 15, 10, 5
```

Quand `Potions` augmente, `Flying` diminue.
La correlation est proche de `-1`.

Dans `scatter_plot.py`, le code teste toutes les paires de features et cherche la paire avec la plus forte correlation en valeur absolue.

Valeur absolue veut dire :

```text
0.95 et -0.95 sont toutes les deux des relations fortes
```

## Correlation ne veut pas dire causalite

Si deux features sont correlees, cela veut seulement dire qu'elles bougent ensemble.

Cela ne prouve pas que l'une cause l'autre.

Exemple :

```text
Les eleves bons en Potions sont souvent bons en Charms.
```

Cela ne veut pas dire :

```text
etre bon en Potions cause une bonne note en Charms.
```

Cela peut simplement vouloir dire :

- certains eleves sont globalement bons;
- les deux cours demandent des competences proches;
- il existe un autre facteur cache.

---

# 5. Pair plot et F-ratio dans `pair_plot.py`

## Pair plot

Un pair plot est une grande grille de graphiques.

Il compare toutes les features deux par deux.

Dans ce projet :

- la diagonale montre des histogrammes;
- les autres cases montrent des scatter plots.

But :

```text
voir quelles features separent bien les maisons
voir quelles features sont redondantes
```

## F-ratio

Le F-ratio mesure si une feature aide a separer les maisons.

Il compare deux choses :

```text
variation entre les maisons
variation a l'interieur des maisons
```

Tu peux le lire comme :

```text
F-ratio = separation entre maisons / dispersion dans les maisons
```

## Variation entre les maisons

On regarde si les moyennes des maisons sont differentes.

Exemple :

```text
Potions :
Gryffindor moyenne = 5
Hufflepuff moyenne = 10
Ravenclaw moyenne = 15
Slytherin moyenne = 20
```

Les maisons sont bien separees.

## Variation a l'interieur des maisons

On regarde si les notes sont tres dispersees dans une meme maison.

Exemple :

```text
Gryffindor en Potions : 2, 5, 8, 14, 20
```

Les notes sont tres eparpillees.
La separation est moins claire.

## Interpretation du F-ratio

```text
F-ratio eleve -> bonne feature pour separer les maisons
F-ratio faible -> feature peu utile pour classifier
```

## Features redondantes

Deux features sont redondantes si elles donnent presque la meme information.

Exemple :

```text
Potions : 5, 10, 15
Charms  : 6, 11, 16
```

`Charms` ressemble presque a `Potions + 1`.

Si le modele connait deja `Potions`, `Charms` ajoute peu d'information nouvelle.

Dans `pair_plot.py`, le code regarde :

```text
correlation forte -> features similaires
F-ratio -> laquelle separe le mieux les maisons
```

Si deux features sont tres correlees, le code recommande de garder celle avec le meilleur F-ratio.

---

# 6. Normalisation dans `logreg_train.py`

Les features du dataset n'ont pas toutes la meme echelle.

Exemple :

```text
Defense Against the Dark Arts : -1000 a 1000
Astronomy : -10 a 10
```

Si on donne ces valeurs brutes au modele, la feature avec les grandes valeurs peut dominer l'apprentissage.

Pour eviter ca, on normalise.

## Z-score

La normalisation utilisee est le z-score.

Recette :

```text
valeur normalisee = (valeur - moyenne) / ecart type
```

Effet :

```text
la moyenne devient environ 0
les valeurs sont exprimees en nombre d'ecarts types autour de la moyenne
```

Exemple :

```text
note = 15
moyenne = 10
ecart type = 5

valeur normalisee = (15 - 10) / 5 = 1
```

Cela veut dire :

```text
cette note est 1 ecart type au-dessus de la moyenne
```

## Pourquoi sauvegarder la moyenne et l'ecart type ?

Le modele est entraine avec les moyennes et ecarts types du train set.

Pendant la prediction, il faut reutiliser exactement les memes valeurs.

Sinon, le test set ne sera pas sur la meme echelle que le train set.

C'est pour ca que `weights.json` contient :

```text
means
stds
```

---

# 7. Regression logistique

Malgre son nom, la regression logistique sert souvent a faire de la classification.

Dans ce projet, elle sert a predire une maison.

## Etape 1 : calculer un score

Le modele commence par calculer un score lineaire :

```text
score = biais + poids1 * feature1 + poids2 * feature2 + ...
```

Dans le code :

```python
z = w[0] + sum(w[j + 1] * X[i][j] for j in range(n))
```

Ce score peut etre n'importe quel nombre :

```text
-200
-4
0
3
150
```

Mais une probabilite doit etre entre `0` et `1`.

## Etape 2 : passer le score dans la sigmoid

La sigmoid transforme n'importe quel score en valeur entre `0` et `1`.

```text
score tres negatif -> proche de 0
score = 0 -> 0.5
score tres positif -> proche de 1
```

Donc :

```text
score lineaire -> sigmoid -> probabilite
```

Exemple :

```text
score = 3
sigmoid(score) = environ 0.95
```

Le modele peut lire ca comme :

```text
forte probabilite d'appartenir a cette classe
```

## Pourquoi c'est une probabilite ?

Parce que la sigmoid force la sortie a rester entre `0` et `1`.

Et pendant l'entrainement, le modele apprend ses poids pour que :

```text
vraie classe -> probabilite proche de 1
mauvaise classe -> probabilite proche de 0
```

---

# 8. One-vs-all

La regression logistique de base repond a une question binaire :

```text
oui ou non ?
classe 1 ou classe 0 ?
```

Mais DSLR a 4 maisons.

On utilise donc une strategie one-vs-all.

Le code entraine 4 modeles :

```text
modele Gryffindor : Gryffindor ou pas Gryffindor ?
modele Hufflepuff : Hufflepuff ou pas Hufflepuff ?
modele Ravenclaw  : Ravenclaw ou pas Ravenclaw ?
modele Slytherin  : Slytherin ou pas Slytherin ?
```

Chaque modele donne une probabilite.

Exemple :

```text
Gryffindor : 0.12
Hufflepuff : 0.25
Ravenclaw  : 0.87
Slytherin  : 0.30
```

On choisit la plus grande :

```text
prediction = Ravenclaw
```

C'est ce que fait `logreg_predict.py`.

---

# 9. Descente de gradient

La descente de gradient est la methode qui permet au modele d'apprendre les poids.

Au debut :

```text
tous les poids = 0
```

Le modele fait des predictions mauvaises.

Ensuite, pour chaque tour d'entrainement :

1. Il fait une prediction.
2. Il compare avec la vraie reponse.
3. Il calcule l'erreur.
4. Il ajuste un peu les poids.
5. Il recommence.

Dans le code, l'erreur est :

```python
err = h - binary_y[i]
```

- `h` = prediction du modele;
- `binary_y[i]` = vraie reponse, 0 ou 1.

Puis les poids sont mis a jour :

```python
w[j] -= lr * grad[j] / m
```

Avec :

- `lr` = learning rate, la taille du pas;
- `grad` = direction de correction;
- `m` = nombre d'exemples.

## Learning rate

Le learning rate controle la taille des corrections.

```text
learning rate trop grand -> le modele peut devenir instable
learning rate trop petit -> le modele apprend tres lentement
```

Dans le code :

```python
lr = 0.5
```

## Epochs

Une epoch est un passage complet sur le dataset.

Dans le code :

```python
epochs = 1000
```

Cela veut dire que le modele repete l'apprentissage 1000 fois.

---

# 10. Entrainement puis prediction

## `logreg_train.py`

Ce fichier :

1. lit `dataset_train.csv`;
2. calcule les moyennes et ecarts types;
3. remplace les valeurs manquantes;
4. normalise les features;
5. entraine 4 modeles one-vs-all;
6. sauvegarde tout dans `weights.json`.

`weights.json` contient :

```text
features
houses
means
stds
weights
```

## `logreg_predict.py`

Ce fichier :

1. lit `dataset_test.csv`;
2. lit `weights.json`;
3. applique la meme normalisation;
4. calcule les probabilites pour les 4 maisons;
5. choisit la plus grande probabilite;
6. ecrit `houses.csv`.

Format attendu :

```csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
```

---

# 11. Pipeline complet du projet

```text
dataset_train.csv
  |
  |-- describe.py
  |     comprendre les statistiques
  |
  |-- histogram.py
  |     voir les distributions
  |
  |-- scatter_plot.py
  |     trouver les features similaires
  |
  |-- pair_plot.py
  |     choisir les features utiles
  |
  |-- logreg_train.py
        apprendre les poids
        produire weights.json

dataset_test.csv + weights.json
  |
  |-- logreg_predict.py
        produire houses.csv
```

---

# 12. Ce qu'il faut retenir pour la soutenance

## `describe.py`

Il sert a comprendre les colonnes numeriques :

```text
moyenne, ecart type, quartiles, min, max
```

## `histogram.py`

Il sert a comparer les distributions des maisons pour chaque cours.

Une feature homogene separe mal les maisons.

## `scatter_plot.py`

Il sert a trouver deux features similaires avec la correlation de Pearson.

Deux features correlees peuvent etre redondantes.

## `pair_plot.py`

Il sert a voir toutes les relations entre features.

Le F-ratio aide a choisir les features qui separent le mieux les maisons.

## `logreg_train.py`

Il entraine 4 regressions logistiques, une par maison.

Il utilise :

```text
normalisation
sigmoid
one-vs-all
descente de gradient
```

## `logreg_predict.py`

Il charge le modele et predit la maison avec la plus grande probabilite.

---

# 13. Resume ultra simple

```text
describe.py
  -> comprendre les donnees

histogram.py
  -> voir si les maisons ont des distributions similaires

scatter_plot.py
  -> voir si deux features se ressemblent

pair_plot.py
  -> choisir les features utiles

logreg_train.py
  -> apprendre les poids du modele

logreg_predict.py
  -> utiliser les poids pour predire les maisons
```

La regression logistique fait :

```text
notes -> score -> sigmoid -> probabilite
```

Le one-vs-all fait :

```text
une probabilite par maison
on choisit la plus grande
```
