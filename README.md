# hobsia
Suite de scripts, notebooks, outils pour entraîner une IA à reconnaître des messages haineux sur twitter.

## Notebooks
### Auto encodeurs
- autoencoder_compare_visualizations : comparaison des différentes techniques de visualisation.
- autoencoder_deep                   : entraînement d'un modèle d'auto encodeur à 5 couches cachées
- autoencoder_feature_extraction     : entraînement d'un autoencodeur pour l'extraction de caractéristiques (espace latent à 15 dimensions)
- autoencoder_load_feature_extr      : code de présentation du chargement et de l'utilisation de l'auto encodeur pour l'extraction de caractéristiques
- autoencoder_medium_deep            : entraînement d'un modèle d'auto encodeur à 3 couches cachées
- autoencoder                        : entraînement d'un modèle d'auto encodeur à 1 couche cachée

### Partitionnement
- clustering_clusters_KMeans : second partitionnement (sur les résultats du premier) avec KMeans
- clustering_clusters_SOM    : second partitionnement avec SOM
- clustering_KMeans          : partitionnement avec KMeans
- clustering_minisom         : partitionnement avec SOM

### Encodage des messages
- encoding_bertweetfr : encodage avec bertweetfr, un modèle CamemBERT dont le pré-entraînement a été poursuivi sur 15Go de tweets pour l'adapter à Twitter.
- encoding_doc2v : encodage avec un modèle Doc to Vec.
- encoding_flaubert : encodage avec le transformer FlauBERT.
- encoding_one_hot : encodage one-hot.
- encoding_sentence-transformer : encodage avec un modèle multilangue du transformer SBERT.
- encoding_train_embedding_models : entraînement de modèles d'embedding : Word to Vec et Doc to Vec.

### Modèle de classification final
- final_model_fine_tuning_berttweetfr : réglage fin de bertweetfr sur une tâche de classification avec notre jeu de données final

### Analyse des résultats du partitionnement
- parse_clusters_IF_method_eval : évaluation des résultats d'isolation forest afin de déterminer le taux de contamination à utiliser.
- parse_clusters_info : analyse générale des résultats du partitionnement
- parse_clusters : présentation du chargement de résultats précédents pour une nouvelle analyse

### Entraînement du modèle de validation
- val_dehatebert_test : avec un modèle multilangue du transformer SBERT
- val_DNN_model : avec un réseau de neurones dense classique
- val_fine_tuning_berttweetfr_IDS : avec bertweetfr et un échantilloneur pour équilibrer le jeu de données
- val_fine_tuning_bertweetfr : avec bertweetsfr
- val_fine_tuning_flaubert : avec flauBERT

## Modules
- clustertools : outils pour l'analyse et la manipulation des résultats du partitionnement
- confmat : affichage d'une matrice de confusion

## Scripts
- interface_test.py : une petite interface graphique pour tester le modèle de classification final
- labeling.py : une petite interface graphique pour faciliter l'étiquetage

## Fichiers divers
- kw_hate.txt : liste de mots-clé haineux