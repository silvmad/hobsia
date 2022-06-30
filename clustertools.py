import numpy as np
import matplotlib.pyplot as plt
import os
import re
from itertools import zip_longest

# n mots les plus fréquents ignorés
n_most_freq_ignored = 40
# Nombre de mots les plus fréquents donnés dans les infos
n_clust_info = 20
# Fichier contenant les mots clés haineux
kw_file = "kw_hate.txt"

#Paramètres des figures
plt.rcParams["figure.figsize"] = (13,10)
plt.rcParams["lines.markersize"] = 4
plt.rcParams["scatter.marker"] = '.'

def init_globals(cd, rd, tdd, cld):
    """
    Initialisation des variables globales.
    cd : jeu de données nettoyé
    rd : jeu de données brut
    tdd : jeu de données encodé en deux dimensions
    cld : dossier dans lequel enregistrer les clusters
    """
    global clean_dataset_ct
    clean_dataset_ct = cd
    global raw_dataset_ct
    raw_dataset_ct = rd
    global two_dim_dataset_ct
    two_dim_dataset_ct = tdd
    global cluster_dir_ct
    cluster_dir_ct = cld
    global ignored_ct
    ignored_ct = [tup[0] for tup in create_sorted_wlist(clean_dataset_ct)[:n_most_freq_ignored]]


#########################################
# Analyse et manipulation des résultats #
#########################################
    

def create_wdict(lines):
    """
    Création du dictionnaire de mots avec leur occurence.
    lines : Jeu de données sous forme d'une une liste de de messages.
    Les messages doivent être eux même sous forme d'une liste de mots (sous 
    forme de chaîne de caractères)
    Renvoie un dictionnaire associant chaque mot du jeu de données 
    avec son nombre d'occurences.
    """
    dico = {}
    for msg in lines:
        for mot in msg:
            if mot in dico: 
                dico[mot] += 1
            else:
                dico[mot] = 1
    return dico

def create_sorted_wlist(lines):
    """
    Création de la liste des mots triée par nombre d'occurences décroissant.
    lines : Jeu de données sous forme d'une une liste de messages.
    Les messages doivent être eux même sous forme d'une liste de mots (sous 
    forme de chaîne de caractères).
    Renvoie une liste de tuples associant un mot à son nombre d'occurences 
    dans le jeu de données triée par nombre d'occurences décroissant.
    """
    dico = create_wdict(lines)
    # Crée une liste de tuples associant mot et nombre d'occurences
    wlist = [(mot, dico[mot]) for mot in dico.keys()]
    # Renvoie la liste triée par nombre d'occurences décroissant
    return sorted(wlist, key = lambda mot: mot[1], reverse = True)

def build_res_dict(pred):
    """
    Construire le dictionnaire des résultats.
    pred : La prédiction donnée par le modèle de partitionnement. Il s'agit 
    d'une liste d'entiers de même taille que le jeu de données. Chaque 
    élément de la liste correspond au message de même indice dans le jeu de
    données. La valeur de cet élément correspond au cluster auquel appartient
    le message correspondant.
    Renvoie le dictionnaire des résultats  qui à chaque cluster associe la 
    liste des indices des messages qu'il contient. 
    """
    res = {}
    for i, lab in enumerate(pred): 
        if lab in res:
            res[lab].append(i)
        else:
            res[lab] = [i]
    return res
    
def create_word_and_msg_lists(res):
    """
    Construire la liste des mots et les listes des messages (bruts, nettoyés 
    et encodés) de chaque cluster.
    res : Le dictionnaire des résultats qui à chaque cluster associe la 
    liste des indices des messages qu'il contient (construit par la fonction
    build_res_dict). 
    Valeurs de retour :
    wlists : liste des listes de mots triées par ordre décroissant
    d'occurences pour chaque cluster.
    mlists : liste des listes de messages bruts de chaque cluster.
    cmlists : liste des listes de messages nettoyés de chaque cluster.
    e2dmlists : liste des listes de messages encodés en deux dimensions de 
    chaque cluster.
    Utilise les variables globales clean_dataset_ct et raw_dataset_ct
    """
    wlists = []
    mlists = []
    cmlists = []
    e2dmlists = []
    # Parcourt les clés (donc les clusters) dans l'ordre croissant
    for i in sorted(res):
        clean_msg_list = []
        raw_msg_list = []
        e2d_msg_list = []
        # Création des listes de messages
        for idx in res[i]:#idx_list: 
            clean_msg_list.append(clean_dataset_ct[idx])
            raw_msg_list.append(raw_dataset_ct[idx])
            e2d_msg_list.append(two_dim_dataset_ct[idx])
        wlists.append(create_sorted_wlist(clean_msg_list))
        mlists.append(raw_msg_list)
        cmlists.append(clean_msg_list)
        e2dmlists.append(e2d_msg_list)
    return wlists, mlists, cmlists, e2dmlists


def most_freq_words(n, wlists, ignored_ct):
    """
    Détermine les mots les plus fréquents de chaque cluster
    Renvoie un dictionnaire qui associe à chaque cluster la liste des 
    n mots les plus fréquents
    n : nombre de mots les plus fréquents à considérer
    wlists : liste de listes contenant les messages des clusters
    ignored_ct : lists des mots à ignorer
    """
    ret = []
    for i, tuplist in enumerate(wlists):
        j = 0
        k = 0
        wlist = []
        # Boucle jusqu'à avoir n mots ou être arrivé au bout de la liste
        while (j < n and k < len(tuplist)):
            word = tuplist[k][0]
            if word not in ignored_ct:
                wlist.append(word)
                j += 1
            k += 1
        ret.append(wlist)
    return ret

def search_hate_words(mlists, file):
    """
    Recherche les mots-clé haineux dans les messages de chaque cluster.
    mlists : liste contenant les messages bruts (sous forme de liste de
    mots) de chaque cluster.
    file : fichier contenant les mots-clés haineux.
    Renvoie une liste de tuples qui associent pour chaque cluster le nombre 
    et le pourcentage de messages contenant un mot-clé haineux.
    """
    ret = []
    with open(file, "r") as f:
        kws = f.read().splitlines()
    for i, mlist in enumerate(mlists):
        n_msg = 0
        for msg in mlist:
            for kw in kws:
                if kw in msg:
                    n_msg += 1
                    break
        perc = (n_msg / len(mlist))*100
        ret.append((n_msg, perc))
    return ret

def n_msg_by_clust(res):
    """
    Crée la liste du nombre de messages par cluster.
    res : dictionnaire des résultats (obtenu avec la fonction build_res_dict)
    Renvoie une liste de tuples qui associent pour chaque cluster son numéro 
    à son nombre de messages.
    """
    return [(i, len(res[i])) for i in sorted(res)]

##############
# Impression
##############
        
def print_clusters_info(n_msg, mfw, hkw, hper=[], centers=[]):
    """
    Imprime des informations sur chaque cluster.
    n_msg : liste du nombre de messages par cluster.
    mfw : liste des mots les plus fréquents par cluster.
    hkw : liste du pourcentage de messages contenant au moins un mot haineux
    par cluster.
    hper : liste du pourcentage de messages détecté comme haineux par le 
    modèle de validation.
    centers : liste des mots les plus proches du centre par cluster.
    """
    for n, m, h, p, c in zip_longest(n_msg, mfw, hkw, hper, centers):
        print_cluster_info(n, m, h, p, c)
        
def print_cluster_info(n_msg, mfw, hkw, hper=None, centers=None):
    """
    Imprime les information d'un cluster.
    n_msg : tuple associant le numéro du cluster à son nombre de messages.
    mfw : liste des mots les plus fréquents du cluster.
    hkw : tuple associant le nombre de messages contenant au moins un mot clé 
    haineux dans le cluster au pourcentage de messages que cela représente.
    hper : pourcentage de messages détectés haineux par le modèle de validation
    dans le cluster.
    centers : liste des mots les plus proches du centre du cluster.
    """
    print("Cluster {} : {} messages".format(n_msg[0], n_msg[1]))
    print("{} messages contiennent un mot-clé haineux (soit {:.2f}%)".format(hkw[0], hkw[1]))
    if (hper is not None):
        print("{:.2f}% de messages sont détectés comme haineux par le modèle".format(hper))
    print("Les {} mots les plus fréquents :".format(len(mfw)), end=' ')
    for word in mfw:
        print(word, end=' ')
    print()
    if (centers):
        print("Les mots les plus proches du centre :", end = ' ')
        for w in centers:
            print(w, end=' ')
    print('\n')
    
def print_hate_clusters_info(n_msg, mfw, hkw, hper=[], centers=[]):
    """
    Idem que print_clusters_info mais n'imprime les informations que pour les
    clusters ayant plus de 10% de messages contenant au moins un mot haineux
    ou au moins 50% de messages détectés comme haineux par le modèle de 
    validation.
    
    n_msg : liste du nombre de messages par cluster.
    mfw : liste des mots les plus fréquents par cluster.
    hkw : liste du pourcentage de messages contenant au moins un mot haineux
    par cluster.
    hper : liste du pourcentage de messages détecté comme haineux par le 
    modèle de validation.
    centers : liste des mots les plus proches du centre par cluster.
    """
    for n, m, h, p, c in zip_longest(n_msg, mfw, hkw, hper, centers):
        if (h[1] > 10 or (p and p > 50)):
            print_cluster_info(n, m, h, p, c)

##################################################################
# Affichage de graphiques                                        #
##################################################################

def plot_results(pred, tdd):
    """
    Affiche un graphique représentant les résultats du clustering.
    Tous les messages sont représentés par des points sur un graphique, les 
    messages appartenant à un même cluster sont de la même couleur.
    
    pred : La prédiction donnée par le modèle de clustering.
    tdd : Le jeu de données encodé en deux dimensions.
    """
    tcd = tdd.T
    plt.scatter(tcd[0], tcd[1], c=pred, cmap=plt.get_cmap('gist_rainbow'))
    plt.colorbar()
    plt.show()

def plot_clusters(e2dmlists):
    """
    Comme plot_results mais à partir d'un argument différent.
    e2dmlists : la liste des messages encodés en deux dimensions de chaque 
    cluster.
    
    Utile pour visualiser les clusters que l'on a chargé depuis le disque.
    """
    pred = []
    tdd = np.concatenate(e2dmlists, axis=0)
    for i, mlist in enumerate(e2dmlists):
        pred += [i for e in mlist]
    tml = tdd.T
    plt.scatter(tml[0], tml[1], c= pred, cmap=plt.get_cmap('gist_rainbow'))
    plt.colorbar()
    plt.show

##################################################################
# Fonctions d'enregistrement sur disque                          #
##################################################################    
    
def save_clusters_info(n_msg, mfw, hkw, method, hper=[], centers=[]):
    """
    Écrit les informations sur les clusters dans un fichier. Le nom du fichier
    obéit à une convention de nommage et sera placé dans le même dossier que
    les clusters.
    
    n_msg : liste du nombre de messages par cluster.
    mfw : liste des mots les plus fréquents par cluster.
    hkw : liste du pourcentage de messages contenant au moins un mot haineux
    par cluster.
    hper : liste du pourcentage de messages détecté comme haineux par le 
    modèle de validation.
    centers : liste des mots les plus proches du centre par cluster.
    """
    fname = cluster_dir_ct + "{0}/{0}_clusters_info.txt".format(method)
    for n, m, h, p, c in zip_longest(n_msg, mfw, hkw, hper, centers):
        save_cluster_info(fname, n, m, h, p, c)

def save_cluster_info(file, n_msg, mfw, hkw, hper, centers):
    """
    Écrit les informations sur un cluster dans un fichier sans l'écraser.
    
    file : Le nom du fichier dans lequel écrire les informations.
    n_msg : tuple associant le numéro du cluster à son nombre de messages.
    mfw : liste des mots les plus fréquents du cluster.
    hkw : tuple associant le nombre de messages contenant au moins un mot clé 
    haineux dans le cluster au pourcentage de messages que cela représente.
    hper : pourcentage de messages détectés haineux par le modèle de validation
    dans le cluster.
    centers : liste des mots les plus proches du centre du cluster.
    """
    with open(file, "a") as f:
        f.write("Cluster {} : {} messages\n".format(n_msg[0], n_msg[1]))
        f.write("{} messages contiennent un mot-clé haineux (soit {:.2f}%)\n".format(hkw[0], hkw[1]))
        if (hper):
            f.write("{:.2f}% de messages sont détectés comme haineux par le modèle\n".format(hper))
        f.write("Les {} mots les plus fréquents : ".format(len(mfw)))
        for word in mfw:
            f.write(word + ' ')
        f.write('\n')
        if centers:
            f.write("Les mots les plus proches du centre : ")
            for w in centers:
                f.write(w + ' ')
        f.write('\n\n')

def prepare_clusters_dir(method):
    """
    Vérifie l'existence des dossiers de clusters et les crée si besoin.
    
    method : La méthode de clustering, permet de retrouver les noms de dossier
    de clusters grâce aux conventions de nommage.
    
    Utilise la variable globale cluster_dir_ct
    """
    if not os.path.exists(cluster_dir_ct): 
        os.mkdir(cluster_dir_ct)
    if not os.path.exists(cluster_dir_ct + method): 
        os.mkdir(cluster_dir_ct + method)
    # Si le dossier existe déjà on efface tous les fichiers qu'il contient
    else :
        for file in os.scandir(cluster_dir_ct + method):
            os.remove(file.path)
    
def save_clusters_raw_msg(mlists, method):
    """
    Enregistre les messages bruts par cluster.
    
    mlists : Liste contenant les listes de messages bruts de chaque cluster.
    method : La méthode d'encodage sous forme de chaîne de caractères.
    
    Utilise la variable globale cluster_dir_ct.
    """
    for i, mlist in enumerate(mlists):
        save_cluster_raw_msg(mlist, cluster_dir_ct + "{0}/{0}_cluster_{1}_raw_msg.txt".format(method, i))

def save_cluster_raw_msg(mlist, file):
    """
    Enregistre les messages bruts d'un cluster dans un fichier.
    
    mlist : Liste des messages bruts de ce cluster.
    file : Nom du fichier dans lequel écrire les messages.
    """
    with open(file, "w") as f:
        for msg in mlist:
            f.write(msg + '\n')
                
def save_clusters_two_dim_enc_msg(e2dmlists, method):
    """
    Enregistre les messages encodés en 2 dimensions par cluster.
    
    e2dmlists : La liste des messages encodés en deux dimensions de chaque 
    cluster.
    method : La méthode de clustering.
    """
    for i, e2dmlist in enumerate(e2dmlists):
        np.save(cluster_dir_ct + "{0}/{0}_cluster_{1}_enc_msg.npy".format(method, i), e2dmlist)    

def save_clusters_clean_msg(cmlists, method):
    """
    Enregistre les messages nettoyés par cluster.
    
    cmlists : La liste des messages nettoyés de chaque cluster.
    method : La méthode de clustering.
    """
    for i, cmlist in enumerate(cmlists):
        save_cluster_clean_msg(cmlist, cluster_dir_ct + "{0}/{0}_cluster_{1}_clean_msg.txt".format(method, i))

def save_cluster_clean_msg(cmlist, file):
    """
    Enregistre les messages nettoyés d'un cluster dans un fichier.
    
    cmlist : Liste des messages nettoyés du cluster.
    file : Fichier dans lequel écrire les messages.
    """
    with open(file, "w") as f:
        for msg in cmlist:
            f.write(" ".join(msg) + '\n')
    
def save_clusters_msg(cmlists, mlists, e2dmlists, method):
    """
    Enregistre les messages par cluster.
    
    cmlists : La liste des messages nettoyés de chaque cluster.
    mlists : Liste contenant les listes de messages bruts de chaque cluster.
    e2dmlists : La liste des messages encodés en deux dimensions de chaque 
    cluster.
    """
    prepare_clusters_dir(method)
    save_clusters_raw_msg(mlists, method)
    save_clusters_clean_msg(cmlists, method)
    save_clusters_two_dim_enc_msg(e2dmlists, method)

#######################################################     
#   Fonctions pour analyser les clusters en mémoire   #
#######################################################

def parse(pred):
    """
    Fonction de confort pour un traitement rapide.
    
    pred : La prédiction du modèle de clustering
    
    Renvoie :
    res : Dictionnaire des résultats.
    wlists : Liste des listes de mots triées par ordre décroissant
    d'occurences pour chaque cluster.
    mlists : Liste des listes de messages bruts de chaque cluster.
    cmlists : Liste des listes de messages nettoyés de chaque cluster.
    e2dmlists : Liste des listes de messages encodés en deux dimensions de 
    chaque cluster.
    mfw : Liste des mots les plus fréquents par cluster.
    hkw : Liste du pourcentage de messages contenant au moins un mot haineux
    par cluster
    clust_n_msg : Liste du nombre de messages par cluster.
    """
    res = build_res_dict(pred)
    wlists, mlists, cmlists, e2dmlists = create_word_and_msg_lists(res)
    mfw = most_freq_words(n_clust_info, wlists, ignored_ct)
    hkw = search_hate_words(mlists, kw_file)
    clust_n_msg = n_msg_by_clust(res)
    return res, wlists, mlists, cmlists, e2dmlists, mfw, hkw, clust_n_msg
    
def parse_results(pred, method, centers=[]):
    """
    Procédure qui analyse les résultats du clustering et sauvegarde des 
    résultats.
    
    pred : La prédiction du modèle de clustering.
    method : La méthode de clustering (chaîne de caractères).
    centers : Liste des messgaes les plus proches du centre pour chaque 
    cluster.
    """
    res, wlists, mlists, cmlists, e2dmlists, mfw, hkw, clust_n_msg = parse(pred)
    print_clusters_info(clust_n_msg, mfw, hkw, centers)
    plot_results(pred, two_dim_dataset_ct)
    save_clusters_msg(cmlists, mlists, e2dmlists, method)
    save_clusters_info(clust_n_msg, mfw, hkw, method, centers)

def parse_no_save(pred, centers=[], hper=[]):
    """
    Procédure qui analyse sans sauvegarder.
    
    pred : La prédiction du modèle de clustering.
    centers : Liste des messgaes les plus proches du centre pour chaque 
    cluster.
    hper : Liste contenant le pourcentage de messages détectés comme haineux 
    par le modèle de validation de chaque cluster.
    """
    res, wlists, mlists, cmlists, e2dmlists, mfw, hkw, clust_n_msg = parse(pred)
    print_clusters_info(clust_n_msg, mfw, hkw, hper, centers)
    plot_results(pred, two_dim_dataset_ct)
                       
##########################################################################        
#   Fonctions pour analyser les clusters enregistrés dans des fichiers   #
##########################################################################

def load_cluster(file):
    """
    Charge les messages d'un cluster.
    
    file : Fichier contenant les messages du cluster.
    Renvoie une liste contenant les messages du cluster.
    """
    with open(file, "r") as f:
        clust_msgs = f.read().splitlines()
    return clust_msgs

def extr_num(string):
    """
    Renvoie le numéro de cluster dans une chaîne correspondant à un nom de fichier
    Permet de trier les noms de fichiers contenant les messages des
    clusters par ordre croissant.
    
    string : Nom du fichier.
    """
    num = re.search("cluster_([0-9]+)_", string)
    return int(num.group(1)) if num else -1

def load_clusters(clusters_dir):
    """
    Charge en mémoire tous les clusters d'un répertoire.

    clusters_dir : Le nom du répertoire contenant les clusters, il doit 
    impérativement se terminer par /.
    
    Valeurs de retour :
    clean_clusters : La liste des messages nettoyés de chaque cluster.
    raw_clusters : La liste des messages bruts de chaque cluster.
    two_dim_clusters : La liste des messages encodés en deux dimensions.
    de chaque cluster.
    """
    dir_it = os.scandir(path=clusters_dir)
    names = [entry.name for entry in dir_it]
    names.sort(key=extr_num)
    clean_clusters = [load_cluster(clusters_dir + name) for name in names if ("clean" in name)]
    clean_clusters = [[msg.split() for msg in clust] for clust in clean_clusters]
    raw_clusters = [load_cluster(clusters_dir + name) for name in names if ("raw" in name)]
    two_dim_clusters = [np.load(clusters_dir + name) for name in names if ("enc" in name)]
    return clean_clusters, raw_clusters, two_dim_clusters
    
def parse_clusters(clean_clusters, raw_clusters, two_dim_clusters):
    """
    Procédure analysant les clusters chargés depuis le disque et imprimant les 
    résultats.
    
    clean_clusters : La liste des messages nettoyés de chaque cluster.
    raw_clusters : La liste des messages bruts de chaque cluster.
    two_dim_clusters : La liste des messages encodés en deux dimensions.
    """
    dataset = []
    for cluster in clean_clusters:
        dataset += cluster
    wlists = [create_sorted_wlist(cluster) for cluster in clean_clusters]
    ignored_ct = [tup[0] for tup in create_sorted_wlist(dataset)[:n_most_freq_ignored]]
    mfw = most_freq_words(n_clust_info, wlists, ignored_ct)
    hkw = search_hate_words(raw_clusters, kw_file)
    clust_n_msg = [(i, len(clust)) for i, clust in enumerate(clean_clusters)]
    print_clusters_info(clust_n_msg, mfw, hkw)
    plot_clusters(two_dim_clusters)