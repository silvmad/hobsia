import numpy as np
import matplotlib.pyplot as plt
import os
import re
from itertools import zip_longest

# n mots les plus fréquents ignorés
n_most_freq_ignored = 40
# Nombre de mots les plus fréquents donnés dans les infos
n_clust_info = 20
kw_file = "kw_hate.txt"

#Paramètres des figures
plt.rcParams["figure.figsize"] = (13,10)
plt.rcParams["lines.markersize"] = 4
plt.rcParams["scatter.marker"] = '.'

def init_globals(cd, rd, tdd, cld):
    global clean_dataset_ct
    clean_dataset_ct = cd
    global raw_dataset_ct
    raw_dataset_ct = rd
    #global dataset_encoded
    #dataset_encoded = ed
    global two_dim_dataset_ct
    two_dim_dataset_ct = tdd
    global cluster_dir_ct
    cluster_dir_ct = cld
    global ignored_ct
    ignored_ct = [tup[0] for tup in create_sorted_wlist(clean_dataset_ct)[:n_most_freq_ignored]]


#########################################
# Analyse et manipulation des résultats #
#########################################
    
#Création du dictionnaire de mots avec leur occurence
def create_wdict(lines):
    dico = {}
    for msg in lines:
        for mot in msg:
            if mot in dico: 
                dico[mot] += 1
            else:
                dico[mot] = 1
    return dico

#Création de la liste des mots triée par nombre d'occurences décroissant
def create_sorted_wlist(lines):
    dico = create_wdict(lines)
    # Crée une liste de tuples associant mot et nombre d'occurences
    wlist = [(mot, dico[mot]) for mot in dico.keys()]
    # Renvoie la liste triée par nombre d'occurences décroissant
    return sorted(wlist, key = lambda mot: mot[1], reverse = True)

#Construire le dictionnaire des résultats qui à chaque cluster associe la liste indices des messages correspondants
def build_res_dict(pred):
    res = {}
    for i, lab in enumerate(pred): 
        if lab in res:
            res[lab].append(i)
        else:
            res[lab] = [i]
    return res
    
#construire la liste des mots et les liste des messages (bruts, nettoyés et encodés) de chaque cluster
#utilise les variables globales clean_dataset_ct et raw_dataset_ct
def create_word_and_msg_lists(res):
    wlists = []
    mlists = []
    #emlists = []
    cmlists = []
    e2dmlists = []
    # Parcourt les clés (donc les clusters) dans l'ordre croissant
    for i in sorted(res):
        idx_list = []
        clean_msg_list = []
        raw_msg_list = []
        #encoded_msg_list = []
        e2d_msg_list = []
        # Création de la liste des indices pour ce cluster
        for n in res[i]: idx_list.append(n)
        # Création des listes de messages
        for idx in idx_list: 
            clean_msg_list.append(clean_dataset_ct[idx])
            raw_msg_list.append(raw_dataset_ct[idx])
            #encoded_msg_list.append(dataset_encoded[idx])
            e2d_msg_list.append(two_dim_dataset_ct[idx])
        wlists.append(create_sorted_wlist(clean_msg_list))
        mlists.append(raw_msg_list)
        #emlists.append(encoded_msg_list)
        cmlists.append(clean_msg_list)
        e2dmlists.append(e2d_msg_list)
    return wlists, mlists, cmlists, e2dmlists

# Détermine les mots les plus fréquents de chaque cluster
# Renvoie un dictionnaire qui associe à chaque cluster la liste des 
# n mots les plus fréquents
def most_freq_words(n, wlists, ignored_ct):
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
            
        #ret[i] = []
        #for tup in tuplist[:n]:
        #    ret[i].append(tup[0])
        #ret[i] = [tup[0] for tup in tuplist[:n]]
    return ret

# Recherche les mots-clé haineux dans les messages de chaque cluster.
# Renvoie un dictionnaire qui associe à chaque cluster le nombre et le 
# pourcentage de messages contenant un mot-clé haineux.
def search_hate_words(mlists, file):
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

# Renvoie une liste de tuples qui associent pour chaque cluster son numéro à son nombre de messages
def n_msg_by_clust(res):
    return [(i, len(res[i])) for i in sorted(res)]

##############
# Impression
##############

# Pour mémoire le temps de bidouiller 
# def print_clusters_info(n_msg, mfw, hkw, centers=[]):
#     for i in range(len(n_msg)):
#         print("Cluster {} : {} messages".format(i, n_msg[i]))
#         print("{} messages contiennent un mot-clé haineux (soit {:.2f}%)".format(hkw[i][0], hkw[i][1]))
#         print("Les {} mots les plus fréquents :".format(len(mfw[i])), end=' ')
#         for word in mfw[i]:
#             print(word, end=' ')
#         print()
#         if len(centers) > 0:
#             print("Les mots les plus proches du centre :", end = ' ')
#             for w in centers[i]:
#                 print(w, end=' ')
#         print('\n')
        
def print_clusters_info(n_msg, mfw, hkw, centers=[]):
    for n, m, h, c in zip_longest(n_msg, mfw, hkw, centers):
        print_cluster_info(n, m, h, c)
        
def print_cluster_info(n_msg, mfw, hkw, centers=[]):
    print("Cluster {} : {} messages".format(n_msg[0], n_msg[1]))
    print("{} messages contiennent un mot-clé haineux (soit {:.2f}%)".format(hkw[0], hkw[1]))
    print("Les {} mots les plus fréquents :".format(len(mfw)), end=' ')
    for word in mfw:
        print(word, end=' ')
    print()
    if (centers):
        print("Les mots les plus proches du centre :", end = ' ')
        for w in centers:
            print(w, end=' ')
    print('\n')

def print_hate_clusters_info(n_msg, mfw, hkw, centers=[]):
    for i in range(len(n_msg)):
        if (hkw[i][1] > 5):
            print("Cluster {} : {} messages".format(i, n_msg[i]))
            print("{} messages contiennent un mot-clé haineux (soit {:.2f}%)".format(hkw[i][0], hkw[i][1]))
            print("Les {} mots les plus fréquents :".format(len(mfw[i])), end=' ')
            for word in mfw[i]:
                print(word, end=' ')
            print()
            if len(centers) > 0:
                print("Les mots les plus proches du centre :", end = ' ')
                for w in centers[i]:
                    print(w, end=' ')
            print('\n')
        else : 
            print("Cluster {} : {:.2f}% de messages contiennent un mot-clé haineux".format(i, hkw[i][1]))

##################################################################
# Affichage de graphiques                                        #
##################################################################

# # Crée le graphique des résultats
# def plot_results(res, tdd=two_dim_dataset_ct):
#     for key in res.keys():
#         cluster_data = [tdd[idx] for idx in res[key]]
#         tcd = np.array(cluster_data).T
#         plt.scatter(tcd[0], tcd[1])#, s=20, marker='.')
#     plt.show()

# Crée le graphique des résultats
def plot_results(pred, tdd):
    tcd = tdd.T
    plt.scatter(tcd[0], tcd[1], c=pred, cmap=plt.get_cmap('gist_rainbow'))
    plt.colorbar()
    plt.show()

def plot_clusters(e2dmlists):
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
    
def save_clusters_info(n_msg, mfw, hkw, method, centers=[]):
    fname = cluster_dir_ct + "{0}/{0}_clusters_info.txt".format(method)
    for n, m, h, c in zip_longest(n_msg, mfw, hkw, centers):
        save_cluster_info(fname, n, m, h, c)

def save_cluster_info(file, n_msg, mfw, hkw, centers):
    with open(file, "a") as f:
        f.write("Cluster {} : {} messages\n".format(n_msg[0], n_msg[1]))
        f.write("{} messages contiennent un mot-clé haineux (soit {:.2f}%)\n".format(hkw[0], hkw[1]))
        f.write("Les {} mots les plus fréquents : ".format(len(mfw)))
        for word in mfw:
            f.write(word + ' ')
        f.write('\n')
        if centers:
            f.write("Les mots les plus proches du centre : ")
            for w in centers:
                f.write(w + ' ')
        f.write('\n\n')

# Vérifie l'existence des dossiers de clusters et les crée si besoin.
def prepare_clusters_dir(method):
    if not os.path.exists(cluster_dir_ct): 
        os.mkdir(cluster_dir_ct)
    if not os.path.exists(cluster_dir_ct + method): 
        os.mkdir(cluster_dir_ct + method)
    else :
        for file in os.scandir(cluster_dir_ct + method):
            os.remove(file.path)
    
# Enregistre les messages bruts par cluster.
# utilise la variable globale cluster_dir_ct
def save_clusters_raw_msg(mlists, method):
    for i, mlist in enumerate(mlists):
        save_cluster_raw_msg(mlist, cluster_dir_ct + "{0}/{0}_cluster_{1}_raw_msg.txt".format(method, i))

# Enregistre les messages bruts d'un cluster dans un fichier
def save_cluster_raw_msg(mlist, file):
    with open(file, "w") as f:
        for msg in mlist:
            f.write(msg + '\n')
                
# Enregistre les messages encodés en 2 dimensions par cluster
def save_clusters_two_dim_enc_msg(e2dmlists, method):
    for i, e2dmlist in enumerate(e2dmlists):
        np.save(cluster_dir_ct + "{0}/{0}_cluster_{1}_enc_msg.npy".format(method, i), e2dmlist)    

# Enregistre les messages nettoyés par clusters
def save_clusters_clean_msg(cmlists, method):
    for i, cmlist in enumerate(cmlists):
        save_cluster_clean_msg(cmlist, cluster_dir_ct + "{0}/{0}_cluster_{1}_clean_msg.txt".format(method, i))

# Enregistre les messages nettoyés d'un cluster dans un fichier
def save_cluster_clean_msg(cmlist, file):
    with open(file, "w") as f:
        for msg in cmlist:
            f.write(" ".join(msg) + '\n')
    
# Enregistre les messages par clusters
def save_clusters_msg(cmlists, mlists, e2dmlists, method):
    prepare_clusters_dir(method)
    save_clusters_raw_msg(mlists, method)
    save_clusters_clean_msg(cmlists, method)
    save_clusters_two_dim_enc_msg(e2dmlists, method)

#######################################################     
#   Fonctions pour analyser les clusters en mémoire   #
#######################################################

# Analyse la prédiction et renvoie :
# La liste des mots par clusters, les listes des messages bruts, nettoyés et encodés en 2
# dimensions par clusters, la liste des mots les plus fréquents par clusters et la liste
# contenant le nombre et le pourcentage de messages contenant un mot-clé haineux par
# cluster
def parse(pred):
    res = build_res_dict(pred)
    wlists, mlists, cmlists, e2dmlists = create_word_and_msg_lists(res)
    mfw = most_freq_words(n_clust_info, wlists, ignored_ct)
    hkw = search_hate_words(mlists, kw_file)
    clust_n_msg = n_msg_by_clust(res)
    return res, wlists, mlists, cmlists, e2dmlists, mfw, hkw, clust_n_msg
    
# Analyse des résultats du clustering et sauvegarde des résultats
def parse_results(pred, method, centers=[]):
    res, wlists, mlists, cmlists, e2dmlists, mfw, hkw, clust_n_msg = parse(pred)
    print_clusters_info(clust_n_msg, mfw, hkw, centers)
    plot_results(pred, two_dim_dataset_ct)
    save_clusters_msg(cmlists, mlists, e2dmlists, method)
    save_clusters_info(clust_n_msg, mfw, hkw, method, centers)

# Analyse sans sauvegarde.
def parse_no_save(pred, centers=[]):
    res, wlists, mlists, cmlists, e2dmlists, mfw, hkw, clust_n_msg = parse(pred)
    print_clusters_info(clust_n_msg, mfw, hkw, centers)
    plot_results(pred, two_dim_dataset_ct)

# A FINIR
def parse_hate_clusters(pred, centers=[]):
    res = build_res_dict(pred)
    wlists, mlists, cmlists, e2dmlists = create_word_and_msg_lists(res)
    mfw = most_freq_words(n_clust_info, wlists, ignored_ct)
    hkw = search_hate_words(mlists, kw_file)
    clust_n_msg = n_msg_by_clust(res)
    print_hate_clusters_info(clust_n_msg, mfw, hkw, centers)
    #save_hate_clusters(clust_n_msg, mfw, hkw, )

                       
##########################################################################        
#   Fonctions pour analyser les clusters enregistrés dans des fichiers   #
##########################################################################

# Charge les messages d'un cluster dans une liste
def load_cluster(file):
    with open(file, "r") as f:
        clust_msgs = f.read().splitlines()
    return clust_msgs

# Renvoie le numéro de cluster dans une chaîne correspondant à un nom de fichier
# Permet de trier les noms de fichiers contenant les messages des
# clusters par ordre croissant
def extr_num(string):
    num = re.search("cluster_([0-9]+)_", string)
    return int(num.group(1)) if num else -1

# Charge en mémoire tous les clusters d'un répertoire.
# Renvoie deux liste : clusters nettoyés et bruts.
# dir est le nom du répertoire contenant les clusters, il doit 
# impérativement se terminer par /
def load_clusters(clusters_dir):
    dir_it = os.scandir(path=clusters_dir)
    names = [entry.name for entry in dir_it]
    names.sort(key=extr_num)
    clean_clusters = [load_cluster(clusters_dir + name) for name in names if ("clean" in name)]
    clean_clusters = [[msg.split() for msg in clust] for clust in clean_clusters]
    raw_clusters = [load_cluster(clusters_dir + name) for name in names if ("raw" in name)]
    two_dim_clusters = [np.load(clusters_dir + name) for name in names if ("enc" in name)]
    return clean_clusters, raw_clusters, two_dim_clusters
    
def parse_clusters(clean_clusters, raw_clusters, two_dim_clusters):
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