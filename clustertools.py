import numpy as np
import matplotlib.pyplot as plt
import os

# Nombre de mots les plus fréquents donnés dans les infos
n_most_freq_ignored = 40
n_clust_info = 20
kw_file = "kw_hate.txt"

def init_globals(cd, rd, ed, tdd, cld):
    global clean_sample
    clean_sample = cd
    global raw_sample
    raw_sample = rd
    global dataset_encoded
    dataset_encoded = ed
    global two_dim_dataset
    two_dim_dataset = tdd
    global cluster_dir
    cluster_dir = cld

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
#utilise les variables globales clean_sample et raw_sample
def create_word_and_msg_lists(res):
    wlists = []
    mlists = []
    emlists = []
    cmlists = []
    # Parcourt les clés (donc les clusters) dans l'ordre croissant
    for i in sorted(res):
        idx_list = []
        clean_msg_list = []
        raw_msg_list = []
        encoded_msg_list = []
        # Création de la liste des indices pour ce cluster
        for n in res[i]: idx_list.append(n)
        # Création des listes de messages
        for idx in idx_list: 
            clean_msg_list.append(clean_sample[idx])
            raw_msg_list.append(raw_sample[idx])
            encoded_msg_list.append(dataset_encoded[idx])
        mlists.append(raw_msg_list)
        emlists.append(encoded_msg_list)
        wlists.append(create_sorted_wlist(clean_msg_list))
        cmlists.append(clean_msg_list)                        
    return wlists, mlists, emlists, cmlists

# Détermine les mots les plus fréquents de chaque cluster
# Renvoie un dictionnaire qui associe à chaque cluster la liste des 
# n mots les plus fréquents
def most_freq_words(n, wlists, ignored):
    ret = {}
    for i, tuplist in enumerate(wlists):
        j = 0
        k = 0
        ret[i] = []
        while (j < n and k < len(tuplist)):
            word = tuplist[k][0]
            if word not in ignored:
                ret[i].append(word)
                j += 1
            k += 1
            
        #ret[i] = []
        #for tup in tuplist[:n]:
        #    ret[i].append(tup[0])
        #ret[i] = [tup[0] for tup in tuplist[:n]]
    return ret

# Recherche les mots-clé haineux dans les messages de chaque cluster.
# Renvoie un dictionnaire qui associe à chaque cluster le nombre et le 
# pourcentage de messages contenant un mot-clé haineux.
def search_hate_words(mlists, file):
    ret = {}
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
        ret[i] = (n_msg, perc)
    return ret

# Renvoie une liste qui contient le nombre de messages de chaque cluster
def n_msg_by_clust(res):
    return [len(res[i]) for i in sorted(res)]


##### Fonctions pour analyser les résultats du clustering en mémoire

def print_clusters_info(n_msg, mfw, hkw, centers=[]):
    for i in range(len(n_msg)):
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
        
# Imprime les informations pour chaque cluster
# def print_clusters_info(n, res, wlists, centers=[]):
#     for i, key in enumerate(sorted(res)):
#         print("cluster {} : {} messages".format(key, len(res[key])))
#         print("Les {} mots les plus fréquents :".format(n), end=' ')
#         for tup in wlists[i][:n]:
#             print(tup[0], end=' ')
#         print()
#         if len(centers) > 0:
#             print("Les mots les plus proches du centre :", end = ' ')
#             for w in centers[i]:
#                 print(w, end=' ')
#             print('\n')
            
# Enregistre un résumé des informations pour chaque cluster
def save_clusters_info(n, res, wlists, method, centers=[]):
    f = open(cluster_dir + "{0}/{0}_clusters_info.txt".format(method), "w")
    for i, key in enumerate(sorted(res)):
        f.write("cluster {} : {} messages\n".format(key, len(res[key])))
        f.write("Les {} mots les plus fréquents : ".format(n))
        for tup in wlists[i][:n]:
            f.write(tup[0] + " ")
        f.write('\n')
        if len(centers) > 0:
            f.write("Les mots les plus proches du centre : ")
            for w in centers[i]:
                f.write(w + " ")
            f.write('\n')
        f.write('\n')
    f.close()
        
# Imprime un échantillon de messages pour chaque cluster
def print_sample(sample_size, res):
    for i, key in enumerate(sorted(res)):
        print("cluster {} (clé {})".format(i, key))
        try:
            sample = random.sample(res[key], sample_size)
        except ValueError:
            sample = res[key]
        for j in sample:
            print(raw_sample[j])
        print()

# Crée le graphique des résultats
def plot_results(res):
    styles = ["r.", "b.", "k.", "g.", "y.", "c.", "m."]
    i = 0
    j = 0
    for key in res.keys():
        cluster_data = []
        for idx in res[key]:
            cluster_data.append(two_dim_dataset[idx])
        tcd = np.array(cluster_data).T
        if (j > 6): j = 0
        plt.plot(tcd[0], tcd[1], styles[j])
        i += 1
        j += 1
    plt.show()

# Vérifie l'existence des dossiers de clusters et les crée si besoin.
def prepare_clusters_dir(method):
    if not os.path.exists(cluster_dir): 
        os.mkdir(cluster_dir)
    if not os.path.exists(cluster_dir + method): 
        os.mkdir(cluster_dir + method)
    else :
        for file in os.scandir(cluster_dir + method):
            os.remove(file.path)
    
# Enregistre les messages bruts par cluster.
def save_clusters_raw_msg(mlists, method):
    for i, mlist in enumerate(mlists):
        with open(cluster_dir + "{0}/{0}_cluster_{1}_raw_msg.txt".format(method, i), "w") as f:
            for msg in mlist:
                f.write(msg)
                
#Enregistre les messages encodés par cluster.
def save_clusters_encoded_msg(emlists, method):
    for i, emlist in enumerate(emlists):
        np.save(cluster_dir + "{0}/{0}_cluster_{1}_enc_msg.npy".format(method, i), emlist)
#        with open(cluster_dir + "{0}/{0}_cluster_{1}_enc_msg.txt".format(method, i), "w") as f:
#            for idx in res[i]:
#                for num in dataset_encoded[idx]:
#                    f.write(str(num) + " ")

# Enregistre les messages nettoyés par clusters
def save_clusters_clean_msg(cmlists, method):
    for i, cmlist in enumerate(cmlists):
        with open(cluster_dir + "{0}/{0}_cluster_{1}_clean_msg.txt".format(method, i), "w") as f:
            for msg in cmlist:
                f.write(" ".join(msg) + '\n')
    
# Enregistre les messages par clusters
def save_clusters_msg(cmlists, mlists, method):
    prepare_clusters_dir(method)
    save_clusters_raw_msg(mlists, method)
    #save_clusters_encoded_msg(emlists, method)
    save_clusters_clean_msg(cmlists, method)
        
# Analyse des résultats du clustering et sauvegarde des résultats
def parse_results(pred, method, centers=[]):
    res = build_res_dict(pred)
    wlists, mlists, emlists, cmlists = create_word_and_msg_lists(res)
    print_clusters_info(n_clust_info, res, wlists, centers)
    plot_results(res)
    save_clusters_msg(cmlists, mlists, method)
    save_clusters_info(n_clust_info, res, wlists, method, centers)

# Analyse sans sauvegarde.
def parse_no_save(pred, centers=[]):
    res = build_res_dict(pred)
    wlists, mlists, emlists, cmlists = create_word_and_msg_lists(res)
    ignored = [tup[0] for tup in create_sorted_wlist(clean_sample)[:n_most_freq_ignored]]
    mfw = most_freq_words(n_clust_info, wlists, ignored)
    hwk = search_hate_words(mlists, kw_file)
    clust_n_msg = n_msg_by_clust(res)
    print_clusters_info(clust_n_msg, mfw, hwk, centers)


        
        
#### Fonctions pour analyser les clusters enregistrés dans des fichiers 

def load_cluster(file):
    with open(file, "r") as f:
        clust_msgs = f.read().splitlines()
    return clust_msgs