import numpy as np
import matplotlib.pyplot as plt
import os

n_clust_info = 20

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
    wlist = [ (mot, dico[mot]) for mot in dico.keys()]
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
    
#construire la liste des mots et la liste des messages de chaque cluster
#utilise les variables globales clean_sample et raw_sample
def create_word_and_msg_lists(res):
    wlists = []
    mlists = []
    emlists = []
    cmlists = []
    #parcourt les clés dans l'ordre croissant
    for i in sorted(res):
        idx_list = []
        clean_msg_list = []
        raw_msg_list = []
        encoded_msg_list = []
        for n in res[i]: idx_list.append(n)
        for idx in idx_list: 
            clean_msg_list.append(clean_sample[idx])
            raw_msg_list.append(raw_sample[idx])
            encoded_msg_list.append(dataset_encoded[idx])
        mlists.append(raw_msg_list)
        emlists.append(encoded_msg_list)
        wlists.append(create_sorted_wlist(clean_msg_list))
        cmlists.append(clean_msg_list)                        
    return wlists, mlists, emlists, cmlists

def print_clusters_info(n, res, wlists, centers=[]):
    for i, key in enumerate(sorted(res)):
        print("cluster {} : {} messages".format(key, len(res[key])))
        print("Les {} mots les plus fréquents :".format(n), end=' ')
        for tup in wlists[i][:n]:
            print(tup[0], end=' ')
        print()
        if len(centers) > 0:
            print("Les mots les plus proches du centre :", end = ' ')
            for w in centers[i]:
                print(w, end=' ')
            print('\n')
            
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

def prepare_clusters_dir(method):
    if not os.path.exists(cluster_dir): 
        os.mkdir(cluster_dir)
    if not os.path.exists(cluster_dir + method): 
        os.mkdir(cluster_dir + method)
    else :
        for file in os.scandir(cluster_dir + method):
            os.remove(file.path)
    
def save_clusters_raw_msg(mlists, method):
    for i, mlist in enumerate(mlists):
        with open(cluster_dir + "{0}/{0}_cluster_{1}_raw_msg.txt".format(method, i), "w") as f:
            for msg in mlist:
                f.write(msg)
                
def save_clusters_encoded_msg(emlists, method):
    for i, emlist in enumerate(emlists):
        np.save(cluster_dir + "{0}/{0}_cluster_{1}_enc_msg.npy".format(method, i), emlist)
#        with open(cluster_dir + "{0}/{0}_cluster_{1}_enc_msg.txt".format(method, i), "w") as f:
#            for idx in res[i]:
#                for num in dataset_encoded[idx]:
#                    f.write(str(num) + " ")

def save_clusters_clean_msg(cmlists, method):
    for i, cmlist in enumerate(cmlists):
        with open(cluster_dir + "{0}/{0}_cluster_{1}_clean_msg.txt".format(method, i), "w") as f:
            for msg in cmlist:
                f.write(" ".join(msg) + '\n')
    
def save_clusters_msg(cmlists, mlists, method):
    prepare_clusters_dir(method)
    save_clusters_raw_msg(mlists, method)
    #save_clusters_encoded_msg(emlists, method)
    save_clusters_clean_msg(cmlists, method)
        
def parse_results(pred, method, centers=[]):
    res = build_res_dict(pred)
    wlists, mlists, emlists, cmlists = create_word_and_msg_lists(res)
    print_clusters_info(n_clust_info, res, wlists, centers)
    plot_results(res)
    save_clusters_msg(cmlists, mlists, method)
    save_clusters_info(n_clust_info, res, wlists, method, centers)
