import evaluation_noNoise as ev
import matplotlib.pyplot as plt
import numpy as np
import os
import math
#from importlib import reload
#import mplhep

def colorlabel(y,unique_arr):
    if y == unique_arr[0]:
        return "b"
    elif y == unique_arr[1] : return "g"
    if len(unique_arr)==3 :
        if y == unique_arr[2] : return "y"
    if len(unique_arr)==4 :
        if y == unique_arr[2] : return "y"
        elif y == unique_arr[3] : return "r"

def event_count(event,clustering,count):
    true_cluster_num = np.unique(event.y).size
    predicted_cluster_num = np.unique(clustering).size
    count[0].append(true_cluster_num)
    count[1].append(predicted_cluster_num)
    return count
    
def all_event_display(event,clustering):
    print("If you want to quit, enter 'y'.")
    x = input('>> ')
    if x == "y" : return "y"
    #y = np.load(f'npz_retagged/sample_slcio_{sys.argv[1]}_retagged.npz')
    #print(f"y : {y['recHitFeatures']}")
    x_point=event.x[:,1]
    y_point=event.x[:,2]
    z_point=event.x[:,3]
    fig = plt.figure(figsize = (8, 8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    label=clustering
    unique_label=np.unique(label)
    print(f"predict unique_label : {len(unique_label)}")
    l = 0
    for x1,y1,z1,label1 in zip(x_point,y_point,z_point,label):
        #if colorlabel(label1) == "b":               
        #print(f"label : {colorlabel(label1)}")
        ax1.scatter(x1, y1, z1,c=colorlabel(label1,unique_label))
                
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("predicted label")
            
    label=event.y
    unique_label=np.unique(label)
    l = 0
    for x1,y1,z1,label1 in zip(x_point,y_point,z_point,label):
        #if colorlabel(label1) == "b":
        #print(f"label : {colorlabel(label1)}")                
        ax2.scatter(x1, y1, z1,c=colorlabel(label1,unique_label))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("true label")
    plt.show()
    return -1

def count_hit_inCluster(event,clustering,acc):
    count = check_prediction(event,clustering)
    #assert count in range(len(count))
    #assert sorted(count[0].values(),reverse=True) in range(len(sorted(count[0].values(),reverse=True)))

    if len(count[0]) == 1 or len(count[1]) == 1 : return acc
    accuracy_1stcluster = sorted(count[1].values(),reverse=True)[0] / sorted(count[0].values(),reverse=True)[0]
    if len(sorted(count[1].values(),reverse=True))!=1 and len(sorted(count[0].values(),reverse=True)) !=1:       
        accuracy_2ndcluster = sorted(count[1].values(),reverse=True)[1] / sorted(count[0].values(),reverse=True)[1]
    else : event_display(event,clustering) 
    acc[0].append(accuracy_1stcluster)
    #acc[1].append(accuracy_2ndcluster)
    return acc

def check_prediction(event,clustering):
    count=[{},{}]
    type_hit_true = np.unique(event.y)
    type_hit_predicted = np.unique(clustering)
    for type_hit in type_hit_true:
        count[0][type_hit]=np.count_nonzero(event.y == type_hit) #count the number of hits of true label
    for type_hit in type_hit_predicted:
        count[1][type_hit]=np.count_nonzero(clustering == type_hit) #predicted label
    return count
        
def cal_energy_rate(event,clustering,energy_acc):
    count = check_prediction(event,clustering)
    if len(count[0]) == 1 or len(count[1]) == 1 :
        energy_acc[0].append(0)
        energy_acc[1].append(0)
        return energy_acc,0,0

    hitnum_1stcluster_predict = int(sorted(count[1].items(),reverse=True,key = lambda i:i[1])[0][0])
    hitnum_1stcluster_true = int(sorted(count[0].items(),reverse=True,key = lambda i:i[1])[0][0])
    hitnum_2ndcluster_predict = int(sorted(count[1].items(),reverse=True,key = lambda i:i[1])[1][0])
    hitnum_2ndcluster_true = int(sorted(count[0].items(),reverse=True,key = lambda i:i[1])[1][0])
    
    tag_id_1stcluster_predict = np.where(clustering==hitnum_1stcluster_predict)[0]
    tag_id_2ndcluster_predict = np.where(clustering==hitnum_2ndcluster_predict)[0]
    tag_id_1stcluster_true = np.where(event.y==hitnum_1stcluster_true)[0]
    tag_id_2ndcluster_true = np.where(event.y==hitnum_2ndcluster_true)[0]

    sum_energy_true_1st = np.sum(event.x[tag_id_1stcluster_true,0])
    sum_energy_predict_1st = np.sum(event.x[tag_id_1stcluster_predict,0])

    sum_energy_true_2nd = np.sum(event.x[tag_id_2ndcluster_true,0])
    sum_energy_predict_2nd = np.sum(event.x[tag_id_2ndcluster_predict,0])
    
    rate_energy_1st = sum_energy_predict_1st/sum_energy_true_1st
    rate_energy_2nd = sum_energy_predict_2nd/sum_energy_true_2nd

    energy_acc[0].append(rate_energy_1st)
    energy_acc[1].append(rate_energy_2nd)
    return energy_acc,rate_energy_1st,rate_energy_2nd

def plot_hist_float(list_hist,list_title,list_xlabel,list_ylabel):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(list_xlabel)
    ax.set_ylabel(list_ylabel)
    ax.set_title(list_title)
    ax.hist(list_hist,bins=30,range=(0.5,1.5))
    plt.show()
    

def plothist_twolabel(list1, list2,list1_title,list2_title,list1_xlabel,list2_xlabel,tag):
    if tag == "Event" : plt.ylabel("Number of Event")
    elif tag == "Cluster" : plt.ylabel("Number of Cluster")
    if type(list1[0]) == float : bins = int(math.floor(max(list1)))-int(math.floor(min(list1)))+1
    elif type(list1[0]) == int :bins = max(list1)-min(list1)+1
    plt.hist(list1,color="red",bins=bins,range=(min(list1)-.5,max(list1)+.5))
    plt.xlabel(list1_xlabel)
    plt.title(list1_title)
    plt.show()

    #bins = max(list2)-min(list2)+1
    if type(list2[0]) == float : bins = int(math.floor(max(list2)))-int(math.floor(min(list2)))+1
    elif type(list2[0]) == int :bins = max(list2)-min(list2)+1
    plt.hist(list2,color="blue",bins=bins,range=(min(list2)-.5,max(list2)+.5))
    plt.xlabel(list2_xlabel)
    plt.title(list2_title)
    plt.show()
    
def ilclabel(ax, text=r'$\itSimulation \; Preliminary$', x=.12, y=.88, dx=.12):
    ax.text(x, y, r'$\bfILD$', ha='left', va='bottom',transform=ax.figure.transFigure,fontsize=36)
    ax.text(x+dx, y, text,ha='left', va='bottom',transform=ax.figure.transFigure,fontsize=28)

def nhit_energy_acc(stats):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    #ax.hist(stats['eta_truth'])
    cat_name = {0: 'EM', 1: 'HAD', 2: 'MIP', 3: 'MIX'}

    ax.set_xlabel(r'$\Sigma E_{hit}^{pred} / \Sigma E_{hit}^{truth}$')
    ax.set_xlabel(r'$N_{hits}^{pred} / N_{hits}^{truth}$ (energy weighted)')
    ax.set_ylabel('A.U.')
    bins = np.linspace(0., 3., 50)
    
    for cat in range(4):
        sel = stats['category'] == cat
        print(f'{cat=}, n={sel.sum()}')
        epred_o_etruth = stats['esum_pred'][sel] / stats['esum_truth'][sel]
        ax.hist(epred_o_etruth, label=cat_name[cat], bins=bins, density=True, histtype=u'step', linewidth=2)

    ax.legend()
    #ilclabel(ax)

    plt.show()

def nhit_acc(stats):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    cat_name = {0: 'EM', 1: 'HAD', 2: 'MIP', 3: 'MIX'}

    ax.set_xlabel(r'$N_{hits}^{pred} / N_{hits}^{truth}$')
    ax.set_xlabel(r'$N_{hits}^{pred} / N_{hits}^{truth}$ (energy weighted)')
    ax.set_ylabel('A.U.')

    bins = np.linspace(0., 3., 50)

    for cat in range(4):
        sel = stats['category'] == cat
        nhitspred_o_nhitstruth = stats['nhits_pred'][sel] / stats['nhits_truth'][sel]
        ax.hist(nhitspred_o_nhitstruth, label=cat_name[cat], bins=bins, density=True, histtype=u'step', linewidth=2)
    ax.legend()
    #ilclabel(ax)

    plt.show()

def nhit_truth(stats):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    cat_name = {0: 'EM', 1: 'HAD', 2: 'MIP', 3: 'MIX'}

    ax.set_xlabel(r'$N_{hits}^{pred} / N_{hits}^{truth}$')
    ax.set_xlabel(r'$N_{hits}^{pred} / N_{hits}^{truth}$ (energy weighted)')
    ax.set_ylabel('A.U.')

    bins = np.linspace(0., 3., 50)

    for cat in range(4):
        sel = stats['category'] == cat
        nhitspred_o_nhitstruth = stats['nhits_truth'][sel]
        ax.hist(nhitspred_o_nhitstruth, label=cat_name[cat], bins=bins, density=True, histtype=u'step', linewidth=2)

    #ilclabel(ax)
    ax.legend()
    plt.show()

def plot_n_pred(stats):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    cat_name = {0: 'EM', 1: 'HAD', 2: 'MIP', 3: 'MIX'}

    ax.set_xlabel(r'$N_{hits}^{pred}$')
    ax.set_ylabel('Number of clusters')

    bins = np.linspace(0., 200., 50)
    nhithist=[]

    for cat in range(len(stats['nhits_truth'])):
        nhithist.append(stats['nhits_truth'][cat])
        #print(f"nhit : {nhithist}")

    ax.hist(nhithist,bins=bins,histtype=u"step",linewidth=2)
    #ilclabel(ax)
    #ax.legend()
    plt.show()

def plot_stats(stats_numpy):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    #ax.hist(stats['eta_truth'])
    cat_name = {0: 'EM', 1: 'HAD', 2: 'MIP', 3: 'MIX'}

    bins = np.linspace(0., 10., 50)
    
    #epred_o_etruth = stats['n_pred_id'][sel] / stats['n_truth_id'][sel]
    #epred_o_etruth = stats_numpy
    ax.hist(stats_numpy, bins=bins, linewidth=2)
    plt.show()
    
def plot_n_pred_in_truth(stats):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    #ax.hist(stats['eta_truth'])
    cat_name = {0: 'EM', 1: 'HAD', 2: 'MIP', 3: 'MIX'}

    ax.set_xlabel(r'$\Sigma E_{hit}^{pred} / \Sigma E_{hit}^{truth}$')
    ax.set_xlabel(r'$N_{hits}^{pred} / N_{hits}^{truth}$ (energy weighted)')
    ax.set_ylabel('A.U.')
    #bins = np.linspace(0., 1.2, 50)
    bins = np.linspace(0., 3., 50)
    
    #epred_o_etruth = stats['n_pred_id'][sel] / stats['n_truth_id'][sel]
    epred_o_etruth = stats['rate_pred']
    ax.hist(epred_o_etruth, bins=bins, linewidth=2)
    plt.show()


def get_stats(tbeta=.2, td=.5, nmax=4, yielder=None):
    stats = ev.Stats()
    if yielder is None:yielder = ev.TestYielder()
    count_cluster_inEvent = [[],[]]
    rate_accuracy = [[],[]]
    energy_rate_accuracy = [[],[]]
    check = True
    for event, prediction, clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        #count_cluster_inEvent=event_count(event,clustering,count_cluster_inEvent)
        #rate_accuracy=count_hit_inCluster(event,clustering,rate_accuracy)
        #energy_rate_accuracy,rate_1st,rate_2nd = cal_energy_rate(event,clustering,energy_rate_accuracy)
        #if check == True :
        #    print(f"matches : {matches}")
        #    if len(matches[1]) > 2:check = False
        stats.extend(ev.statistics_per_match(event,clustering,matches))
        stats.extend(ev.get_hit_matched_vs_unmatched(event,clustering,matches))
        stats.add('confmat',ev.signal_to_noise_confusion_matrix(event,clustering,norm=True))
    return stats

def event_display(event,clustering):
    x_point=event.x[:,1]
    y_point=event.x[:,2]
    z_point=event.x[:,3]
    fig = plt.figure(figsize = (8, 8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    label=clustering
    unique_label=np.unique(label)
    #print(f"predict unique_label : {len(unique_label)}")
    l = 0
    for x1,y1,z1,label1 in zip(x_point,y_point,z_point,label):
        #if colorlabel(label1) == "b":
        #print(f"label : {colorlabel(label1)}")
        #ax1.scatter(x1, y1, z1,c=colorlabel(label1,unique_label))
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("predicted label")

        label=event.y
        unique_label=np.unique(label)
        l = 0
    for x1,y1,z1,label1 in zip(x_point,y_point,z_point,label):
        #if colorlabel(label1) == "b":
        #print(f"label : {colorlabel(label1)}")
        ax2.scatter(x1, y1, z1,c=colorlabel(label1,unique_label))
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("true label")
        plt.show()

def show_Stats(event,clustering,matches,noise_index=0):
    stats = ev.Stats()
    stats.add('event_y',np.array(len(event.y)))
    return stats

#def event_count(event,matching):
def get_hit_matched_vs_unmatched_display(event, clustering, matches, noise_index=0):
    matched_truth = []
    n_matched_truth = []
    matched_pred = []
    id_2clusters = []
    for ip,(truth_ids, pred_ids) in enumerate(matches):
        matched_truth.extend(truth_ids)
        matched_pred.extend(pred_ids)

    stats = ev.Stats()
    if len(matches) ==2:
        stats.add('matched_truth',np.array(len(matched_truth)))
        stats.add('matched_pred',np.array(len(matched_pred)))
    for ip,(truth_ids_temp, pred_ids_temp) in enumerate(zip(matched_truth,matched_pred)):
        pred_all_ids = np.where(event.y == truth_ids_temp)
        n_pred_id = np.count_nonzero(clustering[pred_all_ids] == pred_ids_temp)
        n_truth_id = np.count_nonzero(event.y == truth_ids_temp)
        #rate_pred.append(n_pred_id/n_truth_id)                                                                                               
        if len(matches) == 2:stats.add('rate_pred',np.array(n_pred_id/n_truth_id))
    return stats

        
def scan_stats(tbeta=.2,td=.5,nmax=4,yielder=None):
    stats = ev.Stats()
    if yielder is None :yielder = ev.TestYielder()
    count_cluster_inEvent=[[],[]]
    rate_accuracy = [[],[]]
    energy_rate_accuracy=[[],[]]
    check = True
    for event,prediction,clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        if(len(event.y)<10):continue
        count_cluster_inEvent=event_count(event,clustering,count_cluster_inEvent)
        rate_accuracy=count_hit_inCluster(event,clustering,rate_accuracy)
        energy_rate_accuracy,rate_1st,rate_2nd=cal_energy_rate(event,clustering,energy_rate_accuracy)
        if rate_1st <0.1:
            event_display(event,clustering)

def main():
    #path="/home/tsumura/ILC/GravNet/pytorch_ILC_PFA/check_Grav/grav_check2.txt"
    #if os.path.isfile(path) :
    #    with open(path,'w') as f:
    #        f.write('')
    stats = get_stats(nmax=50)
    
    #check = True
    #for matched_pred,unmatched_pred in zip(stats['matched_pred_size'],stats['unmatched_pred_size']):
    #    for matched_pred in stats['matched_pred']:
    #        if check:
    #            print(f"matched pred :{matched_pred}")
    #            print(f"matched pred size :{matched_pred}")
    #            print(f"unmatched pred size :{unmatched_pred}")
    #            check = False
    #print(f"hit unmatch pred : {stats['nhits_unmatched_pred'][0]}")
    #ev.dump_stats('stats_%b%d',stats)
    #nhit_energy_acc(stats)
    #plot_stats(stats['matched_truth'])
    plot_n_pred_in_truth(stats)
    #nhit_truth(stats)

def main_2():
    stats=scan_stats(nmax=5)
    nhit_acc(stats)
    plot_stats(stats['matched_truth'])
    plot_n_pred_in_truth(stats)

    

if __name__=='__main__':
    main()
    
