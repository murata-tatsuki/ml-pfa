import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
#from tools.readtext import ReadText
#from dataset import ilc_dataset
#from torch_cmspepr.gravnet_model import GravnetModel,GravnetModelWithNoiseFilter
#import evaluation_noNoise as ev
#import event_view_plot as plt_3d
#from display.display_h5 import single_pdata_to_file
#from plot_statistics_Double_res import get_stats,nhit_acc,nhit_energy_acc,plot_n_pred_in_truth
#from plot_statistics_Double import show_accNumber
from test_yielder import TestYielder
from model import get_model
from stats import Stats
from objectcondensation import get_condpoints


def scan_stats(tbeta=.2,td=.5,nmax=4,yielder=None,useTraining=False):
    stats = Stats()
    if yielder is None :yielder = TestYielder(useTraining=useTraining)
    count_cluster_inEvent=[[],[]]
    rate_accuracy = [[],[]]
    energy_rate_accuracy=[[],[]]
    check = True
    for event,prediction,clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        plt_3d.event_display(event,clustering)
        plt.savefig("tmp.png")
        break
        #if(len(event.y)<10):continue
        #count_cluster_inEvent=event_count(event,clustering,count_cluster_inEvent)
        #rate_accuracy=count_hit_inCluster(event,clustering,rate_accuracy)
        #energy_rate_accuracy,rate_1st,rate_2nd=cal_energy_rate(event,clustering,energy_rate_accuracy)
        #if rate_1st <0.1:
        #    plt_3d.event_display(event,clustering)


def test_event_display():
    stats = Stats()
    yielder = TestYielder()
    count=0
    for event,prediction,clustering,matches in yielder.iter_matches(tbeta=.2,td=.5,nmax=200):
        count += 1
        plt_3d.event_display(event,clustering)
        plt.savefig(f"display{count:02d}.png")
        if count==20: break


def plot_coords(nmax = 10):
    model = get_model(jit=False,input_dim=5)
    model.eval()
    tbeta=0.2
    td=0.5
    yielder = TestYielder(model=model)
    count = 0

    for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax)):
        #print(f"{event.y=}")
        count += 1
        label=event.y[:,0]
        unique_label=np.unique(label)
        betas = prediction.pred_betas
        space_coords = prediction.pred_cluster_space_coords
        num_beta_cut = len(betas[betas>tbeta])
        out = model(torch.from_numpy(event.x).cpu(), torch.from_numpy(event.batch).cpu())
        betas = torch.sigmoid(out[:,0])
        coords = out[:,1:3]
        fig = plt.figure(facecolor="white",figsize=[10,10])
        ax = fig.gca()
        for ilabel in unique_label:
            xy = coords[label==ilabel].detach().cpu()
            ax.scatter(xy[:,0],xy[:,1],s=30,alpha=0.5)
            ax.text(xy[0,0],xy[0,1],f"{ilabel}")
        ax.text(0.,0.,f"betas: {num_beta_cut}/{len(betas)}")
        plt.show()
        fig.savefig(f"GravPNG/{count:05d}_coords.png")


def plot_coords_pred_clusters(nmax = 10):
    model = get_model(jit=False,input_dim=5)
    model.eval()
    tbeta=0.2
    td=0.5
    yielder = TestYielder(model=model)
    count = 0

    #for event, prediction in yielder.iter_pred(nmax):
    
    for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax)):
        #print(f"{event.y=}")
        count += 1
        label=clustering
        unique_label=np.unique(label)
        betas = prediction.pred_betas
        space_coords = prediction.pred_cluster_space_coords
        num_beta_cut = len(betas[betas>tbeta])
        out = model(torch.from_numpy(event.x).cpu(), torch.from_numpy(event.batch).cpu())
        betas = torch.sigmoid(out[:,0])
        coords = out[:,1:3]
        fig = plt.figure(facecolor="white",figsize=[10,10])
        ax = fig.gca()
        for ilabel in unique_label:
            xy = coords[label==ilabel].detach().cpu()
            ax.scatter(xy[:,0],xy[:,1],s=30,alpha=0.5)
            ax.text(xy[0,0],xy[0,1],f"{ilabel}")
        ax.text(0.,0.,f"betas: {num_beta_cut}/{len(betas)}")
        plt.show()
        fig.savefig(f"GravPNG/{count:05d}_coords_predclus.png")



def test_matched(tbeta=.2,td=.5,nmax=20,yielder=None,useTraining=False,timingCut=False):
    stats = Stats()
    if yielder is None :yielder = TestYielder(useTraining=useTraining, timingCut=timingCut)
    for event,prediction,clustering,matches in yielder.iter_matches(tbeta,td,nmax):
        stats.extend(ev.get_matched_vs_unmatched_charged_neutral(event,clustering,matches))
    return stats


def plot_realcoords(nmax = 10):
    model = get_model(jit=False,input_dim=5)
    model.eval()
    tbeta=0.2
    td=0.5
    yielder = TestYielder(model=model)
    count = 0
    scale=2000

    for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax)):
        count += 1
        label=event.y[:,0]
        unique_label=np.unique(label)
        betas = prediction.pred_betas
        num_beta_cut = len(betas[betas>tbeta])
        out = model(torch.from_numpy(event.x).cpu(), torch.from_numpy(event.batch).cpu())
        betas = torch.sigmoid(out[:,0])
        fig = plt.figure(facecolor="white",figsize=[10,10])
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.gca()
        for ilabel in unique_label:
            x=event.x[label==ilabel]
            ax.scatter(x[:,1]*scale,x[:,2]*scale,x[:,3]*scale,s=30,alpha=0.5)
            ax.text(x[0,1]*scale,x[0,2]*scale,x[0,3]*scale,f"{ilabel}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            #ax.view_init(elev=0,azim=-45,roll=30)
        #ax.text(0.85,0.85,f"betas: {num_beta_cut}/{len(betas)}")
        plt.show()
        fig.savefig(f"GravPNG/{count:05d}_realcoords.png")


def plot_coords_combined(nmax = 10):
    model = get_model(jit=False,input_dim=5)
    model.eval()
    tbeta=0.2
    td=0.5
    yielder = TestYielder(model=model)
    count = 0
    scale=2000

    for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(nmax=nmax)):
        count += 1
        label=event.y[:,0]
        unique_label=np.unique(label)
        unique_clusters=np.unique(clustering)
        betas = prediction.pred_betas
        num_beta_cut = len(betas[betas>tbeta])
        out = model(torch.from_numpy(event.x).cpu(), torch.from_numpy(event.batch).cpu())
        betas = torch.sigmoid(out[:,0])
        fig = plt.figure(facecolor="white",figsize=[30,10])

        ax = fig.add_subplot(1,3,1,projection='3d')
        for ilabel in unique_label:
            x=event.x[label==ilabel]
            ax.set_title(f"MC Clusters")
            ax.scatter(x[:,1]*scale,x[:,2]*scale,x[:,3]*scale,s=30,alpha=0.5)
            ax.text(x[0,1]*scale,x[0,2]*scale,x[0,3]*scale,f"{ilabel}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        coords = out[:,1:3]
        ax = fig.add_subplot(1,3,2)
        for ilabel in unique_label:
            xy = coords[label==ilabel].detach().cpu()
            ax.set_title(f"MC Clusters")
            ax.scatter(xy[:,0],xy[:,1],s=30,alpha=0.5)
            ax.text(xy[0,0],xy[0,1],f"{ilabel}")

        # plot condensation points
        # xy2 = coords[betas>tbeta].detach().cpu()
        # ax.scatter(xy2[:,0],xy2[:,1],s=50,alpha=0.5,marker="x",color="black")


        ax = fig.add_subplot(1,3,3)
        for ilabel in unique_clusters:
            xy = coords[clustering==ilabel].detach().cpu()
            ax.set_title(f"Predicted Clusters")
            ax.scatter(xy[:,0],xy[:,1],s=30,alpha=0.5)
            ax.text(xy[0,0],xy[0,1],f"{ilabel}")

        plt.show()

        fig.savefig(f"GravPNG/{count:05d}_coords_combined.png")
        plt.close()


def plot_stats(stats_numpy,hist_min,hist_max,key_name,num):
    fig = plt.figure(facecolor="white",figsize=[9,5])
    ax = fig.gca()
    #ax.hist(stats['eta_truth'])
    bins = np.linspace(hist_min, hist_max, 50)
    #epred_o_etruth = stats['n_pred_id'][sel] / stats['n_truth_id'][sel]
    ax.hist(stats_numpy, bins=bins, linewidth=2)
    average_hist = np.round(np.nanmean(stats_numpy),decimals=5)
    #average_hist = np.mean(stats_numpy)
    ax.text(0.85,0.85,f"average : {average_hist}")
    plt.show()
    fig.savefig(f"GravPNG/{key_name}_{hist_min}_{hist_max}_{num}.png")

def dump_betas(useTraining = False):
    print(f"dump_betas")
    #from sklearn.preprocessing import MinMaxScaler
    yielder=TestYielder(useTraining=useTraining)
    nmax=20
    tbeta=0.2
    td=0.5
    for event, prediction in yielder.iter_pred(nmax):

        label=event.y
        unique_label=np.unique(label)
        print(f"true clusters = {len(unique_label)}")

        betas = prediction.pred_betas
        num_beta_cut = len(betas[betas>tbeta])
        print(f"betas: {num_beta_cut} / {len(betas)}")
        #print(f"{betas=}")
        #scaler = MinMaxScaler()
        #betas = betas.reshape(-1, 1)
        #betas_scaled = scaler.fit_transform(betas).flatten()
        #print(f"{betas_scaled=}")
        #betaSel = prediction.pred_betas[ prediction.pred_betas > 0.3 ]
        #print(f"{betaSel=}")
        #clustering = ev.cluster(prediction, tbeta, td)


def main():
    #debug_model()
    #dump_betas(useTraining=True)
    #print(model.state_dict()[list(model.state_dict().keys())[12]])
    #test_event_display()
    #scan_stats()
    #stats = get_stats(nmax=20000)
    #stats = get_stats(nmax=200)
    #nhit_acc(stats)
    #plt.savefig("nhit_acc.png")
    #nhit_energy_acc(stats)
    #plt.savefig("nhit_energy_acc.png")
    #plot_n_pred_in_truth(stats)
    #plt.savefig("npred_in_truth.png")
    #plot_stats(stats['num_pred'],0,1,"num_pred",1)
    #plt.savefig("num_pred.png")
    #plot_stats(stats['rate_pred'],0,1,"rate_pred",1)
    #plt.savefig("rate_pred.png")
    #plot_stats(stats['num_pred_energy'],0,1,"num_pred_energy",1)
    #plt.savefig("num_pred_energy.png")
    #plot_stats(stats['rate_pred_energy'],0.99,1,"rate_pred_energy",1)
    #plt.savefig("rate_pred_energy.png")

    plot_coords_combined(100)
    #plot_coords_combined(1)

    #show_accNumber()

    #stats = test_matched(nmax=2000,useTraining=False)
    #stats = test_matched(nmax=20,useTraining=True)
    #stats = test_matched(nmax=20,useTraining=False)
    #plot_stats(stats['pred_charged_over_all_true_charged'],0,1,"pred_charged_over_all_true_charged","test")
    #plot_stats(stats['true_charged_over_all_pred_charged'],0,1,"true_charged_over_all_pred_charged","test")
    #plot_stats(stats['pred_neutral_over_all_true_neutral'],0,1,"pred_neutral_over_all_true_neutral","test")
    #plot_stats(stats['true_neutral_over_all_pred_neutral'],0,1,"true_neutral_over_all_pred_neutral","test")

    # stats = test_matched(nmax=2000,useTraining=True)
    # plot_stats(stats['pred_charged_over_all_true_charged'],0,1,"pred_charged_over_all_true_charged","train")
    # plot_stats(stats['true_charged_over_all_pred_charged'],0,1,"true_charged_over_all_pred_charged","train")
    # plot_stats(stats['pred_neutral_over_all_true_neutral'],0,1,"pred_neutral_over_all_true_neutral","train")
    # plot_stats(stats['true_neutral_over_all_pred_neutral'],0,1,"true_neutral_over_all_pred_neutral","train")
    pass


if __name__=="__main__":
    main()
