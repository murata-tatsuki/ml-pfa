import objectcondensation as oc

def cluster(event, prediction, tbeta=.2, td=.5, clustering_td_momentum=False):
    #print("== cluster ==")
    # print(f"{prediction.pred_betas=}")
    if not clustering_td_momentum:
        # clustering_filtered = oc.get_clustering_np(
        #     event=event, betas=prediction.pred_betas, X=prediction.pred_cluster_space_coords, charged_hits=prediction.charged_hits, tbeta=tbeta, td=td
        #     )+1
        clustering_filtered = oc.get_clustering_np_new(
            event=event, betas=prediction.pred_betas, X=prediction.pred_cluster_space_coords, charged_hits=prediction.charged_hits, tbeta=tbeta, td=td
            )+1
    else:
        clustering_filtered = oc.get_clustering_np_td_momentum(
            event=event, betas=prediction.pred_betas, X=prediction.pred_cluster_space_coords, charged_hits=prediction.charged_hits, tbeta=tbeta, td=td
            )+1
    #clustering = np.zeros(prediction.pass_noise_filter.shape) #Noise
    #clustering[np.flatnonzero(prediction.pass_noise_filter)] = clustering_filtered
    clustering = clustering_filtered
    # print(f"{clustering=}")
    return clustering
