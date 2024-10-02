import bisect
import numpy as np
from clustering import cluster
from event import Event

def make_matches(event, prediction, tbeta=.2, td=.5, clustering=None):
    if clustering is None: clustering = cluster(event, prediction, tbeta, td)
    i1s, i2s, _ = match(event.y[:,0], clustering, weights=event.energy)

    #j1s, j2s, _ = match(clustering, event.y[:,0], weights=event.energy)
    #assert( np.array_equal(i1s,j2s) )
    #assert( np.array_equal(i2s,j1s) )

    matches12,matches21 = group_matching2(i1s, i2s)
    return matches12,matches21


def match(clustering1, clustering2, weights=None, threshold=.2, noise_index=0):
    # current implementation is symmetric under exchange of clustering1 and clustering2
    # clustering1: MC particles
    # clustering2: reconstructed clusters

    # # np.set_printoptions(threshold=np.inf)
    # print("clustering1 ",type(clustering1), clustering1.shape, clustering1[0:50])
    # # print(type(clustering1), clustering1.shape)
    # print("clustering2 ",type(clustering2), clustering2.shape, clustering2[0:50])
    # # print(type(clustering2), clustering2.shape)

    DEBUG = False

    # weights: the reconstructed cluster energy is used here as the match weights
    if weights is None: weights = np.ones_like(clustering1)

    if DEBUG:
        print("== match ==")
        print(f"  clustering1 : {clustering1.shape}")
        #print(f"  clustering1 : {clustering1}")
        print(f"  clustering2 : {clustering2.shape}")
        #print(f"  clustering2 : {clustering2}")
        #print(f"{weights=}")
        ids1 = list(set(np.unique(clustering1)))
        ids2 = list(set(np.unique(clustering2)))
        print(f"  clustering1 : {ids1}")
        print(f"  clustering2 : {ids2}")


    #Numpy of cluster index for each event
    cluster_ids1, cluster_indices1 = np.unique(clustering1, return_inverse=True)
    cluster_ids2, cluster_indices2 = np.unique(clustering2, return_inverse=True)

    #Int number of cluster for each event
    n_clusters1 = cluster_ids1.shape[0]
    n_clusters2 = cluster_ids2.shape[0]

    # Pre-calculate all 'areas' for all clusters
    areas1 = {id : weights[clustering1 == id].sum().item() for id in cluster_ids1}
    areas2 = {id : weights[clustering2 == id].sum().item() for id in cluster_ids2}

    # Make all possible pairs of ids
    pairs = np.array(np.meshgrid(cluster_ids1,cluster_ids2)).T.reshape(-1,2)
    
    # Remove pairs with a noise index
    if noise_index is not None:
        pairs = pairs[~np.amax(pairs==noise_index, axis=-1).astype(bool)]
    
    # Calculate weighted intersection and "iom" (intersection over minimum)
    ioms = np.zeros(pairs.shape[0])
    intersections = np.zeros(pairs.shape[0])
    for i_pair, (id1, id2) in enumerate(pairs):
        intersection = weights[(clustering1==id1) & (clustering2==id2)].sum()
        minimum = min(areas1[id1], areas2[id2])
        ioms[i_pair] = intersection / minimum
        intersections[i_pair] = intersection

    # Sort lists by highest intersection
    order = np.argsort(intersections)[::-1]
    intersections = intersections[order]
    ioms = ioms[order]
    pairs = pairs[order]

    # Matching algo
    canhavemorematches_1 = set(cluster_ids1)
    canhavemorematches_2 = set(cluster_ids2)
    matched_1 = set()
    matched_2 = set()
    matches = []

    for iom, intersection, (i1, i2) in zip(ioms, intersections, pairs):

        if iom < threshold:
            #if DEBUG: print(f" -- skipped; iom {iom} is lower than threshold {threshold}")
            continue

        if DEBUG: print(f'[L]{i1} with [R]{i2}, {iom=:.2f}, {intersection=:.2f}')

        # comment out to allow many-to-many matchings
        #if i1 not in canhavemorema_tches_1 or i2 not in canhavemorematches_2:
        #    if DEBUG: print(
        #        f'  Matching [L]{i1} and [R]{i2} with {iom=:.2f}/{intersection=:.2f} cannot be made;'
        #        f' can have more matches: {i1}:{i1 in canhavemorematches_1}, {i2}:{i2 in canhavemorematches_2}'
        #        )
        #    continue

        if i1 in matched_1 and i2 in matched_2:
            if DEBUG: print(f'  Both [L]{i1} and [R]{i2} have >0 matches, cannot make this match')
            continue

        # Make the match
        matches.append([i1, i2, iom])

        if i1 in matched_1:
            i2s = [j2 for j1, j2, _ in matches if j1==i1]
            if DEBUG: print(f'  [L]{i1} (now) has >1 match; [R]{i2s} are not allowed more matches')
            canhavemorematches_2.difference_update(i2s)

        if i2 in matched_2:
            i1s = [j1 for j1, j2, _ in matches if j2==i2]
            if DEBUG: print(f'  [R]{i2} (now) has >1 match; [L]{i1s} are not allowed more matches')
            canhavemorematches_1.difference_update(i1s)

        matched_1.add(i1)
        matched_2.add(i2)

    if len(matches) == 0:
        print('Warning: No matches at all')
        return [], [], []

    matches = np.array(matches)
    i1s, i2s, ioms = matches[:,0].astype(np.int32), matches[:,1].astype(np.int32), matches[:,2]

    return i1s, i2s, ioms

def group_matching(i1s, i2s):
    ''' Make groups of matches.
        It assumes two input lists in which the matching is defined by values in the same position.
        The function considers 1-to-1, 1-to-many, and many-to-1 mappings.
        #If a many-to-many match is detected, an exception is raised.
        The output is a list of 1-to-1, 1-to-many, and many-to-1 mappings.
    '''
    assert(len(i1s) == len(i2s))
    match_dict_1_to_2 = {}
    match_dict_2_to_1 = {}

    for i1, i2 in zip(i1s, i2s):
        m1 = i1 in match_dict_1_to_2
        m2 = i2 in match_dict_2_to_1
        mm1 = any( i1 in v for v in match_dict_2_to_1.values() )
        mm2 = any( i2 in v for v in match_dict_1_to_2.values() )

        # if ( (not m1 and mm1) or (not m2 and mm2) ):
        #     raise Exception(
        #         f'Detected many-to-many match:'
        #         f' [L]{i1} and [R]{i2} are both already matched to something else'
        #     )
        
        if (not m1 and not m2):
            match_dict_1_to_2[i1] = [i2]
            match_dict_2_to_1[i2] = [i1]
        elif (m1):
            if (len(match_dict_1_to_2[i1]) == 1):
                del match_dict_2_to_1[ match_dict_1_to_2[i1][0] ]
            bisect.insort(match_dict_1_to_2[i1],i2)
        elif (m2):
            if (len(match_dict_2_to_1[i2]) == 1):
                del match_dict_1_to_2[ match_dict_2_to_1[i2][0] ]
            bisect.insort(match_dict_2_to_1[i2],i1)
        else:
            raise Exception(
                f'Detected many-to-many match:'
                f' [L]{i1} and [R]{i2} are both already matched to something else'
            )
    
    matches = [[[k], v] for k, v in match_dict_1_to_2.items()]
    matches.extend([[v, [k]] for k, v in match_dict_2_to_1.items() if len(v)>1])
    return matches

def group_matching2(i1s, i2s):
    ''' Make groups of matches.
        Like group_matching() but also allow many-to-many matches.
    '''
    assert(len(i1s) == len(i2s))
    match_dict_1_to_2 = {}
    match_dict_2_to_1 = {}

    for i1, i2 in zip(i1s, i2s):
        m1 = i1 in match_dict_1_to_2
        m2 = i2 in match_dict_2_to_1

        if (not m1):
            match_dict_1_to_2[i1] = [i2]
        else:
            bisect.insort(match_dict_1_to_2[i1],i2)

        if (not m2):
            match_dict_2_to_1[i2] = [i1]
        else:
            bisect.insort(match_dict_2_to_1[i2],i1)

    #matches12 = [[k, v] for k, v in match_dict_1_to_2.items()]
    #matches21 = [[k, v] for k, v in match_dict_2_to_1.items()]
    #return matches12,matches21
    return match_dict_1_to_2,match_dict_2_to_1


def get_mask_charged_neutral(event: Event , clustering, matches, noise_index=0):
    matched_truth = []
    matched_pred = []
    truth_ids, pred_ids = matches
    matched_truth.extend(truth_ids)
    matched_pred.extend(pred_ids)
    # for truth_ids, pred_ids in matches:
    #     print(truth_ids, pred_ids)
    #     matched_truth.extend(truth_ids)
    #     matched_pred.extend(pred_ids)
    all_truth_ids = set(np.unique(event.y[:,0]))
    all_pred_ids = set(np.unique(clustering))
    all_truth_ids.discard(noise_index)
    all_pred_ids.discard(noise_index)
    truth_charged_index = event.y[:,0][event.y[:,1]==1]
    pred_charged_index = clustering[event.y[:,1]==1]
    all_truth_ids_charged = set(np.unique(truth_charged_index))
    all_pred_ids_charged = set(np.unique(pred_charged_index))
    true_charged_mask = np.isin(event.y[:,0], list(all_truth_ids_charged))
    pred_charged_mask = np.isin(clustering, list(all_pred_ids_charged))

    debug = False
    if ( debug ):
        print(f"{len(clustering)=}")
        print(f"{clustering=}")
        print(f"{len(event.y[:,0])=}")
        print(f"{event.y[:,0]=}")
        print(f"{event.y[:,1]=}")
        print(f"{truth_charged_index=}")
        print(f"{pred_charged_index=}")
        print(f"{true_charged_mask.astype('int')=}")
        print(f"{pred_charged_mask.astype('int')=}")

    return true_charged_mask, pred_charged_mask


def get_ABCD(z : np.ndarray, mask1 : np.ndarray, mask2 : np.ndarray):
    zA = z[mask1 & mask2]
    zB = z[mask1 & ~mask2]
    zC = z[~mask1 & mask2]
    zD = z[~mask1 & ~mask2]
    return zA.sum(), zB.sum(), zC.sum(), zD.sum()


def get_energy_ABCD(event: Event, true_charged_mask, pred_charged_mask):
    
    # x[:,0] = self.shaper_tanh(x[:, 0]-0.01,1.0,1.0,0.0,0.0)
    # def shaper_tanh(self,x, a=1.0, b=1.0, c=0.0, d=0.0):
    #     return a*np.tanh(b*(x-c))+d

    def inverse_shaper_tanh(y,a=1.0,b=1.0,c=0.0,d=0.0):
        return (np.arctanh((y-d)/a))/b+c
    real_energy = inverse_shaper_tanh(event.energy)+0.01

    A,B,C,D = get_ABCD(real_energy, true_charged_mask, pred_charged_mask)
    debug = False
    if ( debug ):
        print(f"{A=}, {B=}, {C=}, {D=}")
    return A,B,C,D