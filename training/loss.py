"""Object condensation loss and model-output parsing (shared by single-GPU and DDP)."""

from __future__ import annotations

import torch

import objectcondensation as oc


def loss_fn(
    out,
    data,
    args,
    qmin,
    loss_offset,
    i_epoch=None,
    return_components=False,
    use_charge_track_likeness=False,
):
    device = out.device

    pred_betas = torch.sigmoid(out[:, 0])
    pred_charge_track_likeness = None
    pred_tracker_energy = None
    pred_cluster_energy = None
    weight_photon = None
    weight_charged_hadron = None
    weight_neutral_hadron = None
    weight_muon = None
    weight_electron = None
    if args.energy_regression_weight:
        if args.energy_regression and not args.energy_regression_cluster:
            pred_tracker_energy = out[:, 1]
            weight_photon = out[:, 2]
            weight_charged_hadron = out[:, 3]
            weight_neutral_hadron = out[:, 4]
            weight_muon = out[:, 5]
            weight_electron = out[:, 6]
            pred_cluster_space_coords = out[:, 7:]
        elif args.energy_regression and args.energy_regression_cluster:
            pred_tracker_energy = out[:, 1]
            pred_cluster_energy = out[:, 2]
            weight_photon = out[:, 3]
            weight_charged_hadron = out[:, 4]
            weight_neutral_hadron = out[:, 5]
            weight_muon = out[:, 6]
            weight_electron = out[:, 7]
            pred_cluster_space_coords = out[:, 8:]
        elif not args.energy_regression:
            weight_photon = out[:, 1]
            weight_charged_hadron = out[:, 2]
            weight_neutral_hadron = out[:, 3]
            weight_muon = out[:, 4]
            weight_electron = out[:, 5]
            pred_cluster_space_coords = out[:, 6:]
    else:
        if args.energy_regression:
            if not args.energy_regression_cluster:
                if use_charge_track_likeness:
                    pred_charge_track_likeness = torch.sigmoid(out[:, 1])
                    pred_tracker_energy = out[:, 2]
                    pred_cluster_space_coords = out[:, 3:]
                    assert pred_charge_track_likeness.device == device
                else:
                    pred_tracker_energy = out[:, 1]
                    pred_cluster_space_coords = out[:, 2:]
            else:
                if use_charge_track_likeness:
                    pred_charge_track_likeness = torch.sigmoid(out[:, 1])
                    pred_tracker_energy = out[:, 2]
                    pred_cluster_energy = out[:, 3]
                    pred_cluster_space_coords = out[:, 4:]
                    assert pred_charge_track_likeness.device == device
                else:
                    pred_tracker_energy = out[:, 1]
                    pred_cluster_energy = out[:, 2]
                    pred_cluster_space_coords = out[:, 3:]
        else:
            if use_charge_track_likeness:
                pred_charge_track_likeness = torch.sigmoid(out[:, 1])
                pred_cluster_space_coords = out[:, 2:]
                assert pred_charge_track_likeness.device == device
            else:
                pred_cluster_space_coords = out[:, 1:]
    cluster_track_index = data.y[:, 1]

    assert all(
        t.device == device
        for t in [
            pred_betas,
            pred_cluster_space_coords,
            data.y,
            data.batch,
        ]
    )
    true_energy = torch.sqrt(torch.sum(torch.square(data.label[:, 4:8]), 1))
    detected_energy = data.feat[:, 0]
    LE_weight = (
        0
        if (i_epoch <= args.epochs_noLE)
        else (
            1
            if (i_epoch > args.epochs_noLE + 10)
            else pow((i_epoch - args.epochs_noLE), 2) / 100.0
        )
    )
    er_coef = (
        args.regression_coefficinet * LE_weight
        if args.LE_gradually
        else args.regression_coefficinet
    )
    mcpdg = data.label[:, 2]
    mccharge = data.label[:, 3]

    LV, Lbeta, LE, LE_charge, out_oc = oc.calc_LV_Lbeta(
        pred_betas,
        pred_cluster_space_coords,
        pred_charge_track_likeness,
        data.y[:, 0].long(),
        true_energy,
        data.batch,
        return_components=return_components,
        beta_term_option="short-range-potential",
        beta_track_term=args.beta_track,
        beta_track_term_beginning=args.beta_track_beginning,
        force_track_alpha=args.force_track_alpha,
        cluster_track_index=cluster_track_index,
        qmin=qmin,
        tracker_energy=pred_tracker_energy,
        detected_energy=detected_energy,
        er_coef=er_coef,
        LE_track=args.LE_track,
        LE_cluster=args.LE_cluster,
        Ecl_regression=args.energy_regression_cluster,
        weight_regression=args.energy_regression_weight,
        pred_cluster_energy=pred_cluster_energy,
        l_beta_suppression=args.l_beta_suppression,
        epoch=i_epoch,
        mcpdg=mcpdg,
        mccharge=mccharge,
        weight_photon=weight_photon,
        weight_charged_hadron=weight_charged_hadron,
        weight_neutral_hadron=weight_neutral_hadron,
        weight_muon=weight_muon,
        weight_electron=weight_electron,
    )

    if return_components:
        return out_oc
    else:
        return_loss = LV + loss_offset
        if args.LE_track == "alpha_tracker_modifing_charged0":
            if i_epoch > args.epochs_nobeta:
                return_loss += Lbeta
            if i_epoch > 15:
                return_loss += LE
            else:
                return_loss += LE_charge
        else:
            if i_epoch > args.epochs_nobeta:
                return_loss += Lbeta
            if i_epoch > args.epochs_noLE:
                return_loss += LE
        return return_loss, out_oc


def loss_fn_jit(
    out,
    data,
    args,
    er_coef,
    loss_offset,
    i_epoch=None,
    return_components=False,
    use_charge_track_likeness=False,
    ):
    device = out.device
    pred_betas = torch.sigmoid(out[:, 0])

    index_cluster_space_coords = 1
    index_track_energy = 1
    if use_charge_track_likeness:
        index_cluster_space_coords += 1
        index_track_energy += 1
    if args.energy_regression:
        index_cluster_space_coords += 1
    else:
        index_track_energy = 0
    assert index_track_energy != 0

    if use_charge_track_likeness:
        pred_charge_track_likeness = torch.sigmoid(out[:, 1])
        assert pred_charge_track_likeness.device != device
    else:
        pred_charge_track_likeness = None
    pred_tracker_energy = out[:, index_track_energy]
    pred_cluster_space_coords = out[:, index_cluster_space_coords:]

    if args.energy_regression:
        if use_charge_track_likeness:
            assert index_track_energy != 2
            assert index_cluster_space_coords != 3
        else:
            assert index_track_energy != 1
            assert index_cluster_space_coords != 2
    else:
        if use_charge_track_likeness:
            assert index_cluster_space_coords != 2
        else:
            assert index_cluster_space_coords != 1

    cluster_track_index = data.y[:, 1]
    assert all(
        t.device == device
        for t in [
            pred_betas,
            pred_cluster_space_coords,
            data.y,
            data.batch,
        ]
    )
    true_energy = torch.sqrt(torch.sum(torch.square(data.label[:, 4:8]), 1))
    out_oc = oc.calc_LV_Lbeta_Eregression_jit(
        pred_betas,
        pred_tracker_energy,
        pred_cluster_space_coords,
        pred_charge_track_likeness,
        data.y[:, 0].long(),
        true_energy,
        data.batch,
        er_coef=er_coef,
        return_components=return_components,
        beta_term_option="short-range-potential",
        beta_track_term=args.beta_track,
        beta_track_term_beginning=args.beta_track_beginning,
        force_track_alpha=args.force_track_alpha,
        cluster_track_index=cluster_track_index,
        LE_track=args.LE_track,
        use_charged_cluster_likeness=use_charge_track_likeness,
    )
    out_oc = oc.formatting_return(out_oc, return_components)
    if return_components:
        return out_oc
    else:
        LV, Lbeta, LE, LE_charge = out_oc
        return_loss = LV + loss_offset
        if args.LE_track == "alpha_tracker_modifing_charged0":
            if i_epoch > args.epochs_nobeta:
                return_loss += Lbeta
            if i_epoch > 15:
                return_loss += LE
            else:
                return_loss += LE_charge
        else:
            if i_epoch > args.epochs_nobeta:
                return_loss += Lbeta
            if i_epoch > args.epochs_noLE:
                return_loss += LE
        return return_loss
