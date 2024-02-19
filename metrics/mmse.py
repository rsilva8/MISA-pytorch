import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.stats import trim_mean
from .mcc import compute_rdc

def myMSE(Rr, r=None, c=None):
    d = np.min(Rr.shape)
    Rr = np.abs(Rr)
    if r is None and c is None:
        r, c = linear_sum_assignment(-Rr)
        Rr_sorted = Rr[r, c]
        Rr_sorted_mat = Rr[:, c]
    elif r is not None and c is not None:
        Rr_sorted = Rr[r, c]
        Rr_sorted_mat = Rr[:, c]
    else:
        raise ValueError("r and c should be both None or not None")
    
    mse = (1 + np.trace(Rr@Rr.T)/d - 2*np.trace(Rr_sorted_mat)/d)/2
    mcc = np.array([trim_mean(Rr_sorted, 0.25), np.mean(Rr_sorted), np.min(Rr_sorted)])

    return mse, mcc, Rr_sorted_mat, r, c

def MMSE(Y,Yh,u,S=None):
    n_segment = u.shape[1]
    n_source = Y.shape[1]
    n_modality = Y.shape[2]

    if S is None:
        S = [np.eye(n_source) for m in range(n_modality)]

    cind_sorted_pmps = np.zeros((n_modality,n_segment,n_source))
    rind_sorted_pmps = np.zeros((n_modality,n_segment,n_source))

    Rr_ps = np.zeros((n_segment,n_source,n_source))
    Rr_pm = np.zeros((n_modality,n_source,n_source))
    Rr_pmps = np.zeros((n_modality,n_segment,n_source,n_source))
    
    Rr_ps_sorted = np.zeros((n_segment,n_source,n_source))
    Rr_pm_sorted = np.zeros((n_modality,n_source,n_source))
    Rr_pmps_sorted = np.zeros((n_modality,n_segment,n_source,n_source))
    
    mse_ps, mcc_ps = np.zeros(n_segment), np.zeros((n_segment,3))
    mse_pm, mcc_pm = np.zeros(n_modality), np.zeros((n_modality,3))
    mse_pmps, mcc_pmps = np.zeros((n_modality,n_segment)), np.zeros((n_modality,n_segment,3))

    Ryyh = compute_rdc(Yh, Y, u)

    # Compute metrics per modality, per segment
    for seg in range(n_segment):
        for mm in range(n_modality):
            mk = np.split(S[mm] == 1, n_source, axis=0)
            for kr in range(n_source): # rows
                mkr = mk[kr]
                for kc in range(n_source): # columns
                    mkc = mk[kc]
                    # print("per modality, per segment")
                    # print(Ryyh_)
                    Rr_pmps[mm,seg,kr,kc] = Ryyh[mm][seg][mkr.T @ mkc]
            mse_pmps[mm,seg], mcc_pmps[mm,seg], Rr_pmps_sorted[mm,seg], rind_sorted_pmps[mm,seg], cind_sorted_pmps[mm,seg] = myMSE(Rr_pmps[mm,seg])
    
    mse_pm_mean=np.zeros(n_modality)
    mcc_pm_mean=np.zeros((n_modality,3))
    Rr_pm_mean_sorted=np.zeros((n_modality,n_source,n_source))
    cind_sorted_pm_mean=np.zeros((n_modality,n_source))
    rind_sorted_pm_mean=np.zeros((n_modality,n_source))

    Rr_pm_mean = np.mean(Rr_pmps, axis=1) # mean-agg across segments

    for m in range(n_modality):
        mse_pm_mean[m], mcc_pm_mean[m], Rr_pm_mean_sorted[m], rind_sorted_pm_mean[m], cind_sorted_pm_mean[m] = myMSE(Rr_pm_mean[m])
    
    
    ind = np.argmin(mse_pm_mean)
    r = rind_sorted_pm_mean[ind].astype(int)
    c = cind_sorted_pm_mean[ind].astype(int)
    R2_diag_min = np.min(np.stack((Rr_pm_mean[0][r, c], Rr_pm_mean[1][r, c])),axis=0)

    Rr2 = np.max(Rr_pm_mean,axis=0)
    for i in range(n_source):
        Rr2[i,c[i]] = R2_diag_min[i] # min-agg across modalities along the diagonal
    mse2, mcc2, Rr_sorted2, _, _ = myMSE(Rr2, r, c)

    # Compute metrics per segment, aggregated over modalities
    for seg in range(n_segment):
        for mm in range(n_modality):
            mk = np.split(S[mm] == 1, n_source, axis=0)
            for kr in range(n_source): # rows
                mkr = mk[kr]
                for kc in range(n_source): # columns
                    mkc = mk[kc]
                    Ryyh_ = [Ryyh[m][seg][mkr.T @ mkc] for m in range(n_modality)]
                    # print("per segment, aggregated over modalities")
                    # print(Ryyh_)
                    if kc == c[kr]:
                        Rr_ps[seg,kr,kc] = np.abs(Ryyh_).min()
                    else:
                        Rr_ps[seg,kr,kc] = np.abs(Ryyh_).max()

        mse_ps[seg], mcc_ps[seg], Rr_ps_sorted[seg], _, _ = myMSE(Rr_ps[seg], r, c)

    # Compute metrics per modality, aggregated over segments
    for mm in range(n_modality):
        mk = np.split(S[mm] == 1, n_source, axis=0)
        for seg in range(n_segment):
            for kr in range(n_source): # rows
                mkr = mk[kr]
                for kc in range(n_source): # columns
                    mkc = mk[kc]
                    Ryyh_ = [Ryyh[mm][s][mkr.T @ mkc] for s in range(n_segment)]
                    # print("per modality, aggregated over segments")
                    # print(Ryyh_)
                    Rr_pm[mm,kr,kc] = np.abs(Ryyh_).mean()
        mse_pm[mm], mcc_pm[mm], Rr_pm_sorted[mm], _, _ = myMSE(Rr_pm[mm], r, c)

    Rr = np.mean(np.abs(Rr_ps), axis=0)
    mse, mcc, Rr_sorted, _, _ = myMSE(Rr, r, c)

    metric = { 'mse_pmps': mse_pmps, 'mcc_pmps': mcc_pmps, 'Rr_pmps': Rr_pmps_sorted, 'mse_ps': mse_ps, 'mcc_ps': mcc_ps, 'Rr_ps': Rr_ps_sorted, 'mse_pm': mse_pm, 'mcc_pm': mcc_pm, 'Rr_pm': Rr_pm_sorted, 'mse': mse, 'mcc': mcc, 'Rr': Rr_sorted, 'mse2': mse2, 'mcc2': mcc2, 'Rr2': Rr_sorted2 }
    return metric