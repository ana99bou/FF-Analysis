import numpy as np

def model_eq30(params, tvals, nsq_order,
               mDs_gs, mDs_es, mBs_gs, mBs_es,
               Z0_Ds, Z1_Ds, Z0_Bs, Z1_Bs,
               T):
    """
    Eq. (30) model function for combined 3pt/2pt ratio fits.
    """
    tvals = np.asarray(tvals, dtype=float)
    nsq_order = [int(x) for x in nsq_order]
    n = len(nsq_order)

    O00 = params[0*n:1*n]
    O01 = params[1*n:2*n]
    O10 = params[2*n:3*n]

    out = []

    # Use lowest nsq as rest proxy
    first_nsq = sorted(mDs_gs.keys())[0]
    M_D0 = float(mDs_gs[first_nsq])
    M_D1 = float(mDs_es[first_nsq])

    for i, nsq in enumerate(nsq_order):
        E_D0 = float(mDs_gs[int(nsq)])
        E_D1 = float(mDs_es[int(nsq)])
        M_B0 = float(mBs_gs)
        M_B1 = float(mBs_es)

        ZD0 = float(Z0_Ds[int(nsq)])
        ZD1 = float(Z1_Ds[int(nsq)])
        ZB0 = float(Z0_Bs)
        ZB1 = float(Z1_Bs)

        t = tvals

        # numerator
        term00 = ZB0 * O00[i] * ZD0 * np.exp(-E_D0*t - M_B0*(T - t)) / (4*E_D0*M_B0)
        term10 = ZB1 * O10[i] * ZD0 * np.exp(-E_D0*t - M_B1*(T - t)) / (4*E_D0*M_B1)
        term01 = ZB0 * O01[i] * ZD1 * np.exp(-E_D1*t - M_B0*(T - t)) / (4*E_D1*M_B0)

        denom_Bs = ((ZB0**2)/(2*M_B0))*np.exp(-M_B0*(T-t)) + ((ZB1**2)/(2*M_B1))*np.exp(-M_B1*(T-t))
        denom_Ds = ((ZD0**2)/(2*M_D0))*np.exp(-E_D0*t)     + ((ZD1**2)/(2*M_D1))*np.exp(-E_D1*t)
        pref = 4*E_D0*M_B0 / np.exp(-E_D0*t - M_B0*(T - t))

        R = np.sqrt(pref) * (term00 + term10 + term01) / np.sqrt(denom_Bs * denom_Ds)
        out.append(R)

    return np.concatenate(out)
