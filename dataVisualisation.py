import numpy as np
import paket as paketek

TIP_ZLOGA = {
    1: 2, #1=ziroskop 2=pospeskomeer in 3=magnetometer
    2: 2,
    3: 2
}

def sestavi_podatke(seznam_paketov):
    paket_seznam = []
    st_vz_seznam = []
    matrike = []

    for i, paket in enumerate(seznam_paketov):
        st_vzorcev = paket.data.shape[0]
        st_vz_seznam.append(st_vzorcev)
        matrike.append(paket.data)

        if i<len(seznam_paketov)-1:
            dt = seznam_paketov[i+1].ts - paket.ts
            paket_seznam.append((paket.id, st_vzorcev, dt))

    paket_avg = np.mean(st_vz_seznam)
    st_vz_avg = np.mean(st_vz_seznam)
    fvz = st_vz_avg /paket_avg

    signal = np.vstack(matrike)

    return fvz, signal

if __name__ == "__main__":
    pass
    
