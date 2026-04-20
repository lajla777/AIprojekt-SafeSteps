import numpy as np
import matplotlib.pyplot as plt
from dataDecodingSis import dekodiraj_bin, GIRO_ID, ACC_ID, MAG_ID, TOF_ID

def sestavi_podatke(seznam_paketov, tofSenzor=False):
    if len(seznam_paketov) == 0:
        return 0, np.array([])
    
    vsi = [] #matrike podatkov iz vsakega paketa
    T = [] #casovnerazlike med paketi
    N = [] #stevilo vzorcev v vsakem paketu

    for i, p in enumerate(seznam_paketov):
        if p.id in [1, 2, 3]:
            data = np.frombuffer(p.data, dtype=np.int16)
        elif tofSenzor:
            data = np.frombuffer(p.data, dtype=np.uint16)
        else:
            data = np.frombuffer(p.data, dtype=np.uint8)

        Nvz = len(data) if tofSenzor else len(data) // 3
        N.append(Nvz)

        if not tofSenzor:
            data = data.reshape(-1, 3)
        vsi.append(data)

        if i > 0:
            T.append(p.ts - seznam_paketov[i-1].ts)

    signal = np.concatenate(vsi) if tofSenzor else np.vstack(vsi)

    if len(T) > 0:
        Fvz = np.mean(N) / np.mean(T)
    else:
        Fvz = 0

    return Fvz, signal

def prikazi_signal(signal, naslov=None, startInd=None, endInd=None, Fvz=None):
    if startInd is None:
        startInd = 0
    if endInd is None:
        endInd = signal.shape[0]

    sig = signal[startInd:endInd]  

    if Fvz is not None:
        if Fvz > 0:
            x = np.arange(startInd, endInd) / Fvz
            xlabel = "čas (s)"
        else:            
            x = np.arange(startInd, endInd)
            xlabel = "vzorec"

    plt.figure(figsize=(13, 6))
    plt.title(naslov)

    if "ToF" in naslov:
        masked = np.where(sig == 0xFFFF, np.nan, sig)

        plt.plot(x, masked, label="TOF", color="#45067F")
        
        plt.xlabel(xlabel)
        plt.ylabel("razdalja (mm)")
        plt.ylim(0, 2000)
        
        plt.legend()
        plt.grid()
        plt.show()
    else:
        plt.plot(x, sig[:,0], label="X", color='#B063F8')
        plt.plot(x, sig[:,1], label="Y", color="#FC78F3")
        plt.plot(x, sig[:,2], label="Z", color='#A80354')

        if "Žiroskop" in naslov:
            plt.ylabel("rotacija (°/s)")
        elif "Akcelometer" in naslov:
            plt.ylabel("pospešek (G)")
        elif "Magnetometer" in naslov:
            plt.ylabel("magnetno polje (Gauss)")
        
        plt.xlabel(xlabel)
        
        plt.legend()
        plt.grid()
        plt.show()

def signali_skupaj(bin_datoteka):
    vsiPaketi = dekodiraj_bin(bin_datoteka)

    paketiGiro = [p for p in vsiPaketi if p.id == GIRO_ID]
    paketiAcc = [p for p in vsiPaketi if p.id == ACC_ID]
    paketiMag = [p for p in vsiPaketi if p.id == MAG_ID]
    paketiTof = [p for p in vsiPaketi if p.id == TOF_ID]

    FvzGiro, signalGiro = sestavi_podatke(paketiGiro)
    FvzAcc, signalAcc = sestavi_podatke(paketiAcc)
    FvzMag, signalMag = sestavi_podatke(paketiMag)
    FvzTof, signalTof = sestavi_podatke(paketiTof, tofSenzor=True)

    signalGiro = signalGiro * 8.75e-3
    signalAcc = signalAcc * 6.125e-5
    signalMag = signalMag * 1.5e-3

    #casovne osi za vsak signal
    tGiro = np.arange(len(signalGiro)) / FvzGiro
    tAcc = np.arange(len(signalAcc)) / FvzAcc
    tMag = np.arange(len(signalMag)) / FvzMag
    tTof = np.arange(len(signalTof)) / FvzTof

    plt.figure(figsize=(11, 7))
    plt.suptitle(f"Sensor data from {bin_datoteka}", fontsize=16)

    plt.subplot(4,1,1)
    plt.plot(tGiro, signalGiro[:,0], label="x")
    plt.plot(tGiro, signalGiro[:,1], label="y")
    plt.plot(tGiro, signalGiro[:,2], label="z")
    plt.title(f"Gyroscope (Fvz={FvzGiro:.1f} Hz, resolution 8.75e-3 °/s)")
    plt.ylabel("rotational speed (°/s)")
    plt.legend()
    plt.grid()

    plt.subplot(4,1,2)
    plt.plot(tAcc, signalAcc[:,0], label="x")
    plt.plot(tAcc, signalAcc[:,1], label="y")
    plt.plot(tAcc, signalAcc[:,2], label="z")
    plt.title(f"Accelerometer (Fvz={FvzAcc:.1f} Hz, resolution 6.125e-5 g)")
    plt.ylabel("acceleration (G)")
    plt.legend()
    plt.grid()

    plt.subplot(4,1,3)
    plt.plot(tMag, signalMag[:,0], label="x")
    plt.plot(tMag, signalMag[:,1], label="y")
    plt.plot(tMag, signalMag[:,2], label="z")
    plt.title(f"Magnetometer (Fvz={FvzMag:.1f} Hz, resolution 1.5e-3 Gauss)")
    #plt.xlabel("time (s)")
    plt.ylabel("magnetic field (Gauss)")
    plt.legend()
    plt.grid()

    plt.subplot(4,1,4)
    if len(signalTof) > 0 and FvzTof > 0:
        tTof = np.arange(len(signalTof)) / FvzTof
        masked = np.where(signalTof == 0xFFFF, np.nan, signalTof)
        plt.plot(tTof, masked, label="TOF", color="#BA00C0")
    else:
        plt.text(0.5, 0.5, "No TOF data", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title(f"TOF Sensor (Fvz={FvzTof:.1f} Hz)")
    plt.xlabel("time (s)")
    plt.ylabel("distance (mm)")
    plt.ylim(0, 3200)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    print("VIZUALIZACIJA PODATKOV")
    vsiPaketi = dekodiraj_bin("log17.bin")

    while True:
        print("\nKateri signal želiš prikazati?")
        print("- vsi signali skupaj (testni): 1")
        print("- vsi signali skupaj (moji): 2")
        print("- žiroskop: 3")
        print("- akcelometer: 4")
        print("- magnetometer: 5")
        print("- ToF senzor: 6")

        izbira = input("\nIzberi možnost (1-6): ")

        if izbira == "1":
            signali_skupaj("LOG011.BIN")  
        elif izbira == "2":
            signali_skupaj("log17.bin")
            
        elif izbira == "3" or izbira == "4" or izbira == "5":

            if izbira == "3":
                paketiGiro = [p for p in vsiPaketi if p.id == GIRO_ID]
                FvzGiro, signalGiro = sestavi_podatke(paketiGiro)
                signalGiro = signalGiro * 8.75e-3
                print(f"Fvz žiroskopa= {FvzGiro:.2f} Hz")

                prikazi_signal(signalGiro, f"Žiroskop (Fvz={FvzGiro:.2f} Hz)", Fvz=FvzGiro)
                prikazi_signal(signalGiro,
                            "Žiroskop - interval",
                            1,
                            int(FvzGiro * 3)+1,
                            Fvz=FvzGiro)
                #x os = rotacija naprej/nazaj
                #y os = rotacija levo/desno
                #z os = rotacija okoli svoje osi

            elif izbira == "4":
                paketiAcc = [p for p in vsiPaketi if p.id == ACC_ID]
                FvzAcc, signalAcc = sestavi_podatke(paketiAcc)
                signalAcc = signalAcc * 6.125e-5 
                print(f"Fvz akcelometera = {FvzAcc:.2f} Hz")

                prikazi_signal(signalAcc, f"Akcelometer (Fvz={FvzAcc:.2f} Hz)", Fvz=FvzAcc)
                prikazi_signal(signalAcc,
                            "Akcelometer - interval",
                            1,
                            int(FvzAcc * 3)+1,
                            Fvz=FvzAcc)
                #x os = pospešek naprej/nazaj
                #y os = pospešek levo/desno
                #z os = pospešek gor/dol(gravitacija)

            elif izbira == "5":
                paketiMag = [p for p in vsiPaketi if p.id == MAG_ID]
                FvzMag, signalMag = sestavi_podatke(paketiMag)
                signalMag = signalMag * 1.5e-3
                print(f"Fvz magnetometra = {FvzMag:.2f} Hz")    

                prikazi_signal(signalMag, f"Magnetometer (Fvz={FvzMag:.2f} Hz)", Fvz=FvzMag)
                prikazi_signal(signalMag,   
                            "Magnetometer - interval",
                            1,
                            int(FvzMag * 3)+1,
                            Fvz=FvzMag)
                #x os = magnetno polje naprej/nazaj
                #y os = magnetno polje levo/desno
                #z os = magnetno polje gor/dol
                
        elif izbira == "6":
                paketiTof = [p for p in vsiPaketi if p.id == TOF_ID]
                FvzTof, signalTof = sestavi_podatke(paketiTof, tofSenzor=True)
                print(f"Fvz TOF senzorja = {FvzTof:.2f} Hz")    
                prikazi_signal(signalTof, f"ToF Senzor (Fvz={FvzTof:.2f} Hz)", Fvz=FvzTof)
                prikazi_signal(signalTof,
                            "ToF Senzor - interval",
                            1,
                            int(FvzTof * 3)+1,
                            Fvz=FvzTof)
