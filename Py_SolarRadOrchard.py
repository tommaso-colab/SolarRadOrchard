import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3
import math
import numpy as np
from datetime import datetime, timedelta




######PREPROCESS
##input file: read CSV with radiation
def reset(dt):
    return dt + timedelta(hours = +1, minutes = -2)
dateparser = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
#leggi file con radiazione da Winet
mainpath = './Py_SolarRad_Orchards/'
csvFilePath = mainpath + 'meteo_winet_s30.csv'
rad_meteo = pd.read_csv(csvFilePath, sep=';',  decimal = '.', parse_dates = [0], date_parser=dateparser)  
dateTimeRESET = pd.Series(reset(rad_meteo.iloc[:,0])).dt.round("H") #remove to have interger hour correction of datetime of winet
rad_meteo['Data'] = dateTimeRESET #cambio vettore index data con data corretta
RadDT = rad_meteo.set_index('Data')
RadDT = pd.DataFrame(RadDT)
RadDT.plot()
RadDtzT = RadDT.tz_localize('CET')
## Calcola RADIAZIONE da PVLIB
# pvlib.irradiance.poa_horizontal_ratio(84, 90, 15, 0)
# pvlib.irradiance.isotropic(84, 300)
lat = 44.549192
long = 11.416346
tus = Location(lat, long, 'Europe/Rome', 32, 'Cadriano')
times = pd.date_range(start='2020-06-01', end='2020-10-20', freq='60min', tz=tus.tz)
cs = tus.get_clearsky(times)  # RadiationCalculated
solpos = solarposition.get_solarposition(times, lat, long)#sun elevation
frames = pd.concat([cs, solpos],axis = 1)#combine
frames['dhi/ghiRatio'] = frames['dhi']/frames['ghi'] #calculate theoretical ghi/dhi

cs[['dni', 'dhi']].plot();
plt.ylabel('Irradiance $W/m^2$');
plt.title('sun radiation parameters');
frames.plot();
plt.ylabel('degree');
plt.title('solpos');

#FACCIO MERGE dei dati rilevati dalla centralina winet e dei dati simulati con il MODELLO pvlib
##merge Radiazione From Winet with CalRadiation
mergeRadiation = pd.merge(frames, RadDtzT, how='left', left_index=True, right_index=True)
mergeRadiation[['par_wm2', 'ghi']].plot();
mergeRadiation['par_wm2_dhi'] = mergeRadiation['par_wm2']*mergeRadiation['dhi/ghiRatio'] # stimo la radiazione diffusa dal rapporto tra rad glo/rad diff in giornata senza nuvole da modello
mergeRadiation['par_wm2_dir_hi'] = mergeRadiation['par_wm2'] - mergeRadiation['par_wm2_dhi'] # stimo la radiazione diffusa dal rapporto tra rad glo/rad diff in giornata senza nuvole da modello
mergeRadiationSUBSET = mergeRadiation.loc['2020-06-02':'2020-06-10']
mergeRadiationSUBSET[['par_wm2', 'ghi', 'dhi']].plot();
mergeRadiationSUBSET[['par_wm2','par_wm2_dhi','par_wm2_dir_hi', 'ghi', 'dhi']].plot();
mergeRadiationSUBSET[['par_wm2','par_wm2_dhi','par_wm2_dir_hi']].plot();

def CalculateET0 (t, U, R, H, P):
    # t = mean temperature in Kelvin
    # Pbar = Pa atmospheric pressure in kPa
    # U = Um/s mean wind speed in meters per second
    # R = Rn average net radiation over the hour Watts persquare meter
    if t == '--':
        t = 0
    if U == '--':
        U = 0
    if R == '--':
        R = 0
    if H == '--':
        H = 0
    if P == '--':
        P = 0
    t = float(t)
    U = float(U)
    R = float(R)
    H = float(H)
    P =  float(P)
    tk = t + 273.16 #temp in kelvin
    PKpa = P  * 0.1
    ea = 0.6108 * math.exp((17.27 * t)/(t + 237.3))
    ed = ea * H/100
    
    delta = ea/tk * ((6790.4985/tk) - 5.02808)
    psi = 0.000646*(1+0.000946*t)*PKpa
    W = delta /(delta+psi)
    F = 0.030 + 0.0576 * U
    labda = 694.5 *(1 - 0.000946 * t)
    et = W * R/labda +(1-W)*(ed-ea)*F
    
    return et

# ghi - global horizontal irradiance
# dhi - diffuse horizontal irradiance
# dni - direct normal irradiance

class radiation_fraction_model_tilted:
    def __init__(self, radiationArray, fractionVector, surf_param):
        self.radiationArray = radiationArray # array con valori di funzione get_clearSky di pvlib
        self.fractionVector = fractionVector # array di frazione di filtrazione  c, c1, b1, ci, b1i 
        self.surf_param = surf_param # vettore con 4 par: surf tilter, surf azimut # (e.g. surface facing up = 0, surface facing horizon = 90)
        
        
        # vectorFracionOMBREGG = [0.49, 0.94, 0.19,  0.86, 0.39]
        
        # #fraction:
        # c = frazione rad dir ombra 
        # c1 = frazione rad dir keep
        # b1 = frazione rad diff 
        # ci = frazione rad dir  
        # b1i = frazione rad diff 
        
        # TRASMITTANZA DA RILIEVI
        # 0.19 ratio radiazione parte in ombra - OMBREGGIANTE
        # 0.48 ratio radiazione parte in ombra - ANTIGRANDINE 
        
        # 0.39 ratio tratto ombra parte al sole - OMBREGGIANTE
        # 0.94 ratio tratto KEEP parte al sole - OMBREGGIANTE
        # 0.86 ratio parte al sole tutto - ANTIGRANDINE
    
    def CalculateFraction_of_radiation(self):
        # print("#######Calculated Fraction Process")
        radiationArray = self.radiationArray 
        fractionVector = self. fractionVector
        surf_param = self.surf_param
   
        
        timeStep = str(radiationArray.columns[0])
        radValue_diffuse = float(radiationArray.loc['par_wm2_dhi',:]) 
        radValue_direct = float(radiationArray.loc['par_wm2_dir_hi',:]) 
        azValue = float(radiationArray.loc['azimuth',:])
        zenValue = float(radiationArray.loc['zenith',:])
        
        ## CALCOLO Radiazione diretta e diffusa per ogni componente per effetto delle reti sia
        ## utilizzo il rapporto di trasmittanza dei rilievi in campo
        #
        #ombreggiante
        R1ho_omb_dirhi = radValue_direct * fractionVector[0] # radiazione diretta trasmessa tratto sole sezione ombra
        R2ho_omb_keep_dirhi = radValue_direct * fractionVector[1] # radiazione diretta trasmessa tratto sole sezione KIT
        r1ho_omb_diretta = radValue_direct * fractionVector[2] # radiazione diretta trasmessa su tratto in ombra dietro
        # r1ho_omb_diretta = 0
        #
        R1ho_omb_diffuso = radValue_diffuse * fractionVector[0] # radiazione diffusa trasmessa tratto sole sezione ombra
        R2ho_omb_keep_diffuso = radValue_diffuse * fractionVector[1] # radiazione diffusa trasmessa tratto sole sezione kit
        r1ho_omb_diffuso = radValue_diffuse * fractionVector[2] # radiazione diffusa trasmessa su tratto in ombra dietro
        #
        #antigrandine
        R1ho_anti_dirhi = radValue_direct * fractionVector[3] # radiazione diretta trasmessa tratto sole
        r1ho_anti_diretta = radValue_direct * fractionVector[4]# radiazione diretta trasmessa tratto ombra dietro
        # r1ho_anti_diretta = 0
        #
        R1ho_anti_diffuso = radValue_diffuse * fractionVector[3] # radiazione diffusa trasmessa tratto sole
        r1ho_anti_diffuso = radValue_diffuse * fractionVector[4] # radiazione diffusa trasmessa tratto ombra dietro     
        
        if float(azValue)  > 180:
            azimut = (360-azValue) #faccio ciclo solo su parte ad est 
        else:
            azimut = azValue
        ## CALCOLO RADIAZIONE su SUPERFICI NON ORIZZONTALI
        #calcolo della frazione radiazione diretta su superfice non orizzontale
        Direct_Rad_HorTilt_FRACTION = pvlib.irradiance.poa_horizontal_ratio(surf_param[0], surf_param[1], 
                                                     zenValue, azimut) # calcolo la frazione tra horizontale ed inclinato
        ##Calcolo radiazione DIRETTA su sup inclinata
        #ombreggiante
        t_R1ho_omb_dirhi      = R1ho_omb_dirhi * Direct_Rad_HorTilt_FRACTION 
        t_R2ho_omb_keep_dirhi = R2ho_omb_keep_dirhi * Direct_Rad_HorTilt_FRACTION
        #anti
        t_R1ho_anti_dirhi     = R1ho_anti_dirhi * Direct_Rad_HorTilt_FRACTION 
        #
        ##Calcolo radiazione DIFFUSA su sup inclinata
        #ombreggiante
        tilt_R1ho_omb_diff = pvlib.irradiance.isotropic(surf_param[0], R1ho_omb_diffuso) #frazione di radiazione 
        tilt_R2ho_omb_keep_diff = pvlib.irradiance.isotropic(surf_param[0],  R2ho_omb_keep_diffuso) #frazione di radiazione
        #anti
        tilt_R1ho_anti_diffuso = pvlib.irradiance.isotropic(surf_param[0] ,  R1ho_anti_diffuso)
        
        #CALCOLO EFFETTO dell'INCLINAZIONE su RADIAZIONE DIRETTA E DIFFUSA di SUPERFICE in OMBRA
        #tratto in ombra radiazione DIFFUSA
        # tilt_r1ho_omb_diffuso = pvlib.irradiance.isotropic(180-surf_param[0], r1ho_omb_diffuso) #frazione di radiazione 
        # tilt_r1ho_anti_diffuso = pvlib.irradiance.isotropic(180-surf_param[0],  r1ho_anti_diffuso) #frazione di radiazione
        # tratto in ombra radiazione DIRETTA
        #opzione 0: calcolo effetto piano inclinato su radiazione solo su somma radiazione diretta + diffusa
        tilt_r1ho_omb_diffuso = pvlib.irradiance.isotropic(180-surf_param[0], r1ho_omb_diffuso + r1ho_omb_diretta) #frazione di radiazione 
        tilt_r1ho_anti_diffuso = pvlib.irradiance.isotropic(180-surf_param[0],  r1ho_anti_diffuso + r1ho_anti_diretta) #frazione di radiazione
        t_r1ho_omb_diretta = 0
        t_r1ho_anti_diretta  = 0
        #opzione 1: prendo radiazione diretta in tratto in ombra e utilizzo formula per PIANO INCLINATO rad DIFFUSA 
        # t_r1ho_omb_diretta = pvlib.irradiance.isotropic(180-surf_param[0],r1ho_omb_diretta)
        # t_r1ho_anti_diretta  = pvlib.irradiance.isotropic(180-surf_param[0], r1ho_anti_diretta)
        # opzione 2: considero il piano inclinato dalla parte opposta rispetto all'azimut
        # if azValue > 180:
        #     refAzimut = 180
        # else:
        #     refAzimut = 0
        
        # InOMBRA_Direct_Rad_HorTilt_FRACTION = pvlib.irradiance.poa_horizontal_ratio(surf_param[0], surf_param[1]+refAzimut, 
        #                                               zenValue, azimut) # calcolo la frazione tra horizontale ed inclinato        
        # t_r1ho_omb_diretta = r1ho_omb_diretta * InOMBRA_Direct_Rad_HorTilt_FRACTION
        # t_r1ho_anti_diretta  = r1ho_anti_diretta * InOMBRA_Direct_Rad_HorTilt_FRACTION 
        
        # if t_r1ho_omb_diretta < 0:
        #     t_r1ho_omb_diretta = 0
        # if t_r1ho_anti_diretta < 0:
        #     t_r1ho_anti_diretta = 0

        if radValue_direct < 10:
            radValue_diffuse = 0 
            radValue_direct = 0
            
            R1ho_omb_dirhi = 0 
            R2ho_omb_keep_dirhi = 0 
            r1ho_omb_diffuso = 0   
            
            R1ho_anti_dirhi = 0 
            r1ho_anti_diffuso = 0 
            
            Direct_Rad_HorTilt_FRACTION = 0 
            
            t_R1ho_omb_dirhi = 0 
            t_R2ho_omb_keep_dirhi = 0 
            tilt_R1ho_omb_diff = 0 
            tilt_R2ho_omb_keep_diff = 0 
            tilt_r1ho_omb_diffuso = 0 
            
            t_r1ho_omb_diretta = 0 
            t_R1ho_anti_dirhi = 0 
            t_r1ho_anti_diretta = 0 
            
            tilt_R1ho_anti_diffuso = 0 
            tilt_r1ho_anti_diffuso = 0 
            
        
        d_Row = {'indexV': timeStep ,
                 'radValue_diffuse': radValue_diffuse, 
                 'radValue_direct': radValue_direct,  
                 'R1ho_omb_dirhi': R1ho_omb_dirhi, 
                 'R2ho_omb_keep_dirhi': R2ho_omb_keep_dirhi, 
                 'r1ho_omb_diffuso': r1ho_omb_diffuso,  
                 'R1ho_anti_dirhi': R1ho_anti_dirhi, 
                 'r1ho_anti_diffuso': r1ho_anti_diffuso,
                 'Direct_Rad_HorTilt_FRACTION': Direct_Rad_HorTilt_FRACTION,               
                
                 't_R1ho_omb_dirhi': t_R1ho_omb_dirhi, # ombreggiante lato sole diretta tratto  KIT
                 't_R2ho_omb_keep_dirhi': t_R2ho_omb_keep_dirhi, # ombreggiante lato sole diretta tratto OMBRA
                 'tilt_R1ho_omb_diff':tilt_R1ho_omb_diff,# ombreggiante lato sole diffusa tratto  KIT
                 'tilt_R2ho_omb_keep_diff':tilt_R2ho_omb_keep_diff,# ombreggiante lato sole diffusa  tratto OMBRA
                 'tilt_r1ho_omb_diffuso': tilt_r1ho_omb_diffuso, # ombreggiante lato OMBRA diffusa
                 't_r1ho_omb_diretta':t_r1ho_omb_diretta, # ombreggiante lato OMBRA diretta

                 't_R1ho_anti_dirhi':t_R1ho_anti_dirhi,           # antigr lato sole diretta  
                 't_r1ho_anti_diretta':t_r1ho_anti_diretta,       # antigr lato ombra diretta
                 'tilt_R1ho_anti_diffuso':tilt_R1ho_anti_diffuso, # antigr lato sole diffusa
                 'tilt_r1ho_anti_diffuso':tilt_r1ho_anti_diffuso  # antigr lato ombra diffusa                                                          
                                  }
        return d_Row
            

       
class Beam_Surf_canopy_orchard_Calculator:
            
    def __init__(self, SunElevationAngle, distanceOfNetHDis_FromTree, DiscontiDim, 
                 angleTopCrw, BottomCanopythr, bhz, LChioma,
                  minAngleShadow,  HtreeMAX,  TreeBaseAngle, inserzioneRami):
        
        self.SunElevationAngle = SunElevationAngle # gradi elevazione sole
        self.distanceOfNetHDis_FromTree = distanceOfNetHDis_FromTree # distanza in metri dal filare della rete ombreggiante
        self.DiscontiDim = DiscontiDim # distanza in metri tra le due sezioni di rete ombreggianti
        self.angleTopCrw = angleTopCrw # angolo tra asse verticale del filare e prolungamento del lato della chioma
        self.BottomCanopythr = BottomCanopythr # PRMAX h in metri di intersezione prolungamento del beam che tocca fondo base chioma ed interseca il prolungamento del fuso sotto terra
        self.bhz = bhz # distanza tra tetto rete e prolungamento asse pianta e prolungamento perimetro chioma
        self.LChioma = LChioma # lunghezza chioma da apice sopra rete 
        self.minAngleShadow = minAngleShadow # angolo minimo sole al di sotto del quale non ho ombra
        self.HtreeMAX = HtreeMAX # Altezza Max che fa ombra    
        self.TreeBaseAngle = TreeBaseAngle #kangolo alla base della chioma
        self.inserzioneRami = inserzioneRami # h da terra di inserzione rami
        
    def CalculateSurface(self):
        SunElevationAngle = self.SunElevationAngle
        distanceOfNetHDis_FromTree = self.distanceOfNetHDis_FromTree
        DiscontiDim = self.DiscontiDim
        angleTopCrw = self.angleTopCrw
        BottomCanopythr = self.BottomCanopythr
        bhz = self.bhz
        LChioma = self.LChioma
        
        #modello
        sunTang = math.tan(math.radians(SunElevationAngle))                
        b = distanceOfNetHDis_FromTree * sunTang
        B = (distanceOfNetHDis_FromTree + DiscontiDim) * sunTang
        print("B TRUE distance=",B)
        if B > BottomCanopythr:
            B = BottomCanopythr
        D = B - b
        print ("math.tan(sunAngle)=", sunTang)
        print("b distance=", b)
        print("B distance=",B)
        print("D distance=", D)
        
        e = 90 - SunElevationAngle
        m = 180 - e - angleTopCrw
        
        print("e angle=",e)
        print("m angle=", m)
        
        S = D*(math.sin(math.radians(e))/math.sin(math.radians(m)))
        print ("math.sin(e)=", math.sin(math.radians(e)))
        print ("math.sin(m)=", math.sin(math.radians(m)))
        print ("S distance=", S)
        
         
        B1 = bhz / (math.cos(math.radians(angleTopCrw)))
        print ("B1 distance=", B1)
        
        B2 = (b + bhz)*(math.sin(math.radians(e))/math.sin(math.radians(m)))
        print ("B2Length distance=", B2)
        
        LengthTOBottom = LChioma - B2 - S
        if (LengthTOBottom < 0):
            LengthTOBottom = 0
        print ("LengthTOBottom distance=", LengthTOBottom)
        
        # LengthTOTop = LChioma -  B1 - S -  LengthTOBottom
        # print ("LengthTOtop until net distance=", LengthTOBottom)
        
        LengthTOTop = B2 - B1
        print ("LengthTOtop until net distance=", LengthTOBottom)
        
        TLen = S + LengthTOBottom + LengthTOTop
        DicRe = {'KiT length': S, 'len_bottom':LengthTOBottom, 'len_top': LengthTOTop, 'totLe_ver': TLen}
        print(DicRe)
        return DicRe
    
    def CalculateShadowLength (self):
        
         minAngleShadow = self.minAngleShadow
         HtreeMAX = self.HtreeMAX
         TreeBaseAngle = self.TreeBaseAngle
         SunElevationAngle = self.SunElevationAngle
         inserzioneRami = self.inserzioneRami  
         
         
         DeltaSunAng = minAngleShadow - SunElevationAngle
         SectionA = (HtreeMAX - inserzioneRami)/math.sin(math.radians(minAngleShadow))
         CAA = 180-minAngleShadow-TreeBaseAngle
         CAA1 = 180-CAA-DeltaSunAng
         print(CAA)
         print(CAA1)
         LengthShaded = (SectionA*math.sin(math.radians(DeltaSunAng)))/math.sin(math.radians(CAA1))
         if (LengthShaded < 0):
             LengthShaded = 0
         DicRe = {'shadowed len': LengthShaded}
         return DicRe 
    

class averagging_radiation:
    
        def __init__(self, Ref_Length, Shadow_Length, Vector_RadiationTrasmitted):
            self.Ref_Length = Ref_Length        # dictionary Length of sunny shadowed
            self.Shadow_Length = Shadow_Length  # dictionary shadowed length 
            self.Vector_RadiationTrasmitted = Vector_RadiationTrasmitted # pandaDF. radiation parameters

        def calculating_values(self):
            Ref_Length = self.Ref_Length       
            Shadow_Length = self.Shadow_Length  
            Vector_RadiationTrasmitted = self.Vector_RadiationTrasmitted 
            print(Ref_Length)
            print(Shadow_Length)
            print(Vector_RadiationTrasmitted)
            
            ##OMBREGGIANTE
            #preprocesso su shadow che non può mai essere maggiore della lunghezza della sup al sole della chioma
            if Shadow_Length['shadowed len'] > Ref_Length['totLe_ver']:
                Shadow_Length['shadowed len'] = Ref_Length['totLe_ver']
            
            #lato ombra OMBREGGIANTE - Ombraggiante
            Rad_LatoOmbra_ombreg_dir = Ref_Length['totLe_ver'] * Vector_RadiationTrasmitted['t_r1ho_omb_diretta']
            Rad_LatoOmbra_ombreg_diff = Ref_Length['totLe_ver'] * Vector_RadiationTrasmitted['tilt_r1ho_omb_diffuso']

            #lato sole tratto in OMBRA - Ombraggiante
            if Shadow_Length['shadowed len'] > Ref_Length['totLe_ver']:
                shadowRef = Ref_Length['totLe_ver']
            else:
                shadowRef = Shadow_Length['shadowed len']
            
            Rad_LatoSole_omb_dir_shadowed = shadowRef * Vector_RadiationTrasmitted['t_r1ho_omb_diretta']
            Rad_LatoSole_omb_diff_shadowed = shadowRef* Vector_RadiationTrasmitted['tilt_r1ho_omb_diffuso']

            #lato sole tratto KiT e NON Kit- ombreggiante
            if Ref_Length['len_bottom'] < Shadow_Length['shadowed len']:
                Length_kit = Ref_Length['KiT length'] -(Shadow_Length['shadowed len'] - Ref_Length['len_bottom'])
                LenBottom = 0 
                if (Length_kit + Ref_Length['len_bottom']) > Shadow_Length['shadowed len']:
                    LenTop = Ref_Length['len_top']
                else:
                    LenTop = Ref_Length['totLe_ver'] - Shadow_Length['shadowed len']            
            else:
                Length_kit = Ref_Length['KiT length']
                LenBottom = Ref_Length['len_bottom']- Shadow_Length['shadowed len']
                LenTop = Ref_Length['len_top'] 
            
            
            Rad_LatoSole_omb_dir_Kit = Length_kit * Vector_RadiationTrasmitted['t_R2ho_omb_keep_dirhi']
            Rad_LatoSole_omb_diff_Kit = Length_kit * Vector_RadiationTrasmitted['tilt_R2ho_omb_keep_diff']
            
            totaleLen_NoKit = LenTop + LenBottom
            Rad_LatoSole_omb_dir_NoKit = totaleLen_NoKit * Vector_RadiationTrasmitted['t_R1ho_omb_dirhi']
            Rad_LatoSole_omb_diff_NoKit = totaleLen_NoKit * Vector_RadiationTrasmitted['tilt_R1ho_omb_diff']
          
            TotalRadiationOmb = (Rad_LatoOmbra_ombreg_dir + Rad_LatoOmbra_ombreg_diff +
                                Rad_LatoSole_omb_dir_shadowed + Rad_LatoSole_omb_diff_shadowed +
                                Rad_LatoSole_omb_dir_Kit + Rad_LatoSole_omb_diff_Kit +
                                Rad_LatoSole_omb_dir_NoKit + Rad_LatoSole_omb_diff_NoKit)
            ##ANTIGRANDINE
            #lato ombra antigrandine
            Rad_LatoOmbra_anti_dir = Ref_Length['totLe_ver'] * Vector_RadiationTrasmitted['t_r1ho_omb_diretta']
            Rad_LatoOmbra_anti_diff = Ref_Length['totLe_ver'] * Vector_RadiationTrasmitted['tilt_r1ho_anti_diffuso']
            
            #lato sole tratto in OMBRA - ANTI
            if Shadow_Length['shadowed len'] > Ref_Length['totLe_ver']:
                shadowRef = Ref_Length['totLe_ver']
                AntiRef = 0
            else:
                shadowRef = Shadow_Length['shadowed len']
                AntiRef = Ref_Length['totLe_ver'] - Shadow_Length['shadowed len']
            
            Rad_LatoSole_anti_dir_shadowed = shadowRef * Vector_RadiationTrasmitted['t_r1ho_anti_diretta']
            Rad_LatoSole_anti_diff_shadowed = shadowRef* Vector_RadiationTrasmitted['tilt_r1ho_anti_diffuso']
            
            Rad_LatoSole_anti_dir_NoShadowed = AntiRef * Vector_RadiationTrasmitted['t_R1ho_anti_dirhi']
            Rad_LatoSole_anti_diff_NoShadowed = AntiRef * Vector_RadiationTrasmitted['tilt_R1ho_anti_diffuso']
            
            TotalRadiationAnti = (Rad_LatoOmbra_anti_dir + Rad_LatoOmbra_anti_diff +
                                 Rad_LatoSole_anti_dir_shadowed + Rad_LatoSole_anti_diff_shadowed + 
                                 Rad_LatoSole_anti_dir_NoShadowed + Rad_LatoSole_anti_diff_NoShadowed)
            
            if TotalRadiationAnti == 0:
                ratio = 0
            else:
                ratio = TotalRadiationOmb/TotalRadiationAnti
                
            if Vector_RadiationTrasmitted['t_r1ho_omb_diretta'] < 0:
                Rad_LatoOmbra_ombreg_dir = 0
                Rad_LatoOmbra_ombreg_diff = 0
                Rad_LatoSole_omb_dir_shadowed = 0 
                Rad_LatoSole_omb_diff_shadowed = 0
                Rad_LatoSole_omb_dir_Kit = 0
                Rad_LatoSole_omb_diff_Kit = 0
                Rad_LatoSole_omb_dir_NoKit = 0
                Rad_LatoSole_omb_diff_NoKit = 0
                TotalRadiationOmb = 0
                Rad_LatoOmbra_anti_dir = 0 
                Rad_LatoOmbra_anti_diff = 0
                Rad_LatoSole_anti_dir_shadowed = 0 
                Rad_LatoSole_anti_diff_shadowed = 0
                Rad_LatoSole_anti_dir_NoShadowed = 0 
                Rad_LatoSole_anti_diff_NoShadowed = 0
            
            
            
            res =  {'indexV':Vector_RadiationTrasmitted['indexV'],
                    'Rad_LatoOmbra_ombreg_dir':Rad_LatoOmbra_ombreg_dir, 
                    'Rad_LatoOmbra_ombreg_diff':Rad_LatoOmbra_ombreg_diff, 
                    'Rad_LatoSole_omb_dir_shadowed':Rad_LatoSole_omb_dir_shadowed, 
                    'Rad_LatoSole_omb_diff_shadowed':Rad_LatoSole_omb_diff_shadowed, 
                    'Rad_LatoSole_omb_dir_Kit':Rad_LatoSole_omb_dir_Kit, 
                    'Rad_LatoSole_omb_diff_Kit':Rad_LatoSole_omb_diff_Kit, 
                    'Rad_LatoSole_omb_dir_NoKit':Rad_LatoSole_omb_dir_NoKit,
                    'Rad_LatoSole_omb_diff_NoKit':Rad_LatoSole_omb_diff_NoKit,
                    'TotalRad_Omb': TotalRadiationOmb, 
                    'Rad_LatoOmbra_anti_dir':Rad_LatoOmbra_anti_dir, 
                    'Rad_LatoOmbra_anti_diff':Rad_LatoOmbra_anti_diff,
                    'Rad_LatoSole_anti_dir_shadowed':Rad_LatoSole_anti_dir_shadowed, 
                    'Rad_LatoSole_anti_diff_shadowed':Rad_LatoSole_anti_diff_shadowed,
                    'Rad_LatoSole_anti_dir_NoShadowed':Rad_LatoSole_anti_dir_NoShadowed, 
                    'Rad_LatoSole_anti_diff_NoShadowed':Rad_LatoSole_anti_diff_NoShadowed,
                    'TotalRad_Anti': TotalRadiationAnti, 
                    'RefTotLen':Ref_Length['totLe_ver'], 
                    'shadowRef':shadowRef, 
                    'Length_kit':Length_kit, 
                    'LenBottom':LenBottom,
                    'LenTop':LenTop, 
                    'AntiRef':AntiRef,
                    'RatioOmb/Anti': ratio
                      }
            return res

#######################################################################################################
#######################################################################################################
#######################################################################################################
###### MAIN
##PARAMETRI GEOMETRIA COLTURA
distanceOfNetHDis_FromTree = 0.99
DiscontiDim = 1.32
angleTopCrw = 9
bottoCanTh = 4.9
bhz = 1.39
LChioma = 4.75  
minAngleShadow = 52
HtreeMAX = 3.8
TreeBaseAngle = 81 
inserzioneRami = 0.5
#PARAMETRI TRASMISSITIVITA RETI
# vectorFracionOMBREGG = [0.49, 0.94, 0.19,  0.86, 0.39]
vectorFracionOMBREGG = [0.49, 0.94, 0.19,  0.86, 0.39]
surf_par = [81, 90] #inclinazione ed azimut della chioma esposta
#########################################################################################################
## OGGETTI OUTPUT per POST-PROCESS
IrrCalculated = pd.DataFrame(columns=[
                                 'indexV',
                                 'radValue_diffuse', # radiazione diffusa Incidente
                                 'radValue_direct',  # radiazione diretta Incidente
                                 'R1ho_omb_dirhi', # Ombregg-rad diretta da ombreggiante
                                 'R2ho_omb_keep_dirhi', # Ombregg-rad diretta da keepInTouch
                                 'r1ho_omb_diffuso', # Ombregg-rad diffusa da ombreggiante  
                                 'R1ho_anti_dirhi', # AntiGr -rad diretta 
                                 'r1ho_anti_diffuso',# AntiGr -rad diffusa 
                                 'Direct_Rad_HorTilt_FRACTION', #Rapporto RadDiretta tra horiz to tilted
                
                                 't_R1ho_omb_dirhi', # tilter_Ombregg-rad diretta da ombreggiante
                                 't_R2ho_omb_keep_dirhi',# tilter_Ombregg-rad diretta da keepInTouch
                                 't_r1ho_omb_diretta', # Tilter_Ombregg-rad diretta da ombreggiante IN OMBRA
                                 'tilt_r1ho_omb_diffuso', # tilter_Ombregg-rad diffusa da ombreggiante IN OMBRA                                   
                                 'tilt_R1ho_omb_diff',# tilter_Ombregg-rad diffusa da ombreggiate
                                 'tilt_R2ho_omb_keep_diff',#tilter_Ombregg-rad diffusa da keep
                
                                 't_r1ho_anti_diretta', # Tilter_AntiGr-rad diretta IN OMBRA
                                 't_R1ho_anti_dirhi',# Tilter_AntiGr-rad diretta 
                                 'tilt_R1ho_anti_diffuso', # Tilter_AntiGr-rad diffusa IN OMBRA
                                 'tilt_r1ho_anti_diffuso', # tilter_AntiGr-rad diffusa IN OMBRA 

                                    ]) 
CrownRadCalculated = pd.DataFrame(columns=[
                                'indexV',
                                'Rad_LatoOmbra_ombreg_dir', 
                                'Rad_LatoOmbra_ombreg_diff', 
                                'Rad_LatoSole_omb_dir_shadowed', 
                                'Rad_LatoSole_omb_diff_shadowed', 
                                'Rad_LatoSole_omb_dir_Kit', 
                                'Rad_LatoSole_omb_diff_Kit', 
                                'Rad_LatoSole_omb_dir_NoKit',
                                'Rad_LatoSole_omb_diff_NoKit',
                                'TotalRad_Omb',
                                
                                'Rad_LatoOmbra_anti_dir', 
                                'Rad_LatoOmbra_anti_diff',
                                'Rad_LatoSole_anti_dir_shadowed', 
                                'Rad_LatoSole_anti_diff_shadowed',
                                'Rad_LatoSole_anti_dir_NoShadowed', 
                                'Rad_LatoSole_anti_diff_NoShadowed',
                                'TotalRad_Anti',
                                
                                'RefTotLen', 
                                'shadowRef', 
                                'Length_kit', 
                                'LenBottom',
                                'LenTop', 
                                'AntiRef',
                                'RatioOmb/Anti'
                                    ]) 
crown_measurement = pd.DataFrame(columns=['KiT length', 
                                          'len_bottom', 
                                          'len_top', 
                                          'totLe_ver', 
                                          'shadowed len' ]) 
#########################################################################################################
## CICLO per ogni row di matrice con dati di radiazione e meteo con simulati e rilevati da winet
Vlen = len(mergeRadiation['elevation'])-1
for i in range(0, Vlen, 1):
    print("###SUN ELEVATION",  str(frames['elevation'][i]))
    print("###DATE TIME",  str(frames.index[i]))
    SunElevationAngle = mergeRadiation['elevation'][i]   
    RadiationTrasmissitivityFRAMES =  pd.DataFrame(mergeRadiation.iloc[i,])
    RadCalc = radiation_fraction_model_tilted(RadiationTrasmissitivityFRAMES, vectorFracionOMBREGG, surf_par)
    Vector_RadiationTrasmitted = RadCalc.CalculateFraction_of_radiation()
    print("#Vector_RadiationTrasmitted")
    IrrCalculated = IrrCalculated.append(Vector_RadiationTrasmitted, ignore_index=True)
    print(Vector_RadiationTrasmitted)
    print("###")
    CalcLi = Beam_Surf_canopy_orchard_Calculator (SunElevationAngle, distanceOfNetHDis_FromTree, 
                                                  DiscontiDim, angleTopCrw, bottoCanTh, bhz, LChioma,
                                                  minAngleShadow,  HtreeMAX, TreeBaseAngle, inserzioneRami)
    Ref_Length = CalcLi.CalculateSurface()
    Shadow_Length = CalcLi.CalculateShadowLength()
    outputFrame = {'index':Vector_RadiationTrasmitted['indexV']} 
    outputFrame.update(Shadow_Length)
    outputFrame.update(Ref_Length)
    crown_measurement = crown_measurement.append(outputFrame, ignore_index=True)
    Calculate_Radiation = averagging_radiation(Ref_Length, Shadow_Length, Vector_RadiationTrasmitted)
    Values = Calculate_Radiation.calculating_values()
    CrownRadCalculated = CrownRadCalculated.append(Values, ignore_index=True)
    print("##################Start Calculation GEOMETRY:")
    print(Ref_Length)
    print(Shadow_Length)
    print("##################Start Calculation Radiation:")
    print(Values)
    print("##################End")
    
#########################################################################################################
#########################################################################################################
#########################################################################################################
##### POST-PROCESS RADIAZIONE
######
dateStart, dateEnd = ['2020-06-18', '2020-06-18']
# plot geometrie tratto ombra e Kit e anti
crown_measurement['datetime'] = pd.to_datetime(crown_measurement['index'])
crown_measurement_Index = crown_measurement.set_index('datetime')
crown_measurement_IndexSUBSET = crown_measurement_Index.loc[dateStart:dateEnd]
ax = crown_measurement_IndexSUBSET.plot()
ax.legend(loc='upper right'); 
######
# plot radiazione trasmessa da copertura 
IrrCalculated_Index = IrrCalculated.set_index("indexV")
IrrCalculated['datetime'] = pd.to_datetime(IrrCalculated['indexV'])
IrrCalculated_Index = IrrCalculated.set_index('datetime')
IrrCalculated_IndexSUBSET = IrrCalculated_Index.loc[dateStart:dateEnd]
IrrCalculated_IndexSUBSET['Direct_Rad_HorTilt_FRACTION'].plot(rot=45, fontsize = 8)
IrrCalculated_IndexSUBSET[['radValue_diffuse','radValue_direct']].plot()
IrrCalculated_IndexSUBSET[['radValue_diffuse','radValue_direct']].plot()
IrrCalculated_IndexSUBSET[['R1ho_omb_dirhi',
                           'R2ho_omb_keep_dirhi',
                           'r1ho_omb_diffuso', 
                           't_R1ho_omb_dirhi',
                           't_R2ho_omb_keep_dirhi',
                           'tilt_R1ho_omb_diff', 
                           'tilt_R2ho_omb_keep_diff',
                           'tilt_r1ho_omb_diffuso',
                           't_r1ho_omb_diretta']].plot(fontsize = 10)


IrrCalculated_IndexSUBSET[[
                          'R1ho_anti_dirhi',
                          'r1ho_anti_diffuso',
                          't_r1ho_anti_diretta', 
                          't_R1ho_anti_dirhi', 
                          'tilt_R1ho_anti_diffuso',
                          'tilt_r1ho_anti_diffuso']].plot(fontsize = 10)                                

######                               
# plot geometrie radiazione frazioni
CrownRadCalculated['datetime'] = pd.to_datetime(CrownRadCalculated['indexV'])
CrownRadCalculated_Index = CrownRadCalculated.set_index('datetime')
CrownRadCalculated_IndexSUBSET = CrownRadCalculated_Index.loc[dateStart:dateEnd]
ax = CrownRadCalculated_IndexSUBSET[['TotalRad_Omb','TotalRad_Anti']].plot()
ax.legend(loc='upper right');  

tOmb = CrownRadCalculated_IndexSUBSET['TotalRad_Omb']/(3.35*2)
tAnt = CrownRadCalculated_IndexSUBSET['TotalRad_Anti']/(3.35*2)
tRatio = tOmb/tAnt
CrownRadCalculated_IndexSUBSET['tomb_m2'] = tOmb 
CrownRadCalculated_IndexSUBSET['tAnt_m2'] = tAnt 
ax = CrownRadCalculated_IndexSUBSET[['tomb_m2','tAnt_m2' ]].plot()
ax.legend(loc='upper right');  

ax = CrownRadCalculated_IndexSUBSET[['RatioOmb/Anti']].plot()
ax.legend(loc='upper right');  

# plot della razione per ogni tratto moltiplicato per la rispettiva lunghezza
SelectedFeatures = ['Rad_LatoOmbra_ombreg_dir', 
                    'Rad_LatoOmbra_ombreg_diff', 
                    'Rad_LatoSole_omb_dir_shadowed', 
                    'Rad_LatoSole_omb_diff_shadowed', 
                    'Rad_LatoSole_omb_dir_Kit', 
                    'Rad_LatoSole_omb_diff_Kit', 
                    'Rad_LatoSole_omb_dir_NoKit',
                    'Rad_LatoSole_omb_diff_NoKit',
                    'Rad_LatoOmbra_anti_dir', 
                    'Rad_LatoOmbra_anti_diff',
                    'Rad_LatoSole_anti_dir_shadowed', 
                    'Rad_LatoSole_anti_diff_shadowed',
                    'Rad_LatoSole_anti_dir_NoShadowed', 
                    'Rad_LatoSole_anti_diff_NoShadowed'
                                ]  

SelectedFeatures1 = ['Rad_LatoOmbra_anti_dir', 
                    'Rad_LatoOmbra_anti_diff',
                    'Rad_LatoSole_anti_dir_shadowed', 
                    'Rad_LatoSole_anti_diff_shadowed',
                    'Rad_LatoSole_anti_dir_NoShadowed', 
                    'Rad_LatoSole_anti_diff_NoShadowed'
                                ]  

SelectedFeatures2 = ['Rad_LatoOmbra_ombreg_dir', 
                    'Rad_LatoOmbra_ombreg_diff', 
                    'Rad_LatoSole_omb_dir_shadowed', 
                    'Rad_LatoSole_omb_diff_shadowed', 
                    'Rad_LatoSole_omb_dir_Kit', 
                    'Rad_LatoSole_omb_diff_Kit', 
                    'Rad_LatoSole_omb_dir_NoKit',
                    'Rad_LatoSole_omb_diff_NoKit',
                                ]  



ax = CrownRadCalculated_IndexSUBSET[SelectedFeatures].plot()
ax.legend(loc='upper left', prop = {'size':5});

#PLOT radiazioni Calcolate su ANTI
ax = CrownRadCalculated_IndexSUBSET[SelectedFeatures1].plot()
ax.legend(loc='upper left', prop = {'size':8});
#PLOT radiazioni Calcolate su OMBREGG
ax = CrownRadCalculated_IndexSUBSET[SelectedFeatures2].plot()
ax.legend(loc='upper left', prop = {'size':8});

# fig, axes = plt.subplots(nrows = 1, ncols = 2)
# CrownRadCalculated_IndexSUBSET[SelectedFeatures1].plot(ax = axes [0])
# CrownRadCalculated_IndexSUBSET[SelectedFeatures2].plot(ax = axes [1])
# ax.legend(loc='upper left', prop = {'size':5});

#########################################################################################################
#########################################################################################################
##SUMMARY
CrownRadCalculated['datetime'] = pd.to_datetime(CrownRadCalculated['indexV'])
CrownRadCalculated_Index = CrownRadCalculated.set_index('datetime')
CrownRadCalculated_Index_daily = CrownRadCalculated_Index.resample('D').mean() 
CrownRadCalculated_Index_daily = CrownRadCalculated_Index_daily.loc['2020-06-01':'2020-09-30']

tOmb = CrownRadCalculated_Index_daily['TotalRad_Omb']/(3.35*2)
tAnt = CrownRadCalculated_Index_daily['TotalRad_Anti']/(3.35*2)
tRatio = tOmb/tAnt
CrownRadCalculated_Index_daily['tOmb_m2'] = tOmb 
CrownRadCalculated_Index_daily['tAnt_m2'] = tAnt 
ax = CrownRadCalculated_Index_daily[['tOmb_m2','tAnt_m2' ]].plot()
ax.legend(loc='upper right');  

ax = CrownRadCalculated_Index_daily[['RatioOmb/Anti']].plot()
ax.legend(loc='upper right');  

(CrownRadCalculated_Index_daily[['RatioOmb/Anti']]).mean()
Vector = CrownRadCalculated_Index_daily[['RatioOmb/Anti']]
CrownRadCalculated_Index_daily[['RatioOmb/Anti']].mean()
Vector.plot()
#########################################################################################################
#########################################################################################################
##### integrazioen con CALCOLO ET0
# input: leggi file con radiazione da Winet
dateparser = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
mainpath = ./'
csvFilePath_davis = mainpath + 'MeteoDAVIS_17092020.txt'
meteo_davis = pd.read_csv(csvFilePath_davis, sep='\t',  decimal = '.', encoding= 'unicode_escape')        
meteo_davis['DateTime'] = pd.to_datetime(meteo_davis['Date & Time'])
meteo_davis_Index = meteo_davis.set_index('DateTime')
meteo_davis_Index_TZ = meteo_davis_Index.tz_localize('CET')
meteo_davis_Index_RADIATION = pd.merge(meteo_davis_Index_TZ, CrownRadCalculated_Index[['TotalRad_Anti', 'TotalRad_Omb']], how='left', left_index=True, right_index=True)
meteo_davis_Index_RADIATION  = meteo_davis_Index_RADIATION .loc['2020-06-01':'2020-09-30']
# output per post process
et0 = pd.DataFrame(columns=['datetime', 'et0_anti', 'et0_omb']) 
Vlen = len(meteo_davis_Index_RADIATION['Date & Time'])-2
#########################################################################################################
#MAIN et0
for i in range(0, Vlen, 1):
    vindex = meteo_davis_Index_RADIATION['Date & Time'][i] 
    temp_anti = meteo_davis_Index_RADIATION['S3O_antigrandine_Temp - °C'][i]   
    vento_anti = meteo_davis_Index_RADIATION['S3O_antigrandine_Wind Speed - m/s'][i] 
    Rad_anti = meteo_davis_Index_RADIATION['TotalRad_Anti'][i]/(3.35*2) 
    hum_anti = meteo_davis_Index_RADIATION['S3O_antigrandine_Hum - %'][i] 
    pressure_anti = meteo_davis_Index_RADIATION['S3O_antigrandine_Barometer - mb'][i] 
    et0_anti = CalculateET0(temp_anti, vento_anti, Rad_anti, hum_anti, pressure_anti)
    if et0_anti < 0:
        et0_anti = 0
    temp_omb = meteo_davis_Index_RADIATION['S3O_ombreggiante_Temp - °C'][i]
    vento_omb = meteo_davis_Index_RADIATION['S3O_ombreggiante_Wind Speed - m/s'][i] 
    Rad_omb = meteo_davis_Index_RADIATION['TotalRad_Omb'][i]/(3.35*2) 
    hum_omb = meteo_davis_Index_RADIATION['S3O_ombreggiante_Hum - %'][i] 
    pressure_omb = meteo_davis_Index_RADIATION['S3O_ombreggiante_Barometer - mb'][i] 
    et0_omb = CalculateET0(temp_omb, vento_omb, Rad_omb, hum_omb, pressure_omb)
    if et0_omb < 0:
        et0_omb = 0
    vec = {'datetime': vindex, 'et0_anti' : et0_anti, 'et0_omb' : et0_omb }
    et0 = et0.append(vec, ignore_index=True)
#########################################################################################################
#########################################################################################################
#########################################################################################################
##### POST-PROCESS ET0
et0['datetime'] = pd.to_datetime(et0['datetime'])
et0_Index = et0.set_index('datetime')
et0_Index_daily = et0_Index.resample('D').sum() 
et0_Index_daily

ax = et0_Index_daily.plot()
ax.legend(loc='upper right');
 
etRatio = et0_Index_daily['et0_omb']/ et0_Index_daily['et0_anti']
etRatio.mean()
ax = etRatio.plot()
ax.legend(loc='upper right');
 

