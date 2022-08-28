* EE214 HSpice models, Boris Murmann, January 2010
* MOS model Updates:
* Set CJGATE=0
* Set source/drain resistance parameters to zero (RDSW, RSH)
* Added idealized pnp model
* Deleted nf and nr in npn214 model
* Added dwell device

* SiGe npn unit transistor model, emitter area=0.7um^2
.model npn214 npn
+ level=1 tref=25 is=.032f bf=300 br=2 vaf=90
+ cje=6.26f vje=.8 mje=.4 cjc=3.42f vjc=.6 mjc=.33
+ re=2.5 rb=25 rc=60 tf=563f tr=10p
+ xtf=200 itf=80m ikf=12m ikr=10.5m nkf=0.9 

* idealized npn and pnp models
.model npnideal npn is=.032f  bf=300
.model pnpideal pnp is=.032f  bf=300


* 0.18um CMOS models (nominal process)
.MODEL nmos214 nmos (
+acm     = 3              hdif    = 0.32e-6        LEVEL   = 49
+VERSION = 3.1            TNOM    = 27             TOX     = 4.1E-9
+XJ      = 1E-7           NCH     = 2.3549E17      VTH0    = 0.3618397
+K1      = 0.5916053      K2      = 3.225139E-3    K3      = 1E-3
+K3B     = 2.3938862      W0      = 1E-7           NLX     = 1.776268E-7
+DVT0W   = 0              DVT1W   = 0              DVT2W   = 0
+DVT0    = 1.3127368      DVT1    = 0.3876801      DVT2    = 0.0238708
+U0      = 256.74093      UA      = -1.585658E-9   UB      = 2.528203E-18
+UC      = 5.182125E-11   VSAT    = 1.003268E5     A0      = 1.981392
+AGS     = 0.4347252      B0      = 4.989266E-7    B1      = 5E-6
+KETA    = -9.888408E-3   A1      = 6.164533E-4    A2      = 0.9388917
+RDSW    = 0              PRWG    = 0.5            PRWB    = -0.2
+WR      = 1              WINT    = 0              LINT    = 1.617316E-8
+XL      = 0              XW      = -1E-8          DWG     = -5.383413E-9
+DWB     = 9.111767E-9    VOFF    = -0.0854824     NFACTOR = 2.2420572
+CIT     = 0              CDSC    = 2.4E-4         CDSCD   = 0
+CDSCB   = 0              ETA0    = 2.981159E-3    ETAB    = 9.289544E-6
+DSUB    = 0.0159753      PCLM    = 0.7245546      PDIBLC1 = 0.1568183
+PDIBLC2 = 2.543351E-3    PDIBLCB = -0.1           DROUT   = 0.7445011
+PSCBE1  = 8E10           PSCBE2  = 1.876443E-9    PVAG    = 7.200284E-3
+DELTA   = 0.01           RSH     = 0              MOBMOD  = 1
+PRT     = 0              UTE     = -1.5           KT1     = -0.11
+KT1L    = 0              KT2     = 0.022          UA1     = 4.31E-9
+UB1     = -7.61E-18      UC1     = -5.6E-11       AT      = 3.3E4
+WL      = 0              WLN     = 1              WW      = 0
+WWN     = 1              WWL     = 0              LL      = 0
+LLN     = 1              LW      = 0              LWN     = 1
+LWL     = 0              CAPMOD  = 2              XPART   = 1
+CGDO    = 4.91E-10       CGSO    = 4.91E-10       CGBO    = 1E-12
+CJ      = 9.652028E-4    PB      = 0.8            MJ      = 0.3836899
+CJSW    = 2.326465E-10   PBSW    = 0.8            MJSW    = 0.1253131
+CJSWG   = 3.3E-10        PBSWG   = 0.8            MJSWG   = 0.1253131
+CF      = 0              PVTH0   = -7.714081E-4   PRDSW   = -2.5827257
+PK2     = 9.619963E-4    WKETA   = -1.060423E-4   LKETA   = -5.373522E-3
+PU0     = 4.5760891      PUA     = 1.469028E-14   PUB     = 1.783193E-23
+PVSAT   = 1.19774E3      PETA0   = 9.968409E-5    PKETA   = -2.51194E-3
+cjgate  = 0              nlev    = 3              kf      = 0.5e-25)
*
.MODEL pmos214 pmos (
+acm     = 3              hdif    = 0.32e-6        LEVEL   = 49
+VERSION = 3.1            TNOM    = 27             TOX     = 4.1E-9
+XJ      = 1E-7           NCH     = 4.1589E17      VTH0    = -0.374776
+K1      = 0.5861046      K2      = 0.0264302      K3      = 0
+K3B     = 9.1168119      W0      = 1E-6           NLX     = 1.332241E-7
+DVT0W   = 0              DVT1W   = 0              DVT2W   = 0
+DVT0    = 0.6148668      DVT1    = 0.2213746      DVT2    = 0.1
+U0      = 103.636208     UA      = 1.043424E-9    UB      = 1E-21
+UC      = -1E-10         VSAT    = 1.29059E5      A0      = 1.5418178
+AGS     = 0.3123693      B0      = 6.199145E-7    B1      = 1.634457E-6
+KETA    = 0.0313547      A1      = 0.8            A2      = 0.3746405
+RDSW    = 293.751926     PRWG    = 0.5            PRWB    = 0.5
+WR      = 1              WINT    = 0              LINT    = 3.111886E-8
+XL      = 0              XW      = -1E-8          DWG     = -2.715764E-8
+DWB     = 4.525318E-9    VOFF    = -0.0831119     NFACTOR = 1.933495
+CIT     = 0              CDSC    = 2.4E-4         CDSCD   = 0
+CDSCB   = 0              ETA0    = 0.0859596      ETAB    = -0.0520269
+DSUB    = 0.8778821      PCLM    = 2.9202527      PDIBLC1 = 1.333525E-4
+PDIBLC2 = 0.0334217      PDIBLCB = -9.294449E-4   DROUT   = 9.986813E-4
+PSCBE1  = 3.206395E9     PSCBE2  = 9.279348E-10   PVAG    = 15
+DELTA   = 0.01           RSH     = 7.5            MOBMOD  = 1
+PRT     = 0              UTE     = -1.5           KT1     = -0.11
+KT1L    = 0              KT2     = 0.022          UA1     = 4.31E-9
+UB1     = -7.61E-18      UC1     = -5.6E-11       AT      = 3.3E4
+WL      = 0              WLN     = 1              WW      = 0
+WWN     = 1              WWL     = 0              LL      = 0
+LLN     = 1              LW      = 0              LWN     = 1
+LWL     = 0              CAPMOD  = 2              XPART   = 1
+CGDO    = 6.57E-10       CGSO    = 6.57E-10       CGBO    = 1E-12
+CJ      = 1.186426E-3    PB      = 0.8350261      MJ      = 0.4022229
+CJSW    = 1.924495E-10   PBSW    = 0.8            MJSW    = 0.3353329
+CJSWG   = 4.22E-10       PBSWG   = 0.8            MJSWG   = 0.3353329
+CF      = 0              PVTH0   = 1.522756E-3    PRDSW   = 0.7164396
+PK2     = 1.500815E-3    WKETA   = 0.0277401      LKETA   = -1.794554E-3
+PU0     = -0.9454674     PUA     = -4.22507E-11   PUB     = 1E-21
+PVSAT   = -50            PETA0   = 1.003159E-4    PKETA   = -3.89914E-3
+cjgate  = 0              nlev    = 3              kf      = 0.25e-25)     )
*

* well-to-substrate diode
* example instantiation (area = 10um*10um = 100pm^2)
*    (anode)  (cathode) (model) (area)
* d1 sub_node well_node  dwell   100p 
.model dwell d cj0=2e-4 is=1e-5 m=0.5

* ideal balun
* example instantiation
* x1 vdm vcm vp vm balun 
.subckt balun vdm vcm vp vm
e1 vp vcm transformer vdm 0 2
e2 vcm vm transformer vdm 0 2
.ends balun

* differential to single ended conversion
* example instantiation
* x1 vd vc vp vm diffsense 
.subckt diffsense vd vc vp vm
e1 vminv  0 vm 0      -1
e2 vc     0 vp vminv  0.5
e3 vd     0 vp vm     1
* These resistors help suppress "singular node" warnings
rdum1 vd  0   1gig
rdum2 vc  0   1gig
.ends diffsense

* single ended to differential conversion
* example instantiation
* x1 vp vm vd vc diffdrive
.subckt diffdrive vp vm vd vc
e1 vp vc vd 0  0.5
e2 vm vc vd 0  -0.5
.ends diffdrive
