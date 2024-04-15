
'''# transmission around a 90-degree waveguide bend in 2d
import cmath
import matplotlib.pyplot as plt
import math
import meep as mp
import numpy as np
from matplotlib import pyplot as plt

resolution = 150  # pixels/unit length (1 um)

sy = 0.4
sz = 0.4
sx = 6
sl = sy * 0.7
hx = 0.1
hy = sy
hz = sz
e_field = mp.Ez
dpml = 0.4
height = 0.050
sr = 0.06
#sr = 0.5
#sr = 0.6
#height = 0.01
#height = 0.1


theta_deg = 0
eps_averaging = False
eig_src = True

um_scale = 1.0
eV_um_scale = um_scale/1.23984193
Ag_plasma_frq = 9.01 * eV_um_scale

Ag_f = [0.845,
        0.065,
        0.124,
        0.011,
        0.840,
        5.646]

Ag_frq = [1E-10,
          0.816 * eV_um_scale,
          4.481 * eV_um_scale,
          8.185 * eV_um_scale,
          9.083 * eV_um_scale,
          20.29 * eV_um_scale]

Ag_gam = [0.048 * eV_um_scale,
          3.886 * eV_um_scale,
          0.452 * eV_um_scale,
          0.065 * eV_um_scale,
          0.916 * eV_um_scale,
          2.419 * eV_um_scale]

Ag_sig = [Ag_f[index] * math.sqrt(Ag_plasma_frq) / math.sqrt(Ag_frq[index]) for index in range(len(Ag_f))]

#wvl_min = 0.4
wvl_min = 0.43
wvl_max = 0.8
fmin = 1/wvl_max
fmax = 1/wvl_min
fcen = 0.5 * (fmin + fmax)
df = fmax - fmin
boundary_layers = [mp.PML(thickness=dpml, direction=mp.X)]
#boundary_layers=[mp.Absorber(thickness=2)]

x0 = mp.Vector3(dpml - 0.5 * sx, 0, 0)
theta_rad = theta_deg * math.pi/180
kdir = mp.Vector3(math.cos(theta_rad), math.sin(theta_rad), 0)
k = (2 * math.pi * fcen) * kdir

k_point = mp.Vector3(0, 0, 0)

cell_size = mp.Vector3(sx, sy, sz)

pw_amp = lambda x: cmath.exp(1j*k.dot(x + x0))

sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen, fwidth=df),
                              center=mp.Vector3(dpml + sl - 0.5 * sx, 0, 0),
                              size=mp.Vector3(0, sy, sz),
                              direction=mp.Z,
                              amp_func=pw_amp,
                              #eig_kpoint=k_point
                              )]
'''
'''
sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df),
                     component=e_field,
                     center=mp.Vector3(dpml + sl - 0.5 * sx, 0, 0),
                     size=mp.Vector3(0, sy, sz),
                     amp_func=pw_amp)]
'''
'''


geometry = []

geometry.append(mp.Block(material=mp.air, center=mp.Vector3(0, 0, 0), size=cell_size))


sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    geometry=geometry,
                    sources=sources,
                    dimensions=3,
                    k_point=k_point
                    )

nfreq = 400
refl_fr = mp.FluxRegion(center=mp.Vector3(-(- (sx / 2) + (dpml + sy)), 0, 0), size=mp.Vector3(0, sy, sz))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

sim.run(
    until_after_sources = mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(- (sx / 2) + (dpml + 0.1), 0, 0), 1e-3)
)

#sim.run(until=2000)

empty_flux = mp.get_fluxes(refl)
empty_data = sim.get_flux_data(refl)
sim.reset_meep()

Ag = mp.Medium(epsilon=1.0,  E_susceptibilities=[mp.DrudeSusceptibility(frequency=Ag_frq[0], gamma=Ag_gam[0], sigma=Ag_sig[0]),
                                                 mp.LorentzianSusceptibility(frequency=Ag_frq[1], gamma=Ag_gam[1], sigma=Ag_sig[1]),
                                                 mp.LorentzianSusceptibility(frequency=Ag_frq[2], gamma=Ag_gam[2], sigma=Ag_sig[2]),
                                                 mp.LorentzianSusceptibility(frequency=Ag_frq[3], gamma=Ag_gam[3], sigma=Ag_sig[3]),
                                                 mp.LorentzianSusceptibility(frequency=Ag_frq[4], gamma=Ag_gam[4], sigma=Ag_sig[4]),
                                                 mp.LorentzianSusceptibility(frequency=Ag_frq[5], gamma=Ag_gam[5], sigma=Ag_sig[5])
                                                 ]
               )


geometry = []

geometry.append(mp.Block(material=mp.air, center=mp.Vector3(0, 0, 0), size=cell_size))

geometry.append(mp.Cylinder(material=Ag,
                                center=mp.Vector3(0, 0, 0),
                                radius=sr,
                                height=height,
                                axis=mp.Vector3(1, 0, 0)))
# rotation angle of sides relative to Y axis (degrees)

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    geometry=geometry,
                    sources=sources,
                    dimensions=3,
                    k_point=k_point
                    )


refl = sim.add_flux(fcen, df, nfreq, refl_fr)
sim.load_minus_flux_data(refl, empty_data)


sim.run(
    until_after_sources = mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(- (sx / 2) + (dpml + 0.1), 0, 0), 1e-3)
)

#sim.run(until=2000)


refl_flux = mp.get_fluxes(refl)
R_meep = -1*np.divide(refl_flux, empty_flux)

print("R_meep")
print(list(R_meep))
print("")
print("refl_flux", refl_flux)
print(list(refl_flux))
print("")

freqs = mp.get_flux_freqs(refl)
wvls = np.divide(1, freqs)

plt.plot(wvls, empty_flux, label = "empty_flux")
plt.plot(wvls, R_meep, label = "R_meep")
plt.legend()

plt.show()'''
tick1 = 1
tick2 = 0
if tick1 == 1 & tick2 == 1:
    print("success")
else:
    print("fail")