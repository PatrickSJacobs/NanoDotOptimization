#----------------------------------------
sr = 0.06748289135799125
ht = 0.09591995969084945
sy = 0.3849657827159825
theta_deg = 0.0
#----------------------------------------

import sys
no_metal = sys.argv[0]
#no_metal = True

# Parameters

sz = sy
sx = 6
sl = sy * 0.7

hx = 0.1
hy = sy
hz = sz

import meep as mp
import numpy as np

efield = mp.Ez
dpml = 0.4

resolu = 100
nfreq = 400

wvl_min = 0.4
wvl_max = 0.8
fmin = 1 / wvl_max
fmax = 1 / wvl_min
fcen = 0.5 * (fmin + fmax)
df = fmax - fmin

theta_rad = theta_deg * mp.pi / 180
kdir = mp.Vector3(np.cos(theta_rad), np.sin(theta_rad), 0)
k = kdir.scale(2 * mp.pi * fcen)


silver_f = mp.Medium(
    epsilon=1,
    E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.000e-20, gamma=0.00615, sigma=4.444e+41),
        mp.LorentzianSusceptibility(frequency=0.10453, gamma=0.49782, sigma=7.92470),
        mp.LorentzianSusceptibility(frequency=0.57404, gamma=0.05790, sigma=0.50133),
        mp.LorentzianSusceptibility(frequency=1.04854, gamma=0.00833, sigma=0.01333),
        mp.LorentzianSusceptibility(frequency=1.16358, gamma=0.11734, sigma=0.82655),
        mp.LorentzianSusceptibility(frequency=2.59926, gamma=0.30989, sigma=1.11334)
    ]
)

eV_um_scale = 1.0  # Assuming the scaling factor
Ag_plasma_frq = 9.01 * eV_um_scale

Ag_f0 = 0.845
Ag_frq0 = 1e-10
Ag_gam0 = 0.048 * eV_um_scale
Ag_sig0 = (Ag_f0 * (Ag_plasma_frq ** 2)) / (Ag_frq0 ** 2)

Ag_f1 = 0.065
Ag_frq1 = 0.816 * eV_um_scale  # 1.519 um
Ag_gam1 = 3.886 * eV_um_scale
Ag_sig1 = (Ag_f1 * (Ag_plasma_frq ** 2)) / (Ag_frq1 ** 2)

Ag_f2 = 0.124
Ag_frq2 = 4.481 * eV_um_scale  # 0.273 um
Ag_gam2 = 0.452 * eV_um_scale
Ag_sig2 = (Ag_f2 * (Ag_plasma_frq ** 2)) / (Ag_frq2 ** 2)

Ag_f3 = 0.011
Ag_frq3 = 8.185 * eV_um_scale  # 0.152 um
Ag_gam3 = 0.065 * eV_um_scale
Ag_sig3 = (Ag_f3 * (Ag_plasma_frq ** 2)) / (Ag_frq3 ** 2)

Ag_f4 = 0.840
Ag_frq4 = 9.083 * eV_um_scale  # 0.137 um
Ag_gam4 = 0.916 * eV_um_scale
Ag_sig4 = (Ag_f4 * (Ag_plasma_frq ** 2)) / (Ag_frq4 ** 2)

Ag_f5 = 5.646
Ag_frq5 = 20.29 * eV_um_scale  # 0.061 um
Ag_gam5 = 2.419 * eV_um_scale
Ag_sig5 = (Ag_f5 * (Ag_plasma_frq ** 2)) / (Ag_frq5 ** 2)

Ag = mp.Medium(epsilon=1.0,
               E_susceptibilities=[
                   mp.DrudeSusceptibility(frequency=Ag_frq0, gamma=Ag_gam0, sigma=Ag_sig0),
                   mp.LorentzianSusceptibility(frequency=Ag_frq1, gamma=Ag_gam1, sigma=Ag_sig1),
                   mp.LorentzianSusceptibility(frequency=Ag_frq2, gamma=Ag_gam2, sigma=Ag_sig2),
                   mp.LorentzianSusceptibility(frequency=Ag_frq3, gamma=Ag_gam3, sigma=Ag_sig3),
                   mp.LorentzianSusceptibility(frequency=Ag_frq4, gamma=Ag_gam4, sigma=Ag_sig4),
                   mp.LorentzianSusceptibility(frequency=Ag_frq5, gamma=Ag_gam5, sigma=Ag_sig5)
               ])

# Geometry
geometry_lattice = mp.Lattice(size=mp.Vector3(sx, sy, sz))



geometry = []
if no_metal:
    geometry.append(mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, sy, sz), material=mp.air))
else:
    geometry.append(mp.Cylinder(center=mp.Vector3(0, 0, 0), height=ht, radius=sr, axis=mp.Vector3(1, 0, 0), material=Ag))

# Source
def pw_amp(k, x0):
    def func(x):
        return np.exp(1j * mp.Vector3.dot(k, x + x0))
    return func

sources = [
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=efield,
        center=mp.Vector3(-0.5 * sx + dpml + sl, 0, 0),
        size=mp.Vector3(0, sy, sz),
        amp_func=pw_amp(k, mp.Vector3(-0.5 * sx + dpml, 0, 0))
    )
]

# Simulation setup
sim = mp.Simulation(
    cell_size=mp.Vector3(sx, sy, sz),
    geometry=geometry,
    sources=sources,
    resolution=resolu,
    boundary_layers=[mp.PML(thickness=dpml, direction=mp.X)],
    k_point=mp.Vector3(0, 0, 0)
)

# Flux regions
trans_flux = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0.5 * sx - dpml - 0.1, 0, 0), size=mp.Vector3(0, sy, sz)))
refl_flux = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(-0.5 * sx + dpml + sy, 0, 0), size=mp.Vector3(0, sy, sz)))

# Load minus flux if metal is present
if no_metal == False:
    sim.load_minus_flux("refl-flux", refl_flux)

# Run simulation

'''
sim.run(
    mp.at_beginning(mp.output_epsilon),
    mp.to_appended("flux", mp.in_volume(mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx - 2 * (dpml + 0.1), sy, sz)), mp.output_efield_z)),
    until_after_sources=500,
    until=sim.stop_when_fields_decayed(50, efield, mp.Vector3(0.5 * sx - dpml - 0.1, 0, 0), 1e-3)
)
'''

center_point = mp.Vector3(-0.5 * sx + dpml + 0.1, 0, 0)
size_vec = mp.Vector3(sx - 2 * (dpml + 0.1), sy, sz)

sim.run(
mp.at_beginning(
        mp.in_volume(
            mp.Volume(center=mp.Vector3(0, 0, 0), size=size_vec),
            mp.output_epsilon
        )
    ),
    mp.at_end(
        mp.in_volume(
            mp.Volume(center=mp.Vector3(0, 0, 0), size=size_vec),
            mp.output_efield_z
        )
    ),
    until_after_sources=mp.stop_when_fields_decayed(
        dt=50, c=mp.Ez, pt=center_point, decay_by=1e-3
    ),
    until=500
)


# Save flux if no metal
if no_metal == True:
    sim.save_flux("refl-flux", refl_flux)

# Display fluxes
#sim.display_fluxes(trans_flux, refl_flux)