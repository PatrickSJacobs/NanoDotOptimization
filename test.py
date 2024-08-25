from mpi4py import MPI
import meep as mp
print(mp.with_mpi())
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#----------------------------------------
sr = 0.1244229976060725
ht = 0.08467103656737388
sy = 0.498845995212145
theta_deg = 0.0
#----------------------------------------

no_metal = True

# Parameters

sz = sy
sx = 6
sl = sy * 0.7

hx = 0.1
hy = sy
hz = sz

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

comm.Barrier()

# Load minus flux if metal is present
if not no_metal:
    if rank == 0:
        # Rank 0 loads the minus flux data from the file
        minus_flux_data = sim.load_minus_flux("refl-flux", refl_flux)
    else:
        # Other ranks initialize minus_flux_data as None
        minus_flux_data = None

    # Broadcast the loaded minus flux data to all processes
    minus_flux_data = comm.bcast(minus_flux_data, root=0)

    # Apply the minus flux data to all ranks
    sim.load_minus_flux("refl-flux", refl_flux, flux_data=minus_flux_data)

comm.Barrier()

# Run simulation
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

# Ensure all processes have completed their tasks
comm.Barrier()

all_done = np.array(1, dtype='i')
all_done = comm.allreduce(all_done, op=MPI.MIN)

if all_done == 1:
    # Gather flux data on rank 0
    flux_data = comm.gather(sim.get_flux_data(refl_flux), root=0)

    # Only rank 0 writes the flux data
    if rank == 0:
        # Save flux if no metal
        if no_metal:
            # Save the flux data on rank 0
            sim.save_flux("refl-flux", refl_flux)
        # Display fluxes
        sim.display_fluxes(trans_flux, refl_flux)

# Final Barrier to ensure all processes are synchronized before exiting
comm.Barrier()
