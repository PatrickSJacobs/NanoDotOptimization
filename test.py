from mpi4py import MPI
import meep as mp
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(mp.with_mpi())