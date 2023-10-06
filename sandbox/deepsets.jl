using HDF5
using Plots

data = h5open("data/paracetamol_dft.hdf5","r")
input = read(data["CM"])
output = read(data["E"])

plot(input,legend=false)