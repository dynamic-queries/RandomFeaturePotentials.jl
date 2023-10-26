using MAT
using Plots

filename = "data/qm7.mat"
file = matread(filename)
R = file["R"]
Z = file["Z"]
E = file["T"]'




# Angles
angles = compute_angles()

