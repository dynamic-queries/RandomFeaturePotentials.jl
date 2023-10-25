include("../../../approximators/simple.jl")
using HDF5
using Plots
using Measures
using TSne

begin
    filename = "data/toluene.hdf5"
    file = h5open(filename)
    num_features = 144
    cm = read(file["CM"])
    scm = read(file["SCM"])
    ecm = read(file["ECM"])
    sicm = read(file["SICM"])
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

cm_e = tsne(Array(cm'),2,0,3000,10)
scm_e = tsne(Array(scm'),2,0,3000,10)
ecm_e = tsne(Array(ecm'),2,0,3000,10)
sicm_e = tsne(Array(sicm'),2,0,3000,10)
f1 = scatter(cm_e[:,1],cm_e[:,2],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="CM",gridlinewidth=2.0,minorgrid=true)
f2 = scatter(scm_e[:,1],scm_e[:,2],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="SCM",gridlinewidth=2.0,minorgrid=true)
f3 = scatter(ecm_e[:,1],ecm_e[:,2],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="ECM",gridlinewidth=2.0,minorgrid=true)
f4 = scatter(sicm_e[:,1],sicm_e[:,2],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="SICM",gridlinewidth=2.0,minorgrid=true)
g1 = plot(f1,f2,f3,f4,size=(1000,1000))
savefig("experiments/permutation/toluene/embedding.png")