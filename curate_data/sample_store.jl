using StatsBase
using NPZ
using HDF5
using LinearAlgebra
using Pkg
include("descriptors.jl")


function curate(filename, target_name)
    R = npzread(string(filename, "R.npy")); size(R)
    Z = npzread(string(filename, "z.npy")); size(Z)
    E = npzread(string(filename, "E.npy")); size(E)
    F = npzread(string(filename, "F.npy")); size(F)
    s = size(R)
    nsamples = 3_000
    idx = rand(1_000:1_000+nsamples,nsamples)
    R_t = reshape(permutedims(R[idx,:,:],(2,3,1)),s[2]*s[3],:)
    F_t = reshape(permutedims(F[idx,:,:],(2,3,1)),s[2]*s[3],:)
    E_t = reshape(E'[:,idx],1,:)

    CM = coulomb_matrix(Z,reshape(R_t,s[2],s[3],:))
    ECM = eigen_descriptors(CM)
    SICM = singular_descriptors(CM)
    SCM = sorted_descriptors(CM)

    # DCM = derivative_coulomb_matrix(Z,R_t)

    CM_t = reshape(CM,s[2]*s[2],:)
    ECM_t = ECM
    SCM_t = reshape(SCM,s[2]*s[2],:)
    SICM_t = SICM
    # DCM_t = DCM

    
    file = h5open(target_name,"w")
    file["R"] = R_t
    file["E"] = E_t
    file["F"] = F_t
    file["CM"] = CM_t
    # file["DCM"] = DCM_t
    file["ECM"] = ECM_t
    file["SCM"] = SCM_t
    file["SICM"] = SICM_t
    close(file)
end

datasets = ["benzene2017","ethanol","malonaldehyde","naphthalene","salicylic","toluene","uracil","paracetamol_dft"]
for set in datasets
    filename = "data/MD17/md17_$set/"
    target_name = "MOL_RFNN/data/$set.hdf5"
    curate(filename,target_name)
end
