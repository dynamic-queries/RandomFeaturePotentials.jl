include("../approximators/structure.jl")
include("utils.jl")

begin
    filename = "data/benzene2017.hdf5"
    file = h5open(filename)
    R = read(file["R"])
    E = read(file["E"])
    F = read(file["F"])

    xtrain = R[:,1:2700]
    ytrain = E[:,1:2700]
    ztrain = F[:,1:2700]

    xtest = R[:,2701:end]
    ytest = E[:,2701:end]
    ztest = F[:,2701:end]

    layers = [500]
    lam = 1e-8
    f1, f2 = predict(layers,lam,"Benzene",FiniteDifference)
end