include("../approximators/structure.jl")
include("utils.jl")

begin
    R = reshape(-2pi:1e-2:2pi,1,:)
    E = reshape(sin.(12*R[:]),1,:)
    F = reshape(-12*cos.(12*R[:]),1,:)

    xtrain = R[:,10:end-10]
    ytrain = E[:,10:end-10]
    ztrain = F[:,10:end-10]

    xtest = R
    ytest = E
    ztest = F

    layers = [2000]
    lam = 1e-8
    f1, f2 = predict(layers,lam,"Sinusoid",FiniteDifference)
    plot(f1,f2,size=(800,300))
end