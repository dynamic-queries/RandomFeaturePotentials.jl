include("../../../approximators/domain_decomp.jl")
using HDF5
using Plots

begin
    filename = "data/benzene2017.hdf5"
    file = h5open(filename)
    R = read(file["CM"])
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end


begin
    xtrain = R[:,1:2700]
    ytrain = E[:,1:2700]
    xtest = R[:,2701:end]
    ytest = E[:,2701:end]
    layers = 800*ones(Int,12)
    heuristic = Uniform
    lam = 1e-8
    s1 = 1
    s2 = 0
    feature_model = LinearFeatureModel(s1,s2)
    es = RFNN_DD(layers,feature_model,multiplicity=1,activation=gelu)
    m = es(xtrain,ytrain,heuristic,lam)
    
    @show layers[1], lam

    train = (xtrain,ytrain)
    fig1,ma1,rms1 = validate(m,train)
    @show ma1, rms1

    test = (xtest,ytest)
    fig2,ma2,rms2 = validate(m,test)
    @show ma2, rms2

    display(plot(fig1,fig2,size=(900,300)))
end 


begin
    xtrain = R[:,1:2700]
    ytrain = F[:,1:2700]
    xtest = R[:,2701:end]
    ytest = F[:,2701:end]
    layers = 800*ones(Int,12)
    heuristic = Uniform
    lam = 1e-8
    s1 = 1
    s2 = 0
    feature_model = LinearFeatureModel(s1,s2)
    es = RFNN_DD(layers,feature_model,multiplicity=1,activation=gelu)
    m = es(xtrain,ytrain,heuristic,lam)
    
    @show layers[1], lam

    train = (xtrain,ytrain)
    fig1,ma1,rms1 = validate(m,train)
    @show ma1, rms1

    test = (xtest,ytest)
    fig2,ma2,rms2 = validate(m,test)
    @show ma2, rms2

    display(plot(fig1,fig2,size=(900,300)))
end 
