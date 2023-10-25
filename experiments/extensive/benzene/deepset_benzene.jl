include("../../../approximators/domain_decomp.jl")
using HDF5
using Plots

begin
    filename = "data/benzene2017.hdf5"
    file = h5open(filename)
    R = read(file["CM"])
    num_features =  625
    R,f = deepset(read(file["CM"]),num_features,0.5)
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

plot(R,legend=false)

begin
    xtrain = R[:,1:2700]
    ytrain = E[:,1:2700]
    xtest = R[:,2701:end]
    ytest = E[:,2701:end]
    layers = 120*ones(Int,25)
    heuristic = Uniform
    lam = 1e-8
    s1 = 2*log(1.5)
    s2 = log(1.5)
    feature_model = LinearFeatureModel(s1,s2)
    es = RFNN_DD(layers,feature_model,multiplicity=1,activation=tanh)
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
    layers = 150*ones(Int,25)
    heuristic = Uniform
    lam = 1e-8
    s1 = 2*log(1.5)
    s2 = log(1.5)
    feature_model = LinearFeatureModel(s1,s2)
    es = RFNN_DD(layers,feature_model,multiplicity=1,activation=tanh)
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
