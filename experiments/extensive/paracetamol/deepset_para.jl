include("../../../approximators/domain_decomp.jl")
using HDF5
using Plots

begin
    filename = "data/paracetamol_dft.hdf5"
    file = h5open(filename)
    R = read(file["CM"])
    num_features = 625
    R,f = deepset(read(file["CM"]),num_features,0.1)
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

plot(R,legend=false)

begin
    train,test = split_data(R,E)
    xtrain,ytrain = train
    xtest,ytest = test
    layers = 400*ones(Int,22)
    heuristic = Uniform
    lam = 1e-7
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

    # display(plot(fig1,fig2,size=(900,300)))
end 


begin
    train,test = split_data(R,F)
    xtrain,ytrain = train
    xtest,ytest = test
    layers = 500*ones(Int,23)
    heuristic = Uniform
    lam = 1e-7
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

    # display(plot(fig1,fig2,size=(900,300)))
end
