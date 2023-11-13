using HDF5
using Plots

function predict(layers, lam, label)
    s1 = 2*log(1.5)
    s2 = log(1.5)
    activation = tanh
    Dactivation = x->sech(x)^2
    feature_model = LinearFeatureModel(s1,s2)
    heuristic = Uniform
    stru = RFNN_Conservative(layers, feature_model,multiplicity=1,activation=activation,D_activation=Dactivation)
    Ef,Ff = stru(xtrain,ytrain,ztrain,heuristic, lam)

    @info "$label"

    @info "Energy"
    train = (xtrain, ytrain)
    test = (xtest, ytest)
    f1,m1,r1,err1 = validate(Ef,train)
    @show m1,r1
    f2a,m2,r2,err2 = validate(Ef,test)
    @show m2,r2

    @info "Forces"
    train = (xtrain, ztrain)
    test = (xtest, ztest)
    f1,m1,r1,err1 = validate(Ff,train)
    @show m1,r1
    f2b,m2,r2,err2 = validate(Ff,test)
    @show m2,r2

    return f2a, f2b
end 