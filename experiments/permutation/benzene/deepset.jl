include("../../../approximators/simple.jl")
using HDF5
using Plots

begin
    filename = "data/benzene2017.hdf5"
    file = h5open(filename,"r")
    num_features = 144
    cm = read(file["CM"])
    R,f = deepset(read(file["CM"]),num_features,0.5)
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

plot(R,legend=false)

using TSne
e_R = tsne(R',2,0,3000,5)
f1 = scatter(e_R[:,1],e_R[:,2],ms=3.0,zcolor=E[:],title="DeepSet")
f2 = scatter(e_R[:,1],e_R[:,2],ms=3.0,zcolor=1:3000,title="DeepSet")
plot(f1,f2,size=(700,500))

begin 
    # #Approximate energy
    begin
        @info "Energy"
        layers = [5000]
        s1 = 2*log(1.5)
        s2 = log(1.5)
        feature_model = LinearFeatureModel(s1,s2)
        activation = tanh
        m = RFNN(layers,feature_model;activation=activation)
        heuristic=Uniform
        lam = 1e-7
    

        train,test = split_data(R,E)
        xtrain,ytrain = train
        @show layers[1],lam
        Eapprox = m(xtrain,ytrain,heuristic,lam)

        f1,m1,r1,err1 = validate(Eapprox,train)
        @show m1,r1
        @info "BB"
        f2,m2,r2,err2 = validate(Eapprox,test)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        file = h5open("logs/permutation/benzene_energy.hdf5","w")
        file["err"] = err2
        file["layers"] = layers[1]
        file["MAE"] = m2
        file["RMSE"] = r2
        close(file)
    end

    # Approximate forces
    begin
        @info "Forces"
        layers = [8000]
        s1 = 2*log(1.5)
        s2 = log(1.5)
        feature_model = LinearFeatureModel(s1,s2)
        activation = tanh
        m = RFNN(layers,feature_model;activation=activation)
        heuristic = Uniform
        lam = 1e-7

        train,test = split_data(R,F)
        xtrain,ytrain = train
        @show layers[1],lam
        Eapprox = m(xtrain,ytrain,heuristic,lam)

        f1,m1,r1,err1 = validate(Eapprox,train)

        @show m1,r1
        @info "BB"
        f2,m2,r2,err2 = validate(Eapprox,test)
        @show m2,r2
        display(plot(f1,f2,size=(800,300)))

        file = h5open("logs/permutation/benzene_force.hdf5","w")
        file["err"] = err2
        file["layers"] = layers[1]
        file["MAE"] = m2
        file["RMSE"] = r2
        close(file)
    end
end