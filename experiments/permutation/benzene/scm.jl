include("../../../approximators/simple.jl")
using HDF5
using Plots
using Measures

begin
    filename = "data/benzene2017.hdf5"
    file = h5open(filename)
    num_features = 144
    cm = read(file["CM"])
    R = read(file["SCM"])
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

plot(R,label=false)

using TSne

cm_e = tsne(Array(cm'),2,0,3000,10)
scm_e = tsne(Array(R'),2,0,3000,10)
f1 = scatter(cm_e[:,1],cm_e[:,2],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="CM")
f2 = scatter(scm_e[:,1],scm_e[:,2],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="SCM")
plot(f1,f2,size=(800,500))
savefig("benzene_tsne2d.png")

cm_e = tsne(Array(cm'),3,0,3000,10)
scm_e = tsne(Array(R'),3,0,3000,10)
f1 = scatter(cm_e[:,1],cm_e[:,2],cm_e[:,3],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="CM")
f2 = scatter(scm_e[:,1],scm_e[:,2],scm_e[:,3],ms=3.0,zcolor=E[:],label=false,margin=10mm,title="SCM")
plot(f1,f2,size=(1200,500))
savefig("benzene_tsne3d.png")


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
        lam = 1e-8
    

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

        file = h5open("logs/isometry/benzene_energy.hdf5","w")
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
        lam = 1e-8

        train,test = split_data(R,F)
        xtrain,ytrain = train
        @show layers[1],lam
        Eapprox = m(xtrain,ytrain,heuristic,lam)

        f1,m1,r1,err1 = validate(Eapprox,train)
        @show m1,r1

        @info "BB"
        f2,m2,r2,err2 = validate(Eapprox,test)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        file = h5open("logs/isometry/benzene_force.hdf5","w")
        file["err"] = err2
        file["layers"] = layers[1]
        file["MAE"] = m2
        file["RMSE"] = r2
        close(file)
    end
end