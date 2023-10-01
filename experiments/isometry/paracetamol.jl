include("../../approximators/simple.jl")
using HDF5
using Plots

begin
    filename = "data/paracetamol_dft.hdf5"
    file = h5open(filename)
    R = read(file["CM"])
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

size(R)
plot(R,legend=false)

# Black box approximator
begin 
    # Approximate energy
    begin
        @info "Energy"
        layers = [10000]
        s1 = 1.0
        s2 = 0.0
        feature_model = LinearFeatureModel(s1,s2)
        activation = gelu
        m = RFNN(layers,feature_model;activation=activation)
        heuristic=Uniform
        lam = 1e-9

        train,test = split_data(R,E)
        xtrain,ytrain = train
        @show layers[1],lam
        Eapprox = m(xtrain,ytrain,heuristic,lam)

        f1,m1,r1,err1 = validate(Eapprox,train)

        xtest,ytest = test

        @info "Permutation"
        idx = sample(1:20,20,replace=false)
        r = reshape(xtest,20,20,:)[idx,idx,:]
        r = reshape(r,400,:)
        test_permute = (r,ytest)
        f2,m2,r2,err2 = validate(Eapprox,test_permute)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        @info "BB"
        f2,m2,r2,err2 = validate(Eapprox,test)
        @show m2,r2
        display(plot(f1,f2,size=(800,300)))

        file = h5open("logs/isometry/uracil_energy.hdf5","w")
        file["err"] = err2
        file["layers"] = layers[1]
        file["MAE"] = m2
        file["RMSE"] = r2
        close(file)
    end


    # # Approximate forces
    # begin
    #     @info "Forces"
    #     layers = [800]
    #     s1 = 1.0
    #     s2 = 0.0
    #     feature_model = LinearFeatureModel(s1,s2)
    #     activation = gelu
    #     m = RFNN(layers,feature_model;activation=activation)
    #     heuristic=Uniform
    #     lam = 1e-8

    #     train,test = split_data(R,F)
    #     xtrain,ytrain = train
    #     @show layers[1],lam
    #     Eapprox = m(xtrain,ytrain,heuristic,lam)

    #     f1,m1,r1,err1 = validate(Eapprox,train)

    #     xtest,ytest = test
    #     @info "Permutation"
    #     idx = sample(1:20,20,replace=false)
    #     r = reshape(xtest,20,20,:)[idx,idx,:]
    #     r = reshape(r,400,:)
    #     test_permute = (r,ytest)
    #     f2,m2,r2,err2 = validate(Eapprox,test_permute)
    #     @show m2,r2
    #     # display(plot(f1,f2,size=(800,300)))

    #     @info "BB"
    #     f2,m2,r2,err2 = validate(Eapprox,test)
    #     @show m2,r2
    #     display(plot(f1,f2,size=(800,300)))

    #     file = h5open("logs/isometry/uracil_force.hdf5","w")
    #     file["err"] = err2
    #     file["layers"] = layers[1]
    #     file["MAE"] = m2
    #     file["RMSE"] = r2
    #     close(file)
    # end
end

plot(Eapprox.coeff,legend=false,ms=1.0)