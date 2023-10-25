include("../../approximators/simple.jl")
using HDF5
using Plots

begin
    filename = "data/benzene2017.hdf5"
    file = h5open(filename)
    R = read(file["R"])
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

# Black box approximator

begin 
    # Approximate energy
    begin
        @info "Energy"
        layers = [5000]
        s1 = 1.0
        s2 = 0.0
        feature_model = LinearFeatureModel(s1,s2)
        activation = gelu
        m = RFNN(layers,feature_model;activation=activation)
        heuristic=Uniform
        lam = 1e-8

        train,test = split_data(R,E)
        xtrain,ytrain = train
        Eapprox = m(xtrain,ytrain,heuristic,lam)

        f1,m1,r1,err1 = validate(Eapprox,train)

        xtest,ytest = test
        @info "Translation"
        test_trans = (xtest .+ 40,ytest)
        f2,m2,r2,err2 = validate(Eapprox,test_trans)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        @info "Translation and Rotation"
        r = permutedims(reshape(xtest,12,3,:),(2,1,3))
        RM = [0.36 0.48 -0.8; -0.8 0.6 0.0; 0.48 0.64 0.6]
        r = reshape(permutedims(reshape(RM*reshape(r,3,:),(3,12,:)),(2,1,3)),36,:).+40
        test_trans = (r,ytest)
        f2,m2,r2,err2 = validate(Eapprox,test_trans)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        @info "Translation, Rotation and Permutation"
        r = permutedims(reshape(xtest,12,3,:),(2,1,3))
        RM = [0.36 0.48 -0.8; -0.8 0.6 0.0; 0.48 0.64 0.6]
        idx = sample(1:12,12,replace=false)
        r = reshape(permutedims(reshape(RM*reshape(r,3,:),(3,12,:)),(2,1,3))[idx,:,:],(36,:)).+40
        test_trans = (r,ytest)
        f2,m2,r2,err2 = validate(Eapprox,test_trans)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        @info "BB"
        f2,m2,r2,err2 = validate(Eapprox,test)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        file = h5open("logs/black_box/benzene_energy.hdf5","w")
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
        s1 = 1.0
        s2 = 0.0
        feature_model = LinearFeatureModel(s1,s2)
        activation = gelu
        m = RFNN(layers,feature_model;activation=activation)
        heuristic=Uniform
        lam = 1e-8

        train,test = split_data(R,F)
        xtrain,ytrain = train
        Eapprox = m(xtrain,ytrain,heuristic,lam)

        f1,m1,r1,err1 = validate(Eapprox,train)

        xtest,ytest = test
        @info "Translation"
        test_trans = (xtest .+ 40,ytest)
        f2,m2,r2,err2 = validate(Eapprox,test_trans)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        @info "Translation and Rotation"
        r = permutedims(reshape(xtest,12,3,:),(2,1,3))
        RM = [0.36 0.48 -0.8; -0.8 0.6 0.0; 0.48 0.64 0.6]
        r = reshape(permutedims(reshape(RM*reshape(r,3,:),(3,12,:)),(2,1,3)),36,:).+40
        test_trans = (r,ytest)
        f2,m2,r2,err2 = validate(Eapprox,test_trans)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))


        @info "Translation, Rotation and Permutation"
        r = permutedims(reshape(xtest,12,3,:),(2,1,3))
        RM = [0.36 0.48 -0.8; -0.8 0.6 0.0; 0.48 0.64 0.6]
        idx = sample(1:12,12,replace=false)
        r = reshape(permutedims(reshape(RM*reshape(r,3,:),(3,12,:)),(2,1,3))[idx,:,:],(36,:)).+40
        test_trans = (r,ytest)
        f2,m2,r2,err2 = validate(Eapprox,test_trans)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        @info "BB"
        f2,m2,r2,err2 = validate(Eapprox,test)
        @show m2,r2
        # display(plot(f1,f2,size=(800,300)))

        file = h5open("logs/black_box/benzene_force.hdf5","w")
        file["err"] = err2
        file["layers"] = layers[1]
        file["MAE"] = m2
        file["RMSE"] = r2
        close(file)
    end
end