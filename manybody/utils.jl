using HDF5
using Plots
using LinearAlgebra
include("../approximators/simple.jl")

function eval_angles(R1,R2,R3)
    a = norm(R1-R2)
    b = norm(R2-R3)
    c = norm(R3-R1)
    function θ(a,b,c)
        t = (b^2+c^2-a^2)/(2*b*c)
        if isnan(t) | (abs.(t)>1.0) | isinf(t)
            return 0.0
        end
        return acos(t)
    end 
    return θ(a,b,c),θ(b,c,a),θ(c,a,b)
end 

function compute_angles(R)
    m,n,d = size(R)
    # Z = zeros(3,n,n,n,m)
    Z = []
    for l=1:m
        Z1 = []
        for i=1:n
            for j=1:n
                for k=1:n
                    temp = eval_angles(R[l,i,:],R[l,j,:],R[l,k,:])
                    # Z[1,i,j,k,l] = temp[1]
                    # Z[2,i,j,k,l] = temp[2]
                    # Z[3,i,j,k,l] = temp[3]
                    push!(Z1,temp...)
                end 
            end 
        end 
        push!(Z,Z1)
    end 
    hcat(Z...)
end 

function gen_cm(R)
    σ = 0.5
    m,n,d = size(R)
    Z = []
    for l=1:m
        Z1 = []
        for i=1:n
            for j=1:n
                for k=1:n
                    R1,R2,R3 = R[l,i,:],R[l,j,:],R[l,k,:]
                    temp = exp.(-(1/σ^2)*(norm(R1-R2) + norm(R2-R3) + norm(R1-R3)))
                    push!(Z1,temp)
                end 
            end 
        end
        push!(Z,Z1)
    end 
    return hcat(Z...)
end

function predict(R,E,label,des) 
    nd = size(R,1)
    k = floor(Int,nd/3)
    R = vcat(reshape(R[1:k,:],(1,k,:)),reshape(R[k+1:2*k,:],(1,k,:)),reshape(R[2*k+1:end,:],(1,k,:)))
    R = permutedims(R,(3,2,1))
    angles = des(R)
    CM = read(file["CM"])
    layers = [7500]
    s1 = 2*log(1.5)
    s2 = log(1.5)
    feature_model = LinearFeatureModel(s1,s2)
    activation = tanh
    m = RFNN(layers,feature_model;activation=activation)
    heuristic=Uniform
    lam = 1e-8

    @info "$label"
    @info "Compound"
    Inter = vcat(CM,angles)
    train,test = split_data(Inter,E)
    xtrain,ytrain = train
    Eapprox = m(xtrain,ytrain,heuristic,lam)
    f1,m1,r1,err1 = validate(Eapprox,train)
    @show m1,r1
    xtest,ytest = test
    f2a,m2,r2,err2 = validate(Eapprox,test)
    @show m2,r2

    @info "Simple"
    Inter = vcat(CM)
    train,test = split_data(Inter,E)
    xtrain,ytrain = train
    Eapprox = m(xtrain,ytrain,heuristic,lam)

    f1,m1,r1,err1 = validate(Eapprox,train)
    @show m1,r1
    xtest,ytest = test

    f2b,m2,r2,err2 = validate(Eapprox,test)
    @show m2,r2

    @info "Complex"
    Inter = vcat(angles)
    train,test = split_data(Inter,E)
    xtrain,ytrain = train
    Eapprox = m(xtrain,ytrain,heuristic,lam)

    f1,m1,r1,err1 = validate(Eapprox,train)
    @show m1,r1
    xtest,ytest = test

    f2c,m2,r2,err2 = validate(Eapprox,test)
    @show m2,r2
    return plot(f2a,f2b,f2c,size=(1200,500))
end

