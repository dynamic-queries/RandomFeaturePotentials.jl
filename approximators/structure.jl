include("utils.jl")

struct RFNN_Conservative <: AbstractApproximator
    rng
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation
    D_activation

    function RFNN_Conservative(layers, feature_model;multiplicity=1,activation=tanh,D_activation=x->(sech(x))^2)
        new(Xoshiro(0), layers, multiplicity, feature_model, activation, D_activation)
    end
end

function (stru::RFNN_Conservative)(xtrain, ytrain, ztrain, heuristic::typeof(AbstractHeuristic),λ,Dxtrain=nothing)
    d_in, M = size(xtrain)
    d_e, _ = size(ytrain)
    d_f, _ = size(ztrain)
    d_out = d_e + d_f
    Nl = stru.layers[1]

    # Sample bases
    H = heuristic()
    idxs,ρ = H(xtrain, ztrain, Nl, 1)
    W,b = stru.feature_model(stru.rng, xtrain, idxs, ρ, Nl)

    # Intermediates
    k1 = stru.activation.(W*xtrain .- b) # Nl,M
    k2 = -1*stru.D_activation.(W*xtrain .- b) # Nl,M
    
    Nl,_ = size(W)
    # Compute jacobian-vec product
    J = zeros(Nl,d_f,M)
    if !isnothing(Dxtrain)
        for i=1:M
            J[:,:,i] .= W*Dxtrain[:,:,i] # Nl*d_in,d_in*d_l
        end 
    else
        for i=1:M
            J[:,:,i] .= W*I # Nl*d_in,d_in*d_l
        end
    end 
    
    # Assemble inference matrices
    Y = vec(vcat(ytrain,ztrain)) # d_out,M
    Phi = zeros(d_out,M,Nl)
    for i=1:M
        Phi[:,i,:] = hcat(k1[:,i],diagm(0=>k2[:,i])*J[:,:,i])'
    end 
    Infer = reshape(Phi,:,Nl)
    AInfer = Infer'*Infer
    bInfer = Infer'*Y
    k = (AInfer + λ*I)\bInfer

    # Return force and energy functions
    function E(x)
        return k'*stru.activation.(W*x .- b)
    end 

    function F(x, Dx=nothing)
        m = size(x,2)
        J = zeros(Nl,d_f,m)
        if !isnothing(Dx)
            for i=1:m
                J[:,:,i] = W*Dx[:,:,i] 
            end 
        else
            for i=1:m
                J[:,:,i] = W*I 
            end
        end 
        y = zeros(size(x,1),m)
        for i=1:m
            y[:,i] .= vec(k'*diagm(0=>-1*stru.D_activation.(W*x[:,i] .- b))*J[:,:,i]) 
        end 
        return y
    end 
    return E,F
end

using HDF5
filename = "data/benzene2017.hdf5"
file = h5open(filename)
R = read(file["R"])
E = read(file["E"])
F = read(file["F"])

layers = [5000]
s1 = 2*log(1.5)
s2 = log(1.5)
activation = tanh
Dactivation = x->sech(x)^2
feature_model = LinearFeatureModel(s1,s2)
heuristic = Uniform
stru = RFNN_Conservative(layers, feature_model,multiplicity=1,activation=activation,D_activation=Dactivation)
Ef,Ff = stru(R,E,F,heuristic, 1e-8)

plot(E[:],E[:])
scatter!(E[:],Ef(R)[:])