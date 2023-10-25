using LinearAlgebra
using StatsBase
using Random
using Statistics
using StatsPlots


abstract type AbstractApproximator end
abstract type AbstractFeatureModel end

mutable struct LinearFeatureModel <: AbstractFeatureModel 
    s1::Float64
    s2::Float64
    W::Any
    b::Any

    function LinearFeatureModel(s1,s2)
        new(s1,s2,nothing,nothing)
    end
end

function (model::LinearFeatureModel)(rng, xtrain, idxs, ρ, K)
    W(x1,x2) = model.s1*(x1-x2)/norm(x1-x2).^2
    b(x1,x2) = ((model.s1*(x1-x2)/(norm(x1-x2)).^2)' * x1) + model.s2
    
    nsamples = K
    L = length(ρ)
    idx = wsample(rng, 1:L, Weights(ρ), nsamples, replace=true)
    idx_from = idxs[1][idx]
    idx_to = idxs[2][idx]
    W1 = []
    b1 = []
    for i=1:K
        k1 = xtrain[:,idx_from[i]]
        k2 = xtrain[:,idx_to[i]]
        if norm(k1-k2)>1e-12
            push!(W1,W(k1,k2))
            push!(b1,b(k1,k2))
        end
    end

    if !isempty(W1) | !isempty(b1) 
        model.W = reduce(hcat, W1)'
        model.b = b1
        return model.W, model.b
    else 
        return nothing,nothing
    end 
end

# Heuristics for sampling density

# ---------------------------------------------------------------------------------------------- #

abstract type AbstractHeuristic end
struct Uniform <: AbstractHeuristic end
struct FiniteDifference <: AbstractHeuristic end
struct FullDerivative <: AbstractHeuristic end
struct RandomDerivative <: AbstractHeuristic end

function (heuristic::Uniform)(xtrain,  ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    nsamples = Nl*multiplicity
    idx_from = sample(1:M, nsamples,replace=true)
    idx_to = sample(1:M, nsamples,replace=true)
    ρ = (1/nsamples)*ones(nsamples)
    return [idx_from, idx_to], ρ
end 

function (heuristic::FiniteDifference)(xtrain, ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    nsamples = Nl*multiplicity
    idx_from = sample(1:M, nsamples, replace=true)
    idx_to = sample(1:M, nsamples, replace=true)
    num = ytrain[:,idx_to] .- ytrain[:,idx_from]
    den = xtrain[:,idx_to] .- xtrain[:,idx_from]
    ϵ = 1e-10
    ρ = abs.(map(x->norm(x,1), eachslice(num, dims=2)) ./ (map(norm, eachslice(den, dims=2)).+ϵ)) .+ 1e-12
    return [idx_from, idx_to],ρ
end

# ---------------------------------------------------------------------------------------------- #
# Activation functions
gelu = x -> 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x^3)))
relu = x -> max(x,0)

function deepset(CM,ts,k)
    s = size(CM)
    d = s[1]
    F = zeros(ts,s[2])
    idx = sample(1:d, ts)
    id1 = rand(1:s[2])
    id2 = rand(1:s[2])
    W = CM[idx,id1]
    b = CM[idx,id2]
    # W = rand(ts, 1)
    # b = rand()
    ρ = x -> sin(k*x) 
    # ρ = relu
    for i=1:s[2]
        temp = W*CM[:,i]' .+ b
        F[:,i] = sum(ρ.(temp ),dims=2)[:]
    end 
    # F = (F.-minimum(F)) ./ (maximum(F).-minimum(F))
    return F,x->ρ.(W*reshape(x,1,:) .+ b)
end 

function deepset(CM)
    s = size(CM)
    d = s[1]
    idx = rand(1:s[2],d)
    W = CM[:,idx]
    W = reduce(hcat, [W[:,i]./norm(W[:,i]) for i=1:size(W,1)])
    temp = sum(W*CM,dims=1)
    return temp ./ maximum(temp[:])
end 

function deepset(CM,f)
    s = size(CM)
    q = size(f.W,1)
    F = zeros(q,s[2])
    for i=1:s[2]
        F[:,i] = sum(f(CM[:,i]),dims=2)[:]
    end 
    return F
end

# ---------------------------------------------------------------------------------------------- #

function split_data(X,Y)
    d,m = size(X)
    idx = sample(1:m,m,replace=false)
    r = ratio = floor(Int,m*0.7)
    train_idx = idx[1:r]
    test_idx = idx[r+1:end]
    xtrain = X[:,train_idx]
    ytrain =  Y[:,train_idx]
    xtest = X[:,test_idx]
    ytest = Y[:,test_idx]

    train = (xtrain,ytrain)
    test = (xtest,ytest)
    train,test
end 

function rmse(err)
    sqrt(mean(err[:].^2))
end 

function mae(err)
    mean(abs.(err[:]))
end 

function validate(model,data)
    x,y = data
    pred = model(x)
    err = pred - y
    MAE,RMSE = mae(err), rmse(err)
    f1 = plot(y[:],y[:],label="GT",gridlinewidth=2.0,minorgrid=true)
    scatter!(y[:],pred[:],label="Fit",ms=1.0,title="MAE = $MAE")
    return f1,MAE,RMSE,err    
end 

# ---------------------------------------------------------------------------------------------- #