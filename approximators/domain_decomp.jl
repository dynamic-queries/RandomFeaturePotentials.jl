include("utils.jl")

struct RFNN_DD <: AbstractApproximator
    rng
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation

    function RFNN_DD(layers, feature_model; multiplicity=1, activation=tanh)
        new(Xoshiro(0), layers, multiplicity, feature_model, activation)
    end
end

function (es::RFNN_DD)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic), λ)
    n = floor(Int,sqrt(size(xtrain,1)))
    Nls = es.layers
    Ws = []
    bs = []
    xtrain = reshape(xtrain, n, n, :)
    for i=1:n
        H = heuristic()
        idxs, ρ = H(xtrain[i,:,:], ytrain, Nls[i], es.multiplicity)
        W,b = es.feature_model(es.rng, xtrain[i,:,:], idxs, ρ, Nls[i])
        push!(Ws,W)
        push!(bs,b)
    end 

    bases = []
    ϕ = (W,b,x) -> es.activation.(W*x .- b)
    for i=1:n
        push!(bases, ϕ(Ws[i],bs[i],xtrain[i,:,:])')
    end 

    Bases = hcat(bases...)
    lam_opt = λ
    Bases_cond = (Bases'*Bases + lam_opt*I)
    ytrain_cond = Bases'*ytrain'
    k = Bases_cond\ytrain_cond

    function f(xtest)
        n = floor(Int,sqrt(size(xtest,1)))
        bases = []
        xtest = reshape(xtest, n, n, :)
        ϕ = (W,b,x) -> es.activation.(W*x .- b)
        for i=1:n
            push!(bases, ϕ(Ws[i],bs[i],xtest[i,:,:])')
        end 
        Bases = hcat(bases...)
        ypred = k'*Bases'
        ypred
    end

    return f
end

