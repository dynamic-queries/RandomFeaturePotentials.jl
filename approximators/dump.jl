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
    b(x1,x2) = ((x1-x2)/(norm(x1-x2)).^2)' * x1 + model.s2
    
    nsamples = K
    L = length(ρ)
    idx = wsample(rng, 1:L, Weights(ρ), nsamples, replace=false)
    idx_from = idxs[1][idx]
    idx_to = idxs[2][idx]
    W1 = []
    b1 = []
    for i=1:K
        k1 = xtrain[:,idx_from[i]]
        k2 = xtrain[:,idx_to[i]]
        if k1!=k2
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
    ϵ = 1e-8
    ρ = map(norm, eachslice(num, dims=2)) ./ (map(norm, eachslice(den, dims=2)).+ϵ)
    return [idx_from, idx_to],ρ
end

# ---------------------------------------------------------------------------------------------- #
struct SamplingNN <: AbstractApproximator
    rng
    dims_in::Int
    dims_out::Int
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation

    function SamplingNN(dims_in, dims_out, layers, feature_model; multiplicity=1, activation=tanh)
        rng = Xoshiro(0)
        new(rng, dims_in, dims_out, layers, multiplicity, feature_model, activation)
    end 
end

function (snn::SamplingNN)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic),λ;atol=1e-12,optimize=false,device=:gpu)
    # Evaluate sampling density
    M = size(xtrain)[end]
    Nl = snn.layers[1]
    H = heuristic()
    idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)

    # Sample weights and biases
    W,b = snn.feature_model(snn.rng, xtrain, idxs, ρ, Nl)

    # Solve for coefficients of last layer
    bases = snn.activation.(W*xtrain .+ b)'
    @show cond(bases)
    if optimize==false
        if device==:gpu
            bases = CuArray(bases)
            y  = CuArray(ytrain')
            coeff = Array(CUDA.pinv(bases,atol=λ)*y)
        else
            coeff = pinv(bases,atol=λ)*ytrain'
        end 
    else
        r = size(ytrain,1)
        coeff,stats = Krylov.lslq(Array(bases),Array(ytrain'[:]),λ=λ,atol=atol)
        coeff = reshape(coeff,r,:)'
    end  # size(coeff) = K, output_dims 

    # Setup model
    model = x -> (coeff' * snn.activation.(snn.feature_model.W*x .+ snn.feature_model.b))

    return model
end

# ---------------------------------------------------------------------------------------------- #

# Extensive systems model
struct ExtensiveSamplingNN <: AbstractApproximator
    rng
    dims_in::Int
    dims_out::Int
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation

    function ExtensiveSamplingNN(dims_in, dims_out, layers, feature_model; multiplicity=1, activation=tanh)
        new(Xoshiro(0), dims_in, dims_out, layers, multiplicity, feature_model, activation)
    end
end

function (es::ExtensiveSamplingNN)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic), λ; atol=1e-12, optimize=false, sampler=:force)
    n = floor(Int,sqrt(size(xtrain,1)))
    Nls = es.layers
    Ws = []
    bs = []
    xtrain = reshape(xtrain, n, n, :)
    for i=1:n
        H = heuristic()
        idxs, ρ = H(xtrain[i,:,:], ytrain, Nls[i], multiplicity)
        W,b = f_model(es.rng, xtrain[i,:,:], idxs, ρ, Nls[i])
        push!(Ws,W)
        push!(bs,b)
    end 


    bases = []
    ϕ = (W,b,x) -> gelu.(W*x .+ b)
    for i=1:n
        push!(bases, ϕ(Ws[i],bs[i],xtrain[i,:,:])')
    end 

    Bases = hcat(bases...)
    lam_opt = 1e-4
    Bases_cond = (Bases'*Bases + lam_opt*I)
    ytrain_cond = Bases'*ytrain[:]
    k = Bases_cond\ytrain_cond
end

# ---------------------------------------------------------------------------------------------- #

# Conservative systems modeller
struct SamplingPotential <: AbstractApproximator
    rng
    dims_in::Int
    dims_out::Int
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation
    D_activation

    function SamplingPotential(dims_in, dims_out, layers, feature_model; multiplicity=1, activation=tanh, D_activation=x->sech(x)^2)
        rng = Xoshiro(0)
        new(rng, dims_in, dims_out, layers, multiplicity, feature_model, activation,D_activation)
    end 
end 

function (sp::SamplingPotential)(xtrain, Etrain, Ftrain, heuristic::typeof(AbstractHeuristic), λ; atol=1e-12,optimize=false, sampler=:force)
    # Evaluate sampling density
    M = size(xtrain)[end]
    Nl = snn.layers[1]
    H = heuristic()
    d = size(xtrain,1)

    # Sample weights and biases
    if sampler==:force
        idxs,ρ = H(xtrain, Ftrain, Nl, snn.multiplicity)
    elseif sampler ==:energy
        idxs,ρ = H(xtrain, Etrain, Nl, snn.multiplicity)
    end 
    W,b = sp.feature_model(snn.rng, xtrain, idxs, ρ, Nl)

    # Assemble bases
    E_bases = sp.activation.(W*xtrain .+ b)'
    F_bases = sp.D_activation.(W*xtrain .+ b)'
    K = size(W,1)
    Bases = zeros(d+1,size(E_bases)...)
    Bases[1,:,:] .= E_bases
    for j=2:d+1
       for i=1:K
            Bases[j,i,:] = W[i,j-1] * F_bases[i,:]
       end 
    end 
    Bases = reshape(permutedims(Bases,(2,1,3)),(K,:))
    @show cond(Bases)

    # Fit coefficients
    ytrain = vcat(Etrain, Ftrain)

    if optimize==false
        coeff = pinv(Bases,atol=λ) * ytrain[:]
    else 
        coeff,stats = Krylov.lslq(Bases,ytrain',λ=λ,atol=atol)
    end 

    energy = x -> (coeff' * sp.activation.(sp.feature_model.W*x .+ sp.feature_model.b))
    
    function force(x)
        K,d = size(snn.feature_model.W)
        force_bases = zeros(K,d,size(x,2))
        for i=1:K
            for j=1:d
                force_bases[i,j,:] =  sp.D_activation.(sp.feature_model.W*x .+ sp.feature_model.b)[i,:] .* sp.W[i,j]
            end 
        end 
        return reshape(coeff' * reshape(force_bases,(K,:)),(d,m))
    end

    return energy, force
end 
# ---------------------------------------------------------------------------------------------- #
