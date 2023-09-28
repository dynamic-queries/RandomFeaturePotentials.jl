include("utils.jl")

struct RFNN <: AbstractApproximator
    rng
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation

    function RFNN(layers, feature_model; multiplicity=1, activation=tanh)
        rng = Xoshiro(0)
        new(rng, layers, multiplicity, feature_model, activation)
    end 
end

function (snn::RFNN)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic),λ;atol=1e-12)
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
    

    Ainfer = bases'*bases
    
    binfer = bases'*ytrain'
    coeff = (Ainfer + λ*I)\binfer

    # Setup model
    model = x -> (coeff' * snn.activation.(snn.feature_model.W*x .+ snn.feature_model.b))

    return model
end


