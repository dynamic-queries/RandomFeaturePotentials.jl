include("utils.jl")

struct RFNN_Conservative <: AbstractApproximator
    rng
    layers::Int
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation
    D_activation

    function RFNN_Conservative(layers, feature_model,multiplicity=1,activation=gelu,D_activation=Dgelu)
        new(Xoshiro(0), layers, multiplicity, feature_model,  activation, D_activation)
    end
end

function (stru::RFNN_Conservative)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic),λ)
    
end