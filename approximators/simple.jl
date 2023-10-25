include("utils.jl")

struct RFNN <: AbstractApproximator
    rng
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation

    function RFNN(layers, feature_model; multiplicity=10, activation=tanh)
        rng = Xoshiro(0)
        new(rng, layers, multiplicity, feature_model, activation)
    end 
end

function (snn::RFNN)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic),λ)
    # Evaluate sampling density
    M = size(xtrain)[end]
    Nl = snn.layers[1]
    H = heuristic()
    idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)

    # Sample weights and biases
    W,b = snn.feature_model(snn.rng, xtrain, idxs, ρ, Nl)

    # Solve for coefficients of last layer
    bases = snn.activation.(W*xtrain .- b)'
    
    # Moorse-Penrose Pseudoinverse
    Ainfer = bases'*bases
    binfer = bases'*ytrain'
    coeff = (Ainfer + λ*I)\binfer

    # Setup model
    model = x -> (coeff' * snn.activation.(snn.feature_model.W*x .- snn.feature_model.b))

    return model
end


# Unit test
# function approximate(x,y)
#     s1 = 2*log(1.5)
#     s2 = log(1.5)
#     feature_model = LinearFeatureModel(s1,s2)
#     activation = tanh

#     m = RFNN(layers,feature_model;activation=activation,multiplicity=1)
#     heuristic = Uniform
#     lam = 1e-8
#     Eapprox = m(x,y,heuristic,lam)
#     return norm(Eapprox(x)-y,2)
# end 

# error = []
# x = reshape(LinRange(0.0,2π,1000),1,:)
# y = reshape(sin.(x),1,:)
# Nl = 100:100:2000
# for N in Nl
#     push!(error, approximate(x,y))
# end

# plot(Nl,error)
