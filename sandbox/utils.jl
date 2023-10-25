include("../approximators/utils.jl")
include("../approximators/simple.jl")

# Predictor
function predict(l1,l2,reg1,reg2,Inter,E,F;viz=false)
    layers = l1
    s1 = 1
    s2 = 0
    feature_model = LinearFeatureModel(s1,s2)
    activation = relu
    m = RFNN(layers,feature_model;activation=activation)
    heuristic = Uniform
    lam = reg1


    train,test = split_data(Inter,E)
    xtrain,ytrain = train

    @show layers[1],lam

    Eapprox = m(xtrain,ytrain,heuristic,lam)

    f1,m1,r1,err1 = validate(Eapprox,train)
    @show m1,r1

    @info "BB"
    f2,m2,r2,err2 = validate(Eapprox,test)
    @show m2,r2

    if viz
        display(plot(f1,f2,size=(900,300)))
    end 

    @info "Forces"
    layers = l2
    s1 = 1
    s2 = 0
    feature_model = LinearFeatureModel(s1,s2)
    activation = relu
    m = RFNN(layers,feature_model;activation=activation)
    heuristic = Uniform
    lam = reg2

    train,test = split_data(Inter,F)
    xtrain,ytrain = train
    @show layers[1],lam
    Eapprox = m(xtrain,ytrain,heuristic,lam)

    f1,m1,r1,err1 = validate(Eapprox,train)

    @show m1,r1
    @info "BB"
    f2,m2,r2,err2 = validate(Eapprox,test)
    @show m2,r2

    if viz
        display(plot(f1,f2,size=(800,300)))
    end 

    file = h5open("logs/permutation/benzene_force.hdf5","w")
    file["err"] = err2
    file["layers"] = layers[1]
    file["MAE"] = m2
    file["RMSE"] = r2
    close(file)
end

## Flattened features
# Random normal projection
function random_normal(num_features,ndims,R)
    q = ndims
    r = num_features
    Wi = 1e-1*randn(r,q)
    d,m = size(R)
    Inter = zeros(r,d,m)
    for i=1:m
        Inter[:,:,i] .= Wi*reshape(R[:,i],1,:)
    end 
    Inter = reshape((1/d)*sum(Inter,dims=2),r,m)
    Inter
end 

# Log projection
function log_projection(num_features, ndims, R)
    q = ndims
    r = num_features
    s = 1e-2
    f = x -> reduce(vcat,[log.(s*k*x) for k=1:num_features])
    d,m = size(R)
    Inter = zeros(r,d,m)
    for i=1:m
        Inter[:,:,i] .= f(reshape(R[:,i],1,:))
    end 
    Inter = reshape((1/d)*sum(Inter,dims=2),r,m)
    Inter
end 

# Sampled descriptor
function sampled_projection(num_features, ndims, R, E)
    q = ndims
    r = num_features
    s = 1e-2
    x = reshape(R,1,:)
    y = reshape(kron(E[:],ones(size(R,2))),1,:)
    M = size(x)[end]
    Nl = r
    heuristic = Uniform
    H = heuristic()
    idxs,ρ = H(x, y, Nl, 10)
    activation = relu

    # Sample weights and biases
    feature_model = LinearFeatureModel(1,0)
    W,b = feature_model(Xoshiro(0), x, idxs, ρ, Nl)
    bases(x) = activation.(W*x .- b)
    d,m = size(R)
    Inter = zeros(r,d,m)
    for i=1:m
        Inter[:,:,i] .= bases(reshape(R[:,i],1,:))
    end 
    Inter = reshape((1/d)*sum(Inter,dims=2),r,m)
    Inter
end

## Compounded features
# Random normal fetures
function compound_random_normal_features(num_features, ndims, R, E)
    k = floor(Int,sqrt(ndims))
    C = reshape(R,k,k,:)
    m = size(C,3)
    for j=1:m
        for i=1:k
            as = sortperm(C[i,:,j])
            C[i,:,j] .= C[i,:,j][as]
        end 
    end
    Wi = 1e-1*randn(num_features,k)
    f = zeros(num_features,k,m)
    for j=1:m
        f[:,:,j] .= Wi*C[:,:,j]
    end 
    Inter = sum(f,dims=2)
    reshape(Inter,:,m)
end 

## Sampled fetures
function compound_sampled_features(num_features, ndims, R, E)
    k = floor(Int, sqrt(ndims))
    C = reshape(R,k,k,:)
    m = size(R,3)
    for j=1:m
        for i=1:k
            as = sortperm(C[i,:,j])
            C[i,:,j] .= C[i,:,j][as]
        end 
    end
    heuristic = Uniform
    H = heuristic()
    idx, ρ = H(reshape(C,k,:), reshape(kron(E[:],ones(size(R,2))),1,:), num_features, 10)
    feature_model = LinearFeatureModel(1,0)
    activation = relu
    W,b = feature_model(Xoshiro(0), reshape(C,k,:), idx, ρ, num_features)
    bases(x) = activation.(W*x .- b)
    d,m = size(R)
    Inter = zeros(num_features,k,m)
    for i=1:m
        Inter[:,:,i] .= bases(C[:,:,i])
    end 
    reshape(sum(Inter,dims=2),:,m)
end 