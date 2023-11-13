include("utils.jl")

struct ConservativeRFNN <: AbstractApproximator
    rng
    feature_model::LinearFeatureModel
    activation::Function
    Dactivation::Function
    
end 


x = -π:1e-3:π
y = sin.(12*x).*cos.(x)
z = 12*cos.(12*x).*cos.(x) - sin.(x).*sin.(12*x)

xtrain = reshape(x[50:10:end-50],1,:)
ytrain = reshape(y[50:10:end-50],1,:)
ztrain = reshape(z[50:10:end-50],1,:)

begin
    Nl = 3000
    H = Uniform()
    s1 = log(1.5)
    s2 = 2*log(1.5)
    activation = tanh
    Dactivation = x -> sech(x)^2
    linear_model = LinearFeatureModel(s1,s2)

    M = size(xtrain,2)
    idxs,ρ = H(xtrain, ztrain, Nl, 1)
    W,b = linear_model(Xoshiro(0), xtrain, idxs, ρ, Nl)
    λ = 1e-6

    Nl = size(W,1)
    F1 = activation.(W*xtrain .+ b)
    temp = Dactivation.(W*xtrain .+ b)
    F2 = zeros(size(F1)...,size(xtrain,1))
    for i=1:Nl
        for j=1:size(xtrain,1)
            F2[i,:,j] .= W[i,j]*temp[i,:,j]
        end 
    end 
    F3 = zeros(Nl,M,size(xtrain,1)+1)
    F3[:,:,1] .= F1
    F3[:,:,2:end] .= F2[:,:,1:end]

    A = reshape(F3,Nl,:)'
    Y = hcat(ytrain,ztrain)'
    Ainfer = A'*A
    Yinfer = A'*Y
    k = (Ainfer + λ*I)\Yinfer

    Ypred = reshape(k'*reshape(permutedims(F3,(1,3,2)),Nl,:),:,M)

    plot(xtrain[1,:],Ypred[1,:],label="F prediction")
    scatter!(xtrain[1,:],ytrain[1,:],ms=1.0,label="F data")
    plot!(xtrain[1,:],Ypred[2,:], label="DF prediction")
    scatter!(xtrain[1,:], ztrain[1,:],ms=1.0, label="DF data")

    xtest = reshape(x,1,:)
    ytest = reshape(y,1,:)
    ztest = reshape(z,1,:)

    F1 = activation.(W*xtest .+ b)
    temp = Dactivation.(W*xtest .+ b)
    F2 = zeros(size(F1)...,size(xtest,1))
    for i=1:Nl
        for j=1:size(xtest,1)
            F2[i,:,j] .= W[i,j]*temp[i,:,j]
        end 
    end 
    m = size(xtest,2)
    F3 = zeros(Nl,m,size(xtest,1)+1)
    F3[:,:,1] .= F1
    F3[:,:,2:end] .= F2[:,:,1:end]

    Ytest = reshape(k'*reshape(permutedims(F3,(1,3,2)),Nl,:),:,m)
    f1 = plot(xtest[1,:],Ytest[1,:],label="F prediction",linewidth=5.0)
    scatter!(xtrain[1,:],ytrain[1,:],ms=1.0,label="F data")
    plot!(xtest[1,:],Ytest[2,:], label="DF prediction",linewidth=5.0)
    scatter!(xtrain[1,:], ztrain[1,:],ms=1.0, label="DF data")

    f2 = plot(xtest[1,:],Ytest[1,:]-ytest[1,:],label="F error",linewidth=0.5)
    plot!(xtest[1,:],Ytest[2,:]-ztest[1,:],label="DF error",linewidth=0.5)

    plot(f1,f2,size=(800,300))
end