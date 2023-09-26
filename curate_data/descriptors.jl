using LinearAlgebra

function coulomb_matrix(Z, R)
    s = size(R)
    cm = zeros(eltype(R),s[1],s[1],s[3])
    g2 = (x1,x2,z1,z2) -> z1*z2/norm(x1-x2)
    g1 = (z1,z2) -> (z1*z2)^0.24   
    for i=1:s[3]
        for j=1:s[1]
            for k=1:s[1]
                if j==k
                    cm[j,k,i] = g1(Z[j],Z[k]) 
                else 
                    cm[j,k,i] = g2(R[j,:,i],R[k,:,i],Z[j],Z[k])
                end 
            end 
        end
    end 
    return cm
end

function eigen_descriptors(SM)
    s = size(SM)
    sm = zeros(s[1],s[3])
    for i=1:s[3]
        sm[:,i] = eigvals(SM[:,:,i])
    end 
    return sm
end 

function singular_descriptors(SM)
    s = size(SM)
    sm = zeros(s[1],s[3])
    for i=1:s[3]
        _,s,_ = svd(SM[:,:,i])
        sm[:,i] = s
    end 
    return sm
end 


function sorted_descriptors(SM)
    s = size(SM)
    sd = zero(SM)
    for i=1:s[3]
        idx = sortperm(sum(SM[:,:,i],dims=1)[:])
        sd[:,:,i] .= SM[idx,idx,i]
    end 
    sd
end 

function derivative_coulomb_matrix(Z,R)
    s = size(R)
    cm = coulomb_matrix(Z,R)
    dcm = zeros(s[1],s[1],s[1],s[2],s[3])
    for k=1:s[3]
        for i=1:s[1]
            for j=1:s[1] 
                for l=1:s[1]
                    if l==i
                        if i==j
                            dcm[i,j,l,:,k] .= 0
                        else 
                            dcm[i,j,l,:,k] = .0
                        end 
                    elseif l==j
                        dcm[i,j,l,:,k] = .0
                    else 
                        dcm[i,j,l,:,k] .= 0
                    end 
                end
            end 
        end 
    end 
end
