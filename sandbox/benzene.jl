include("utils.jl")
using HDF5
using Plots

begin
    filename = "data/benzene2017.hdf5"
    file = h5open(filename)
    num_features = 400
    R = read(file["CM"])
    E = read(file["E"])
    F = read(file["F"])
    close(file)
end

plot(R,legend=false)

## Compound
# Compounded normal projection
Inter = compound_random_normal_features(150,144,R,E)
plot(Inter,legend=false)
predict([5000],[8000],1e-5,1e-5,Inter,E,F)

# Compounded sampled projection
Inter = compound_sampled_features(100,144,R,E)
plot(Inter,legend=false)
predict([5000],[8000],1e-6,1e-6,Inter,E,F)


## Simple 
# Random normal projection 
Inter = random_normal(100,1,R);
plot(Inter,legend=false)
predict([5000],[8000],1e-6,1e-6,Inter,E,F)

# Log projection
Inter = log_projection(100,1,R);
plot(Inter,legend=false)
predict([5000],[8000],1e-6,1e-5,Inter,E,F)

# Sampled projections
Inter = sampled_projection(50,1,R,E);
plot(Inter,legend=false)
predict([5000],[8000],1e-6,1e-6,Inter,E,F)
