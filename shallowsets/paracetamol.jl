include("utils.jl")
using HDF5
using Plots

begin
    filename = "data/paracetamol_dft.hdf5"
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
Inter = compound_random_normal_features(150,400,R,E)
plot(Inter,legend=false)
predict([5000],[8000],1e-5,1e-5,Inter,E,F,viz=true)

# Compounded sampled projection
Inter = compound_sampled_features(100,400,R,E)
plot(Inter,legend=false)
predict([5000],[8000],1e-6,1e-4,Inter,E,F)


# Random normal projection 
Inter = random_normal(401,1,R);
plot(Inter,legend=false)
predict([8000],[10000],1e-5,1e-5,Inter,E,F)

# Log projection
Inter = log_projection(100,1,R);
plot(Inter,legend=false)
predict([5000],[8000],1e-6,1e-5,Inter,E,F)

# Sampled projections
Inter = sampled_projection(50,1,R,E);
plot(Inter,legend=false)
predict([5000],[8000],1e-6,1e-6,Inter,E,F)
