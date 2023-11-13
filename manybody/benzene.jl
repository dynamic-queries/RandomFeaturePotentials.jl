include("utils.jl")

# Read data
filename = "data/benzene2017.hdf5";
file = h5open(filename);
R = read(file["R"]);
E = read(file["E"]);
F = read(file["F"]);

# Angles
@info "Angles"
f1 = predict(R,E,"Benzene-Energy",compute_angles);
f2 = predict(R,F,"Benzene-Force",compute_angles);

# Generalized CM
@info "GCM"
f1 = predict(R,E,"Benzene-Energy",gen_cm);
f2 = predict(R,F,"Benzene-Force",gen_cm);
