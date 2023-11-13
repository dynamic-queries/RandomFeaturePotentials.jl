include("utils.jl")

# Read data
filename = "data/toluene.hdf5";
file = h5open(filename);
R = read(file["R"]);
E = read(file["E"]);
F = read(file["F"]);

# Angles
@info "Angles"
f1 = predict(R,E,"Toluene-Energy",compute_angles);
f2 = predict(R,F,"Toluene-Force",compute_angles);

# Generalized CM
@info "GCM"
f1 = predict(R,E,"Toluene-Energy",gen_cm);
f2 = predict(R,F,"Toluene-Force",gen_cm);
