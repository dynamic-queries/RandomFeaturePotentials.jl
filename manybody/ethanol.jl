include("utils.jl")

# Read data
filename = "data/ethanol.hdf5";
file = h5open(filename);
R = read(file["R"]);
E = read(file["E"]);
F = read(file["F"]);

# Angles
@info "Angles"
f1 = predict(R,E,"Ethanol-Energy",compute_angles);
f2 = predict(R,F,"Ethanol-Force",compute_angles);

# Generalized CM
@info "GCM"
f1 = predict(R,E,"Ethanol-Energy",gen_cm);
f2 = predict(R,F,"Ethanol-Force",gen_cm);
