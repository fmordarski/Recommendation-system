cd("C:/Users/Uzytkownik/Documents/Julia/praca")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta
using JLD2, FileIO
@load "splitted.jld2"
# Create merged dataset by reviews and business datasets

states = by(train, :state, :longitude => mean, :latitude => mean)
stats_count =by(train, :state, nrow)
stats = join(states, stats_count, on = :state, makeunique = true)

# Plot review counts on map

using GMT
coast(region=[-130 -70 24 52], proj=(name=:lambertConic, center=[-100 35], parallels=[33 45]),
frame=:ag, res=:low, borders=((type=1, pen=("thick","red")), (type=2, pen=("thinner",))),
area=500, land=:tan, water=:blue, shore=(:thinnest,:white))
GMT.plot!(stats.longitude_mean,stats.latitude_mean, size=0.000001*stats.x1,
fill="green",show=true)
xnoi=sort!(stats, cols = (order(:x1, rev = true)))
