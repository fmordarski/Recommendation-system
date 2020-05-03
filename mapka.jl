cd("C:/Users/Uzytkownik/Documents/Julia/praca")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta
using JLD2, FileIO
@load "prepared.jld2" reviews business
# Create merged dataset by reviews and business datasets

merged = join(reviews, business, on = :business_id, makeunique = true)

#Create merged dataset by states

states = by(merged, :state, :longitude => mean, :latitude => mean)
stats_count =by(merged, :state, nrow)
stats = join(states, stats_count, on = :state, makeunique = true)

# Plot review counts on map

using GMT
coast(region=[-130 -70 24 52], proj=(name=:lambertConic, center=[-100 35], parallels=[33 45]),
frame=:ag, res=:low, borders=((type=1, pen=("thick","red")), (type=2, pen=("thinner",))),
area=500, land=:tan, water=:blue, shore=(:thinnest,:white))
GMT.plot!(stats.longitude_mean,stats.latitude_mean, size=0.0000003*stats.x1,
fill="green",show=true)
xnoi=sort!(stats, cols = (order(:x1, rev = true)))

final = merged[merged[:state].!="NV",:]
final = final[final[:state].!="AZ",:]
#final = merged[merged[:state].=="NC",:]
cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
@save "final.jld2" final
