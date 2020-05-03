cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta
using JLD2, FileIO
@load "final.jld2"

user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
user_stats = by(final, :user_id, nrow)
final = final[∈(user_stats[(user_stats[:x1].>=20),
                :user_id]).(final.user_id),:]
bus_stats=by(final, :business_id, nrow)
final = final[∈(bus_stats[(bus_stats[:x1].>=20), :business_id]).(final.business_id),:]
bus_stats=by(final, :business_id, nrow)
user_stats = by(final, :user_id, nrow)
nrow(user_stats[user_stats[:x1].<20, :])
nrow(bus_stats[bus_stats[:x1].<20, :])

#train = final
Random.seed!(1)
train, test = splitobs(final, at = 0.99)
Random.seed!(1)
test, val = splitobs(test, at = 0.5)
unique(train[:user_id])
unique(test[:user_id])
unique(train[:business_id])
unique(test[:business_id])
cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
@save "splitted.jld2" train test val
