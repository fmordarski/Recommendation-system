cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics,Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta, Clustering
using JLD2, FileIO
@load "splitted.jld2"
Random.seed!(1)

features = collect(Matrix(train[:, [:latitude, :longitude]])')
result = kmeans(features, 6, maxiter=1000)
train = DataFrame(train)
mu = mean(train.stars)
train.assignments = result.assignments

val = DataFrame(val)
subset = train[:,[:business_id, :assignments]]
val = join(val, subset, on= :business_id, kind=:left, makeunique=true)
unique!(val)

insert!(val, 12, 0.5, :pred_1)
insert!(val, 12, 0.5, :pred_2)
insert!(val, 12, 0.5, :pred_3)
insert!(val, 12, 0.5, :pred_4)
insert!(val, 12, 0.5, :pred_5)
insert!(val, 12, 0.5, :pred_6)

train_1 = train[train[:assignments].==1,:]
train_2 = train[train[:assignments].==2,:]
train_3 = train[train[:assignments].==3,:]
train_4 = train[train[:assignments].==4,:]
train_5 = train[train[:assignments].==5,:]
train_6 = train[train[:assignments].==6,:]
trains = Dict{Int, DataFrame}()

push!(trains, 1=>train_1)
push!(trains, 2=>train_2)
push!(trains, 3=>train_3)
push!(trains, 4=>train_4)
push!(trains, 5=>train_5)
push!(trains, 6=>train_6)

val_1 = val[val[:assignments].==1,:]
val_2 = val[val[:assignments].==2,:]
val_3 = val[val[:assignments].==3,:]
val_4 = val[val[:assignments].==4,:]
val_5 = val[val[:assignments].==5,:]
val_6 = val[val[:assignments].==6,:]



vals =  Dict{Int, DataFrame}()

push!(vals, 1=>val_1)
push!(vals, 2=>val_2)
push!(vals, 3=>val_3)
push!(vals, 4=>val_4)
push!(vals, 5=>val_5)
push!(vals, 6=>val_6)

print("  liczba wierszy zestaw 1:  ")
print(nrow(train_1))
print("  liczba wierszy zestaw 2:  ")
print(nrow(train_2))
print("  liczba wierszy zestaw 2:  ")
print(nrow(train_3))
print("  liczba w4erszy zestaw 2:  ")
print(nrow(train_4))
print("  liczba w5erszy zestaw 2:  ")
print(nrow(train_5))
print("  liczba wierszy zestaw 2:  ")
print(nrow(train_6))



dict_trains = Dict([(1, train_1), (2, train_2), (3, train_3),
                    (4, train_4), (5, train_5), (6, train_6)])

mappings = Dict{Int, Array}()

user_mappings_1, business_mappings_1 = Dict{String,Int}(), Dict{String,Int}()
user_mappings_2, business_mappings_2 = Dict{String,Int}(), Dict{String,Int}()
user_mappings_3, business_mappings_3 = Dict{String,Int}(), Dict{String,Int}()
user_mappings_4, business_mappings_4 = Dict{String,Int}(), Dict{String,Int}()
user_mappings_5, business_mappings_5 = Dict{String,Int}(), Dict{String,Int}()
user_mappings_6, business_mappings_6 = Dict{String,Int}(), Dict{String,Int}()

push!(mappings, 1=>[user_mappings_1, business_mappings_1])
push!(mappings, 2=>[user_mappings_2, business_mappings_2])
push!(mappings, 3=>[user_mappings_3, business_mappings_3])
push!(mappings, 4=>[user_mappings_4, business_mappings_4])
push!(mappings, 5=>[user_mappings_5, business_mappings_5])
push!(mappings, 6=>[user_mappings_6, business_mappings_6])

counters = Dict{Int, Array}()

user_counter_1, business_counter_1 = 0, 0
user_counter_2, business_counter_2 = 0, 0
user_counter_3, business_counter_3 = 0, 0
user_counter_4, business_counter_4 = 0, 0
user_counter_5, business_counter_5 = 0, 0
user_counter_6, business_counter_6 = 0, 0

push!(counters, 1=>[user_counter_1, business_counter_1])
push!(counters, 2=>[user_counter_2, business_counter_2])
push!(counters, 3=>[user_counter_3, business_counter_3])
push!(counters, 4=>[user_counter_4, business_counter_4])
push!(counters, 5=>[user_counter_5, business_counter_5])
push!(counters, 6=>[user_counter_6, business_counter_6])

events = Dict([(1, Event[]), (2, Event[]), (3, Event[]), (4, Event[]), (5, Event[]), (6, Event[])])

for i = 1:6
    for row in eachrow(dict_trains[i])
        user_id = row.user_id
        business_id = row.business_id
        rating = row.stars
        haskey(mappings[i][1], user_id) || (mappings[i][1][user_id] = (counters[i][1] += 1))
        haskey(mappings[i][2], business_id) || (mappings[i][2][business_id] = (counters[i][2] += 1))
        push!(events[i], Event(mappings[i][1][user_id], mappings[i][2][business_id], rating))
    end
end

da_train_1 = DataAccessor(events[1], counters[1][1], counters[1][2])
da_train_2 = DataAccessor(events[2], counters[2][1], counters[2][2])
da_train_3 = DataAccessor(events[3], counters[3][1], counters[3][2])
da_train_4 = DataAccessor(events[4], counters[4][1], counters[4][2])
da_train_5 = DataAccessor(events[5], counters[5][1], counters[5][2])
da_train_6 = DataAccessor(events[6], counters[6][1], counters[6][2])

da_trains = Dict([(1, da_train_1), (2, da_train_2), (3, da_train_3),
                  (4, da_train_4), (5, da_train_5), (6, da_train_6)])

recommenders = Dict{Int, MF}()
print("  1  zestaw")
Random.seed!(1)
recommender = MF(da_trains[1], 100)
build!(recommender, max_iter=500)
print("  build model for set: 1  ")
push!(recommenders, 1=>recommender)
print("  2  zestaw")
Random.seed!(1)
recommender = MF(da_trains[2], 100)
build!(recommender, max_iter=500)
print("  build model for set: 2  ")
push!(recommenders, 2=>recommender)
print("  3  zestaw")
Random.seed!(1)
recommender = MF(da_trains[3], 100)
build!(recommender, max_iter=500)
print("  build model for set: 3  ")
push!(recommenders, 3=>recommender)
print("  4  zestaw")
Random.seed!(1)
recommender = MF(da_trains[4], 700)
build!(recommender, max_iter=500)
print("  build model for set: 4  ")
push!(recommenders, 4=>recommender)
print("  5  zestaw")
Random.seed!(1)
recommender = MF(da_trains[5], 100)
build!(recommender, max_iter=500)
print("  build model for set: 5  ")
push!(recommenders, 5=>recommender)
print("  6  zestaw")
Random.seed!(1)
recommender = MF(da_trains[6], 700)
build!(recommender, max_iter=500)
print("  build model for set: 6  ")
push!(recommenders, 6=>recommender)

@save "model5.jld2" recommenders mappings train
