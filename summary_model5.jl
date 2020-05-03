cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics,Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta, Clustering
using JLD2, FileIO
@load "splitted.jld2"
@load "model5.jld2"

Random.seed!(1)

val = DataFrame(val)
subset = train[:,[:business_id, :assignments]]
val = join(val, subset, on= :business_id, kind=:left, makeunique=true)
unique!(val)

train_1 = train[train[:assignments].==1,:]
train_2 = train[train[:assignments].==2,:]
train_3 = train[train[:assignments].==3,:]
train_4 = train[train[:assignments].==4,:]
train_5 = train[train[:assignments].==5,:]
train_6 = train[train[:assignments].==6,:]

val_1 = val[val[:assignments].==1,:]
val_2 = val[val[:assignments].==2,:]
val_3 = val[val[:assignments].==3,:]
val_4 = val[val[:assignments].==4,:]
val_5 = val[val[:assignments].==5,:]
val_6 = val[val[:assignments].==6,:]

unique(train_1[:user_id])
unique(train_1[:business_id])
unique(train_2[:user_id])
unique(train_2[:business_id])
unique(train_3[:user_id])
unique(train_3[:business_id])
unique(train_4[:user_id])
unique(train_4[:business_id])
unique(train_5[:user_id])
unique(train_5[:business_id])
unique(train_6[:user_id])
unique(train_6[:business_id])

unique(val_1[:user_id])
unique(val_1[:business_id])
unique(val_2[:user_id])
unique(val_2[:business_id])
unique(val_3[:user_id])
unique(val_3[:business_id])
unique(val_4[:user_id])
unique(val_4[:business_id])
unique(val_5[:user_id])
unique(val_5[:business_id])
unique(val_6[:user_id])
unique(val_6[:business_id])
