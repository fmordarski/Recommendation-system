cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta, Clustering
using JLD2, FileIO
@load "splitted.jld2"

user_mappings, business_mappings = Dict{String,Int}(), Dict{String,Int}()

user_counter, business_counter = 0, 0

events = Event[]
for row in eachrow(train)
        global user_counter
        global business_counter
        user_id = row.user_id
        business_id = row.business_id
        rating = row.stars
        haskey(user_mappings, user_id) || (user_mappings[user_id] = (user_counter += 1))
        haskey(business_mappings, business_id) || (business_mappings[business_id] = (business_counter += 1))
        push!(events, Event(user_mappings[user_id], business_mappings[business_id], rating))
end

da = DataAccessor(events, user_counter, business_counter)
train = DataFrame(train)
val = DataFrame(val)
insert!(train, 11, 0.5, :pred)

insert!(val, 11, 0.5, :pred_1)
insert!(val, 11, 0.5, :pred_2)
insert!(val, 11, 0.5, :pred_3)
# insert!(val, 11, 0.5, :pred_4)
# insert!(val, 11, 0.5, :pred_5)
# insert!(val, 11, 0.5, :pred_6)

steps = [100]


mu = mean(train[:stars])
recommenders = Dict{Int, MF}()
rmse_list = []
for k in steps
    print("  k:   ")
    print(k)
    Random.seed!(1)
    recommender = MF(da, k)
    build!(recommender, max_iter=500)
    push!(recommenders, k=>recommender)
    print("   build  model for:  ")
    print(k)
end


for i = 1:nrow(val)
        user_id = val.user_id[i]
        business_id = val.business_id[i]
        if (user_id in collect(keys(user_mappings))) & (business_id in collect(keys(business_mappings)))
            # val.pred_1[i] = Recommendation.predict(recommenders[10], user_mappings[user_id], business_mappings[business_id])
            # val.pred_2[i] = Recommendation.predict(recommenders[50], user_mappings[user_id], business_mappings[business_id])
            val.pred_3[i] = Recommendation.predict(recommenders[100], user_mappings[user_id], business_mappings[business_id])
            # val.pred_5[i] = Recommendation.predict(recommenders[300], user_mappings[user_id], business_mappings[business_id])
            # val.pred_4[i] = Recommendation.predict(recommenders[200], user_mappings[user_id], business_mappings[business_id])
            # val.pred_6[i] = Recommendation.predict(recommenders[500], user_mappings[user_id], business_mappings[business_id])
        else
            # val.pred_1[i] = mu
            # val.pred_2[i] = mu
            val.pred_3[i] = mu
            # val.pred_4[i] = mu
            # val.pred_5[i] = mu
            # val.pred_6[i] = mu
        end
    end
    # val.pred_1[val[:pred_1].>5,:] .= 5
    # val.pred_1[val[:pred_1].<0,:] .= 0
    # val.pred_2[val[:pred_2].>5,:] .= 5
    # val.pred_2[val[:pred_2].<0,:] .= 0
    val.pred_3[val[:pred_3].>5,:] .= 5
    val.pred_3[val[:pred_3].<0,:] .= 0
    # val.pred_4[val[:pred_4].>5,:] .= 5
    # val.pred_4[val[:pred_4].<0,:] .= 0
    # val.pred_5[val[:pred_5].>5,:] .= 5
    # val.pred_5[val[:pred_5].<0,:] .= 0
    # val.pred_6[val[:pred_6].>5,:] .= 5
    # val.pred_6[val[:pred_6].<0,:] .= 0
    # append!(rmse_list, sqrt(mean((val[:stars] - val[:pred_1]).^2)))
    # append!(rmse_list, sqrt(mean((val[:stars] - val[:pred_2]).^2)))
    append!(rmse_list, sqrt(mean((val[:stars] - val[:pred_3]).^2)))
    # append!(rmse_list, sqrt(mean((val[:stars] - val[:pred_4]).^2)))
    # append!(rmse_list, sqrt(mean((val[:stars] - val[:pred_5]).^2)))
    # append!(rmse_list, sqrt(mean((val[:stars] - val[:pred_6]).^2)))

@save "rmse_list4_final.jld2" rmse_list steps
print("  minimum k:   ")
print(steps[argmin(rmse_list)])

print(" minimum RMSE:  ")
print(minimum(rmse_list))

k = steps[argmin(rmse_list)]
recommender = recommenders[k]

@save "model4_final.jld2" train recommender user_mappings business_mappings
