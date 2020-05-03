cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta
using JLD2, FileIO
@load "splitted.jld2"
@load "model4.jld2"

function process(df, pred, user_mappings, business_mappings, recommender)
    for (j, row) in enumerate(df)
        if j == 1000
            print(j)
            print(" ")
        elseif j == 5000
            print(j)
            print("  ")
        end
        pred[j] = Recommendation.predict(recommender,
                                         user_mappings[row.user_id],
                                         business_mappings[row.business_id])
    end
end

test = DataFrame(test)
train = DataFrame(train)
val = DataFrame(val)
insert!(test, 11, 0.5, :stars_pred)
insert!(val, 11, 0.5, :stars_pred)

using Tables, DataFrames
process(Tables.namedtupleiterator(val), val.stars_pred, user_mappings,
        business_mappings, recommender)

val.stars_pred[val[:stars_pred].>5,:] .= 5
val.stars_pred[val[:stars_pred].<0,:] .= 0

rmse2 = sqrt(mean((val[:stars] - val[:stars_pred]).^2))
mae = mean(abs.(val[:stars] - val[:stars_pred]))

print("     val RMSE:    ")
print(rmse2)
print(" val MAE:  ")
print(mae)


using Tables, DataFrames
process(Tables.namedtupleiterator(test), test.stars_pred, user_mappings,
        business_mappings, recommender)

test.stars_pred[test[:stars_pred].>5,:] .= 5
test.stars_pred[test[:stars_pred].<0,:] .= 0

rmse2 = sqrt(mean((test[:stars] - test[:stars_pred]).^2))
mae = mean(abs.(test[:stars] - test[:stars_pred]))
print("     test RMSE:    ")
print(rmse2)
print(" test MAE:  ")
print(mae)

using Tables, DataFrames
process(Tables.namedtupleiterator(train), train.pred, user_mappings,
        business_mappings, recommender)

train.pred[train[:pred].>5,:] .= 5
train.pred[train[:pred].<0,:] .= 0

rmse = sqrt(mean((train[:stars] - train[:pred]).^2))
mae = mean(abs.(train[:stars] - train[:pred]))
print("     train RMSE:    ")
print(rmse)
print(" train MAE:  ")
print(mae)
