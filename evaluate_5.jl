cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random, Recommendation, MLDataUtils, DataFrames, Statistics
using DataFramesMeta, Clustering, ProgressMeter
using JLD2, FileIO
@load "splitted.jld2"
@load "model5.jld2"

mu = mean(train.stars)
test = DataFrame(test)
val = DataFrame(val)
subset = train[:,[:business_id, :assignments]]
test = join(test, subset, on= :business_id, kind=:left, makeunique=true)
val = join(val, subset, on= :business_id, kind=:left, makeunique=true)
unique!(test)
unique!(val)
insert!(test, 12, 0.5, :stars_pred)
insert!(val, 12, 0.5, :stars_pred)
insert!(train, 12, 0.5, :pred)

function process(df, pred, mappings, recommenders, train)
     for (j, row) in enumerate(df)
        global g
        index = row.assignments
        if (row.user_id in collect(keys(mappings[index][1]))) & (row.business_id in collect(keys(mappings[index][2])))
            pred[j] = Recommendation.predict(recommenders[index],
                                                    mappings[index][1][row.user_id],
                                                    mappings[index][2][row.business_id])
        else
            subset = train[train[:business_id].==row.business_id,:]
            subset = subset[subset[:assignments].==row.assignments,:]
            pred[j] = mean(subset.stars)
            g = g + 1
        end
    end
end

g = 0
using Tables, DataFrames
process(Tables.namedtupleiterator(val), val.stars_pred, mappings, recommenders, train)

val.stars_pred[val[:stars_pred].>5,:] .= 5
val.stars_pred[val[:stars_pred].<0,:] .= 0

rmse2 = sqrt(mean((val[:stars] - val[:stars_pred]).^2))
mae = mean(abs.(val[:stars] - val[:stars_pred]))

print("     val RMSE:    ")
print(rmse2)
print(" val MAE:  ")
print(mae)
print( " g:   ")
print(g)

g = 0

using Tables, DataFrames
process(Tables.namedtupleiterator(test), test.stars_pred, mappings, recommenders, train)

test.stars_pred[test[:stars_pred].>5,:] .= 5
test.stars_pred[test[:stars_pred].<0,:] .= 0

rmse2 = sqrt(mean((test[:stars] - test[:stars_pred]).^2))
mae = mean(abs.(test[:stars] - test[:stars_pred]))
print("     test RMSE:    ")
print(rmse2)
print(" test MAE:  ")
print(mae)
print( " g:   ")
print(g)

using Tables, DataFrames
process(Tables.namedtupleiterator(train), train.pred, mappings, recommenders, train)

train.pred[train[:pred].>5,:] .= 5
train.pred[train[:pred].<0,:] .= 0
rmse = sqrt(mean((train[:stars] - train[:pred]).^2))
mae = mean(abs.(train[:stars] - train[:pred]))
print("     train RMSE:    ")
print(rmse)
print(" train MAE:  ")
print(mae)
