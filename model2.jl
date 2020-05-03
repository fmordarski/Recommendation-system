cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta
using JLD2, FileIO
@load "splitted.jld2"

mu_hat = mean(train[:stars])

df = @linq train |>
           groupby(:business_id) |>
           transform(value0 = mean(:stars .- mu_hat))

df_avg = df[:, [:business_id, :value0]]
unique!(df_avg)
test = DataFrame(test)
train = DataFrame(train)
val = DataFrame(val)

business_avg = join(test, df_avg, on = :business_id, makeunique = true, kind=:left)
business_avg_val = join(val, df_avg, on = :business_id, makeunique = true, kind=:left)
business_avg_train = join(train, df_avg, on = :business_id, makeunique = true, kind=:left)

business_avg.value0[ismissing.(business_avg.value0)] .= 0
business_avg_train.value0[ismissing.(business_avg_train.value0)] .= 0
business_avg_val.value0[ismissing.(business_avg_val.value0)] .= 0


business_avg = @linq business_avg |>
           transform(new = :value0 .+ mu_hat)

business_avg_train = @linq business_avg_train |>
            transform(new = :value0 .+ mu_hat)

business_avg_val = @linq business_avg_val |>
            transform(new = :value0 .+ mu_hat)

df = @linq df |>
           groupby(:user_id) |>
           transform(value1 = mean(:stars .- mu_hat .-:value0))

df_user = df[:, [:user_id, :value1]]

unique!(df_user)

user_avg = join(test, df_user, on = [:user_id], makeunique = true, kind = :left)
user_avg_train = join(train, df_user, on = [:user_id], makeunique = true, kind = :left)
user_avg_val = join(val, df_user, on = [:user_id], makeunique = true, kind = :left)


user_avg.value1[ismissing.(user_avg.value1)] .= 0
user_avg_train.value1[ismissing.(user_avg_train.value1)] .= 0
user_avg_val.value1[ismissing.(user_avg_val.value1)] .= 0


user_avg = join(user_avg, df_avg, on = [:business_id], makeunique = true, kind = :left)
user_avg_train = join(user_avg_train, df_avg, on = [:business_id], makeunique = true, kind = :left)
user_avg_val = join(user_avg_val, df_avg, on = [:business_id], makeunique = true, kind = :left)



user_avg.value0[ismissing.(user_avg.value0)] .= 0
user_avg_train.value0[ismissing.(user_avg_train.value0)] .= 0
user_avg_val.value0[ismissing.(user_avg_val.value0)] .= 0


user_avg = @linq user_avg |>
           transform(new = :value0 .+ :value1 .+mu_hat)


user_avg_train = @linq user_avg_train |>
           transform(new = :value0 .+ :value1 .+mu_hat)

user_avg_val = @linq user_avg_val |>
          transform(new = :value0 .+ :value1 .+mu_hat)

user_avg.new[user_avg[:new].>5,:] .= 5
user_avg.new[user_avg[:new].<0,:] .= 0

user_avg_train.new[user_avg_train[:new].>5,:] .= 5
user_avg_train.new[user_avg_train[:new].<0,:] .= 0

user_avg_val.new[user_avg_val[:new].>5,:] .= 5
user_avg_val.new[user_avg_val[:new].<0,:] .= 0

rmse_test = sqrt(mean((user_avg[:stars] - user_avg[:new]).^2))
rmse_train = sqrt(mean((user_avg_train[:stars] - user_avg_train[:new]).^2))
rmse_val = sqrt(mean((user_avg_val[:stars] - user_avg_val[:new]).^2))
mae_test = mean(abs.(user_avg[:stars] - user_avg[:new]))
mae_train = mean(abs.(user_avg_train[:stars] - user_avg_train[:new]))
mae_val = mean(abs.(user_avg_val[:stars] - user_avg_val[:new]))

print("    train RMSE:   ")
print(rmse_train)
print("  train MAE:  ")
print(mae_train)
print("    test RMSE:    ")
print(rmse_test)
print("   test MAE:   ")
print(mae_test)
print("    val RMSE:    ")
print(rmse_val)
print("   val MAE:    ")
print(mae_val)
