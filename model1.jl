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
business_avg = join(test, df_avg, on = :business_id, makeunique = true, kind=:left)

train = DataFrame(train)
val = DataFrame(val)
business_avg_train = join(train, df_avg, on = :business_id, makeunique = true, kind=:left)
business_avg_val = join(val, df_avg, on = :business_id, makeunique = true, kind=:left)

business_avg.value0[ismissing.(business_avg.value0)] .= 0
business_avg_train.value0[ismissing.(business_avg_train.value0)] .= 0
business_avg_val.value0[ismissing.(business_avg_val.value0)] .= 0

business_avg = @linq business_avg |>
           transform(new = :value0 .+ mu_hat)
business_avg_train = @linq business_avg_train |>
                      transform(new = :value0 .+ mu_hat)
business_avg_val = @linq business_avg_val |>
                      transform(new = :value0 .+ mu_hat)



business_avg.new[business_avg[:new].>5,:] .= 5
business_avg.new[business_avg[:new].<0,:] .= 0

business_avg_train.new[business_avg_train[:new].>5,:] .= 5
business_avg_train.new[business_avg_train[:new].<0,:] .= 0

business_avg_val.new[business_avg_val[:new].>5,:] .= 5
business_avg_val.new[business_avg_val[:new].<0,:] .= 0
print("   train RMSE:    ")
print(sqrt(mean((business_avg_train[:stars] - business_avg_train[:new]).^2)))

print("   train MAE:    ")
print(mean(abs.(business_avg_train[:stars] - business_avg_train[:new])))

print("   test RMSE:     ")
rmse1 = sqrt(mean((business_avg[:stars] - business_avg[:new]).^2))
print(rmse1)

print("   test MAE:    ")
print(mean(abs.(business_avg[:stars] - business_avg[:new])))

print("   val RMSE:    ")
print(sqrt(mean((business_avg_val[:stars] - business_avg_val[:new]).^2)))

print("   val MAE:    ")
print(mean(abs.(business_avg_val[:stars] - business_avg_val[:new])))
