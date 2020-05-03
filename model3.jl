cd("C:/Users/Uzytkownik/Documents/Julia/praca/others")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta
using JLD2, FileIO
@load "splitted.jld2"

mu_hat = mean(train[:stars])

val = DataFrame(val)
count_b = by(train, :business_id, nrow)
rename!(count_b, (:x1 => :n_b))
count_n = by(train, :user_id, nrow)
rename!(count_n, (:x1 => :n_u))
df_bus = @linq train |>
           groupby(:business_id) |>
           transform(value0 = mean(:stars .- mu_hat))

df_reg_bus = join(df_bus, count_b, on = :business_id, makeunique = true, kind=:left)

df_reg_bus = by(df_reg_bus, [:business_id,:n_b], :value0 => sum)

df_user = @linq df_bus |>
                      groupby(:user_id) |>
                      transform(value1 = mean(:stars .- mu_hat .-:value0))
df_reg_user = join(df_user, count_n, on = :user_id, makeunique = true, kind=:left)
df_reg_user = by(df_reg_user, [:user_id,:n_u], :value1 => sum)
train = DataFrame(train)
rmse3 = []
mae = []
steps =  [1:1:20;]
#steps =  [0]

for l in steps
    df_reg_bus_new = @linq df_reg_bus |>
                groupby(:business_id) |>
                transform(b_b = :value0_sum/(:n_b.+l))
    df_reg_bus_new = df_reg_bus_new[:, [:business_id, :b_b]]
    f = vcat(df_reg_bus_new[:b_b]...)
    df_reg_bus_new[:b_b] = f[:,]
    business_avg_reg = join(val, df_reg_bus_new, on = :business_id, makeunique = true, kind=:left)
    business_avg_reg.b_b[ismissing.(business_avg_reg.b_b)] .= 0
    business_avg_reg = @linq business_avg_reg |>
               transform(new = :b_b .+ mu_hat)
    df_reg_user_new = @linq df_reg_user |>
        groupby(:user_id) |>
        transform(b_u = :value1_sum/(:n_u.+l))
    df_reg_user_new = df_reg_user_new[:, [:user_id, :b_u]]
    f = vcat(df_reg_user_new[:b_u]...)
    df_reg_user_new[:b_u] = f[:,]
    avg_reg = join(business_avg_reg, df_reg_user_new, on = :user_id, makeunique = true, kind=:left)
    avg_reg.b_u[ismissing.(avg_reg.b_u)] .= 0
    avg_reg = @linq avg_reg |>
          transform(new = :b_b .+ mu_hat .+:b_u)
    avg_reg.new[avg_reg[:new].>5,:] .= 5
    avg_reg.new[avg_reg[:new].<0,:] .= 0
    append!(rmse3,sqrt(mean((avg_reg[:stars] - avg_reg[:new]).^2)))
    append!(mae,mean(abs.(avg_reg[:stars] - avg_reg[:new])))
end
@save "rmse_list3.jld2" rmse3 steps
print("    val RMSE:  ")
print(minimum(rmse3))
print("  val MAE:  ")
print(minimum(mae))
l = steps[argmin(rmse3)]
print("  optimal l:  ")
print(l)
print("    test RMSE:   ")

test = DataFrame(test)

df_reg_bus_new = @linq df_reg_bus |>
            groupby(:business_id) |>
            transform(b_b = :value0_sum/(:n_b.+l))
df_reg_bus_new = df_reg_bus_new[:, [:business_id, :b_b]]
f = vcat(df_reg_bus_new[:b_b]...)
df_reg_bus_new[:b_b] = f[:,]
business_avg_reg = join(test, df_reg_bus_new, on = :business_id, makeunique = true, kind=:left)
business_avg_reg.b_b[ismissing.(business_avg_reg.b_b)] .= 0
business_avg_reg = @linq business_avg_reg |>
           transform(new = :b_b .+ mu_hat)
df_reg_user_new = @linq df_reg_user |>
    groupby(:user_id) |>
    transform(b_u = :value1_sum/(:n_u.+l))
df_reg_user_new = df_reg_user_new[:, [:user_id, :b_u]]
f = vcat(df_reg_user_new[:b_u]...)
df_reg_user_new[:b_u] = f[:,]
avg_reg = join(business_avg_reg, df_reg_user_new, on = :user_id, makeunique = true, kind=:left)
avg_reg.b_u[ismissing.(avg_reg.b_u)] .= 0
avg_reg = @linq avg_reg |>
      transform(new = :b_b .+ mu_hat .+:b_u)
avg_reg.new[avg_reg[:new].>5,:] .= 5
avg_reg.new[avg_reg[:new].<0,:] .= 0

print(sqrt(mean((avg_reg[:stars] - avg_reg[:new]).^2)))

print("  test MAE:   ")
print(mean(abs.(avg_reg[:stars] - avg_reg[:new])))

print("    train RMSE:   ")
l = argmin(rmse3)

train = DataFrame(train)

df_reg_bus_new = @linq df_reg_bus |>
            groupby(:business_id) |>
            transform(b_b = :value0_sum/(:n_b.+l))
df_reg_bus_new = df_reg_bus_new[:, [:business_id, :b_b]]
f = vcat(df_reg_bus_new[:b_b]...)
df_reg_bus_new[:b_b] = f[:,]
business_avg_reg = join(train, df_reg_bus_new, on = :business_id, makeunique = true, kind=:left)
business_avg_reg.b_b[ismissing.(business_avg_reg.b_b)] .= 0
business_avg_reg = @linq business_avg_reg |>
           transform(new = :b_b .+ mu_hat)
df_reg_user_new = @linq df_reg_user |>
    groupby(:user_id) |>
    transform(b_u = :value1_sum/(:n_u.+l))
df_reg_user_new = df_reg_user_new[:, [:user_id, :b_u]]
f = vcat(df_reg_user_new[:b_u]...)
df_reg_user_new[:b_u] = f[:,]
avg_reg = join(business_avg_reg, df_reg_user_new, on = :user_id, makeunique = true, kind=:left)
avg_reg.b_u[ismissing.(avg_reg.b_u)] .= 0
avg_reg = @linq avg_reg |>
      transform(new = :b_b .+ mu_hat .+:b_u)
avg_reg.new[avg_reg[:new].>5,:] .= 5
avg_reg.new[avg_reg[:new].<0,:] .= 0

rmse3 = sqrt(mean((avg_reg[:stars] - avg_reg[:new]).^2))
print(rmse3)

print("  train MAE:   ")
print(mean(abs.(avg_reg[:stars] - avg_reg[:new])))
