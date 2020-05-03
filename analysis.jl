cd("C:/Users/Uzytkownik/Documents/Julia/praca")
using JLD2, FileIO, DataFrames, Statistics, CSV
@load "prepared.jld2" reviews business business

mean(reviews[:stars])
std(reviews[:stars])
median(reviews[:stars])
mo

unique(reviews[:business_id])
unique(reviews[:business_id])

bus_stats = by(reviews, :business_id, nrow)
sort!(bus_stats, :x1, rev = true)
CSV.write("bus_stats.csv", bus_stats, delim = ";")

nrow(bus_stats[(bus_stats[:x1].<5), :])
nrow(bus_stats[(bus_stats[:x1].<10), :]) - nrow(bus_stats[(bus_stats[:x1].<5), :])
nrow(bus_stats[(bus_stats[:x1].<30), :]) - nrow(bus_stats[(bus_stats[:x1].<10), :])
nrow(bus_stats[(bus_stats[:x1].<100), :]) - nrow(bus_stats[(bus_stats[:x1].<30), :])
nrow(bus_stats[(bus_stats[:x1].>=100), :])

business_means = by(reviews, :business_id, :stars => mean)
sort!(business_means, :stars_mean, rev = true)
CSV.write("business_means.csv", business_means, delim = ";")
sort!(business_means, :stars_mean)
CSV.write("business_means.csv", business_means, delim = ";")

nrow(business_means[(business_means[:stars_mean].<2), :])
nrow(business_means[(business_means[:stars_mean].<3), :]) - nrow(business_means[(business_means[:stars_mean].<2), :])
nrow(business_means[(business_means[:stars_mean].<4), :]) - nrow(business_means[(business_means[:stars_mean].<3), :])
nrow(business_means[(business_means[:stars_mean].<5), :]) - nrow(business_means[(business_means[:stars_mean].<4), :])
nrow(business_means[(business_means[:stars_mean].>=5), :])

nrow(reviews[(reviews[:stars].<2), :])
nrow(reviews[(reviews[:stars].<3), :]) - nrow(reviews[(reviews[:stars].<2), :])
nrow(reviews[(reviews[:stars].<4), :]) - nrow(reviews[(reviews[:stars].<3), :])
nrow(reviews[(reviews[:stars].<5), :]) - nrow(reviews[(reviews[:stars].<4), :])
nrow(reviews[(reviews[:stars].>=5), :])

bus_stats = by(reviews, :business_id, nrow)

nrow(bus_stats[(bus_stats[:x1].<5), :])
nrow(bus_stats[(bus_stats[:x1].<10), :]) - nrow(bus_stats[(bus_stats[:x1].<5), :])
nrow(bus_stats[(bus_stats[:x1].<30), :]) - nrow(bus_stats[(bus_stats[:x1].<10), :])
nrow(bus_stats[(bus_stats[:x1].<100), :]) - nrow(bus_stats[(bus_stats[:x1].<30), :])
nrow(bus_stats[(bus_stats[:x1].>=100), :])

business_means = by(reviews, :business_id, :stars => mean)

nrow(business_means[(business_means[:stars_mean].<2), :])
nrow(business_means[(business_means[:stars_mean].<3), :]) - nrow(business_means[(business_means[:stars_mean].<2), :])
nrow(business_means[(business_means[:stars_mean].<4), :]) - nrow(business_means[(business_means[:stars_mean].<3), :])
nrow(business_means[(business_means[:stars_mean].<5), :]) - nrow(business_means[(business_means[:stars_mean].<4), :])
nrow(business_means[(business_means[:stars_mean].>=5), :])

merged = join(reviews, business, on = :business_id, makeunique = true)

stats_count =by(merged, :state, nrow)

stats_mean =by(merged, :state, :stars => mean)

stats_states = join(stats_count, stats_mean, on = :state, makeunique = true)
sort!(stats_states, :x1, rev = true)

stats_states[:frequent] = (stats_states[:x1] / 5039561) * 100

CSV.write("stats_states.csv", stats_states, delim = ";", decimal = ',')
