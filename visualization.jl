cd("C:/Users/Uzytkownik/Documents/Julia/praca")
using JLD2, FileIO, DataFrames, Statistics, CSV, Plotly, StatsPlots, Plots
@load "prepared.jld2" reviews business
gr()
histogram(reviews[:stars], bins=5, legend=false, fill=1)
yaxis!("Częstość")
xaxis!("Ocena")
Plots.savefig("histogram_ocen.png")

users_means = by(reviews, :user_id, :stars => mean)

histogram(users_means[:stars_mean], bins=5, legend=false, fill=1)
yaxis!("Liczba użytkowników")
xaxis!("Średnia ocena")
Plots.savefig("histogram_ocen_uzytkow.png")

business_means = by(reviews, :business_id, :stars => mean)

histogram(business_means[:stars_mean], bins=5, legend=false, fill=1)
yaxis!("Liczba restauracji")
xaxis!("Średnia ocena")
Plots.savefig("histogram_ocen_firm.png")
