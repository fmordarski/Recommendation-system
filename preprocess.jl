cd("C:/Users/Uzytkownik/Documents/Julia/praca")
using Random,Recommendation,MLDataUtils,DataFrames, JSON, LazyJSON, Mmap, Statistics, GMT, Libdl,CSV,SparseArrays, Serialization
using DataFramesMeta, JLD2, FileIO
# load reviews dataset
reviews = DataFrame()

open("review.json") do f

    while !eof(f)

        j = JSON.parse(readline(f))

        push!(reviews, (user_id=j["user_id"],

                        business_id=j["business_id"],

                        stars=j["stars"]))

    end

end

categorical!(reviews)

# load business dataset

business=DataFrame()

open("business.json") do f

    while !eof(f)

        j = JSON.parse(readline(f))

        push!(business, (business_id=j["business_id"],
                         name = j["name"],
                         city=j["city"],
                         state=j["state"],
                         latitude=j["latitude"],
                         longitude=j["longitude"],
                         stars=j["stars"],
                         review_count=j["review_count"]))


    end

end

categorical!(business)

user=DataFrame()

open("user.json") do f

    while !eof(f)

        j = JSON.parse(readline(f))

        push!(user, (user_id=j["user_id"],
                         name = j["name"],
                         review_count=j["review_count"]))


    end

end

categorical!(user)

unique!(reviews)

subset = reviews[nonunique(reviews, [:user_id, :business_id]),
                                    [:user_id, :business_id, :stars]]

subset = by(subset, [:user_id, :business_id], :stars => mean)
reviews = join(reviews, subset, on= [:user_id, :business_id], kind= :left)

reviews[.!ismissing.(reviews[:stars_mean]), :stars] = reviews[.!ismissing.(reviews[:stars_mean]), :stars_mean]

unique!(reviews)

reviews = reviews[:, 1:3]

unique(reviews[:user_id])
unique(reviews[:business_id])

mean(reviews[:stars])

user_stats=by(reviews, :user_id)

# gdf = groupby(reviews, :user_id)
# dobre = [nrow(g) >= 3 for g in gdf]
# DataFrame(gdf[dobre])

reviews = reviews[∈(user_stats[(user_stats[:x1].>=3),
                :user_id]).(reviews.user_id),:]
bus_stats=by(reviews, :business_id, nrow)
reviews = reviews[∈(bus_stats[(bus_stats[:x1].>1), :business_id]).(reviews.business_id),:]
user_stats=by(reviews, :user_id, nrow)
reviews = reviews[∈(user_stats[(user_stats[:x1].>=3),:user_id]).(reviews.user_id),:]
bus_stats=by(reviews, :business_id, nrow)
reviews = reviews[∈(bus_stats[(bus_stats[:x1].>1), :business_id]).(reviews.business_id),:]
user_stats=by(reviews, :user_id, nrow)
reviews = reviews[∈(user_stats[(user_stats[:x1].>=3),:user_id]).(reviews.user_id),:]

@save "prepared.jld2" reviews user business
